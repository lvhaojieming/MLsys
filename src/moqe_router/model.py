"""Full-token L1 router after a frozen base-model token embedding layer.

Every valid prompt token contributes to one sequence-level expert logit vector.
There is no teacher, expert forward pass, or per-token routing decision here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import RouterArchitecture


class EmbeddingRouter(nn.Module):
    """Two-level hierarchical encoding with exactly two Transformer layers.

    Layer 1 attends locally over *every* token in nonoverlapping chunks. Layer 2
    attends globally over all chunk summaries. This avoids the T-squared cost
    of full attention on a long prompt while retaining every token's influence.
    """

    def __init__(self, config: RouterArchitecture) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden),
            nn.LayerNorm(hidden),
        )

        def encoder_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config.num_heads,
                dim_feedforward=2 * hidden,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.local_encoder = nn.TransformerEncoder(
            encoder_layer(), num_layers=1, enable_nested_tensor=False
        )
        self.global_encoder = nn.TransformerEncoder(
            encoder_layer(), num_layers=1, enable_nested_tensor=False
        )
        self.token_pool_score = nn.Linear(hidden, 1)
        self.chunk_pool_score = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden + 2),
            nn.Linear(hidden + 2, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, len(config.expert_ids)),
        )

    @staticmethod
    def _validate_inputs(
        embeddings: Tensor, attention_mask: Tensor, max_new_tokens: Tensor, embedding_dim: int
    ) -> None:
        if embeddings.ndim != 3 or embeddings.shape[2] != embedding_dim:
            raise ValueError("embeddings must have shape [batch, tokens, embedding_dim]")
        batch, tokens, _ = embeddings.shape
        if batch < 1 or tokens < 1:
            raise ValueError("batch and token dimensions must be nonempty")
        if attention_mask.shape != (batch, tokens):
            raise ValueError("attention_mask must have shape [batch, tokens]")
        if max_new_tokens.shape != (batch,):
            raise ValueError("max_new_tokens must have shape [batch]")
        if embeddings.device != attention_mask.device or embeddings.device != max_new_tokens.device:
            raise ValueError("all inputs must be on the same device")
        if not torch.is_floating_point(embeddings):
            raise ValueError("embeddings must be floating point")
        if bool(((attention_mask != 0) & (attention_mask != 1)).any()):
            raise ValueError("attention_mask values must be 0 or 1")
        mask = attention_mask.bool()
        if bool((~mask.any(dim=1)).any()):
            raise ValueError("each prompt must contain at least one token")
        if bool(((~mask[:, :-1]) & mask[:, 1:]).any()):
            raise ValueError("attention_mask must be right-padded")
        if bool((max_new_tokens < 0).any()):
            raise ValueError("max_new_tokens cannot be negative")

    @staticmethod
    def _position_encoding(length: int, hidden: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Sinusoidal positions support any prompt/chunk count without a learned cap."""
        positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        half = hidden // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phases = positions * frequencies[None, :]
        encoding = torch.zeros(length, hidden, device=device, dtype=torch.float32)
        encoding[:, 0 : 2 * half : 2] = phases.sin()
        encoding[:, 1 : 2 * half : 2] = phases.cos()
        return encoding.to(dtype)

    def forward(self, embeddings: Tensor, attention_mask: Tensor, max_new_tokens: Tensor) -> Tensor:
        """Return one `[M]` raw-logit vector per sequence, in expert ID order.

        ``embeddings`` must cover the *entire* visible prompt. No head/middle/
        tail sampling, truncation, or precomputed summary is accepted.
        """
        self._validate_inputs(
            embeddings, attention_mask, max_new_tokens, self.config.embedding_dim
        )
        mask = attention_mask.bool()
        lengths = mask.sum(dim=1)
        batch, tokens, _ = embeddings.shape
        hidden = self.config.hidden_dim
        chunk_size = self.config.chunk_size

        x = self.input_projection(embeddings.to(self.input_projection[0].weight.dtype))
        x = x + self._position_encoding(tokens, hidden, x.device, x.dtype)[None, :, :]

        padding = (-tokens) % chunk_size
        if padding:
            x = F.pad(x, (0, 0, 0, padding))
            mask = F.pad(mask, (0, padding), value=False)
        chunk_count = x.shape[1] // chunk_size
        local = x.reshape(batch * chunk_count, chunk_size, hidden)
        local_valid = mask.reshape(batch * chunk_count, chunk_size)
        active_chunks = local_valid.any(dim=1)

        # A short request can leave whole trailing chunks empty for other batch
        # members. Give those chunks one dummy key to avoid all-masked softmax;
        # discard their summaries immediately afterwards.
        safe_local_valid = local_valid.clone()
        safe_local_valid[:, 0] |= ~active_chunks
        local = self.local_encoder(local, src_key_padding_mask=~safe_local_valid)
        token_scores = self.token_pool_score(local).squeeze(-1)
        token_weights = token_scores.masked_fill(~safe_local_valid, float("-inf")).softmax(-1)
        chunks = (token_weights.unsqueeze(-1) * local).sum(dim=1)
        chunks = chunks.masked_fill(~active_chunks[:, None], 0)
        chunks = chunks.reshape(batch, chunk_count, hidden)
        chunk_valid = active_chunks.reshape(batch, chunk_count)

        chunks = chunks + self._position_encoding(
            chunk_count, hidden, chunks.device, chunks.dtype
        )[None, :, :]
        chunks = self.global_encoder(chunks, src_key_padding_mask=~chunk_valid)
        chunk_scores = self.chunk_pool_score(chunks).squeeze(-1)
        chunk_weights = chunk_scores.masked_fill(~chunk_valid, float("-inf")).softmax(-1)
        pooled = (chunk_weights.unsqueeze(-1) * chunks).sum(dim=1)

        length_feature = torch.log1p(lengths.to(pooled.dtype)) / math.log1p(
            self.config.length_scale
        )
        budget_feature = torch.log1p(max_new_tokens.to(pooled.dtype)) / math.log1p(
            self.config.generation_scale
        )
        features = torch.cat(
            (pooled, length_feature[:, None], budget_feature[:, None]), dim=1
        )
        return self.head(features)
