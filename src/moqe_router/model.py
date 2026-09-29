"""L1 quality router operating strictly *after* a frozen token embedding layer.

The module returns one expert logit vector per request. It contains no language
model, teacher, loss function, or per-token routing decision.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import RouterArchitecture


class EmbeddingRouter(nn.Module):
    """Encode the first, middle and last prompt regions and score experts.

    ``embeddings`` are outputs of the frozen base model's *token embedding*
    layer, shaped ``[batch, padded_tokens, embedding_dim]``. ``attention_mask``
    is right-padded (ones followed by zeros). No expert model is executed here.
    """

    def __init__(self, config: RouterArchitecture) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        width = config.tokens_per_region

        self.input_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden),
            nn.LayerNorm(hidden),
        )
        self.position_embedding = nn.Embedding(width, hidden)
        self.region_embedding = nn.Embedding(3, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=2 * hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.encoder_layers, enable_nested_tensor=False
        )
        self.pool_score = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(3 * hidden + 2),
            nn.Linear(3 * hidden + 2, hidden),
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

    def _three_regions(self, embeddings: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return [B, 3, W, D] and its validity mask without Python batch loops."""
        batch, tokens, _ = embeddings.shape
        width = self.config.tokens_per_region
        lengths = mask.sum(dim=1)
        zeros = torch.zeros_like(lengths)
        starts = torch.stack(
            (zeros, ((lengths - width) // 2).clamp_min(0), (lengths - width).clamp_min(0)),
            dim=1,
        )
        indices = starts.unsqueeze(-1) + torch.arange(width, device=embeddings.device)
        valid = indices < lengths[:, None, None]
        safe_indices = indices.clamp_max(tokens - 1)
        batch_indices = torch.arange(batch, device=embeddings.device)[:, None, None]
        regions = embeddings[batch_indices, safe_indices]
        return regions.masked_fill(~valid.unsqueeze(-1), 0), valid

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor,
        max_new_tokens: Tensor,
        *,
        original_prompt_lengths: Tensor | None = None,
    ) -> Tensor:
        """Return raw logits in ``config.expert_ids`` order, shape ``[B, M]``.

        ``original_prompt_lengths`` is optional metadata for a caller that has
        already shortened the sequence before embedding extraction. The first,
        middle and last windows must still be preserved by that caller.
        """
        self._validate_inputs(
            embeddings, attention_mask, max_new_tokens, self.config.embedding_dim
        )
        mask = attention_mask.bool()
        lengths = mask.sum(dim=1)
        if original_prompt_lengths is None:
            original_prompt_lengths = lengths
        elif (
            original_prompt_lengths.shape != lengths.shape
            or original_prompt_lengths.device != lengths.device
            or bool((original_prompt_lengths < lengths).any())
        ):
            raise ValueError("original_prompt_lengths must be [batch] and >= visible lengths")

        regions, valid = self._three_regions(embeddings, mask)
        batch, _, width, _ = regions.shape
        x = self.input_projection(regions.to(self.input_projection[0].weight.dtype))
        positions = torch.arange(width, device=x.device)
        region_ids = torch.arange(3, device=x.device)
        x = x + self.position_embedding(positions)[None, None, :, :]
        x = x + self.region_embedding(region_ids)[None, :, None, :]

        x = x.reshape(batch, 3 * width, self.config.hidden_dim)
        flat_valid = valid.reshape(batch, 3 * width)
        x = self.encoder(x, src_key_padding_mask=~flat_valid)
        x = x.reshape(batch, 3, width, self.config.hidden_dim)
        attention = self.pool_score(x).squeeze(-1).masked_fill(~valid, float("-inf"))
        attention = attention.softmax(dim=-1)
        pooled = (attention.unsqueeze(-1) * x).sum(dim=2).flatten(start_dim=1)

        length_feature = torch.log1p(original_prompt_lengths.to(x.dtype)) / math.log1p(
            self.config.length_scale
        )
        budget_feature = torch.log1p(max_new_tokens.to(x.dtype)) / math.log1p(
            self.config.generation_scale
        )
        features = torch.cat(
            (pooled, length_feature[:, None], budget_feature[:, None]), dim=1
        )
        return self.head(features)
