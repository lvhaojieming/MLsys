"""Frozen base-model token embedding loading without constructing the LLM."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn


class FrozenEmbeddingProvider(nn.Module):
    """A permanently frozen token embedding loaded from one safetensors key."""

    def __init__(self, weight: Tensor, embedding_dim: int) -> None:
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] != embedding_dim:
            raise ValueError(
                f"embedding weight must have shape [vocab, {embedding_dim}], "
                f"got {tuple(weight.shape)}"
            )
        if not torch.is_floating_point(weight):
            raise ValueError("embedding weight must be floating point")
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True)
        self.embedding.weight.requires_grad_(False)
        self.eval()

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    @property
    def num_embeddings(self) -> int:
        return self.embedding.num_embeddings

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        *,
        weight_key: str,
        embedding_dim: int,
    ) -> "FrozenEmbeddingProvider":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required for Router training") from exc

        root = Path(model_path)
        if not root.is_dir():
            raise FileNotFoundError(f"base_model_path is not a directory: {root}")
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                shard_name = index["weight_map"][weight_key]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{weight_key!r} is missing from {index_path}") from exc
            tensor_path = root / shard_name
        else:
            tensor_path = root / "model.safetensors"

        if not tensor_path.is_file():
            raise FileNotFoundError(f"embedding safetensors file not found: {tensor_path}")
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if weight_key not in handle.keys():
                raise ValueError(f"{weight_key!r} is missing from {tensor_path}")
            weight = handle.get_tensor(weight_key)
        return cls(weight, embedding_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("input_ids must have dtype torch.int32 or torch.int64")
        if input_ids.numel() and (
            bool((input_ids < 0).any())
            or bool((input_ids >= self.num_embeddings).any())
        ):
            raise ValueError("input_ids contain a token outside the embedding vocabulary")
        with torch.no_grad():
            return self.embedding(input_ids)
