"""Versioned architecture configuration; no training hyperparameters belong here."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RouterArchitecture:
    model_family: str
    expert_ids: tuple[str, ...]
    embedding_dim: int
    architecture_version: str = "embedding-router-v1"
    hidden_dim: int = 256
    num_heads: int = 4
    encoder_layers: int = 1
    tokens_per_region: int = 128
    dropout: float = 0.1
    length_scale: int = 8192
    generation_scale: int = 2048

    def __post_init__(self) -> None:
        if not self.model_family or not self.architecture_version:
            raise ValueError("model_family and architecture_version are required")
        if not self.expert_ids or len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("expert_ids must be a nonempty ordered list without duplicates")
        if any(not expert_id for expert_id in self.expert_ids):
            raise ValueError("expert_ids cannot contain empty strings")
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if self.num_heads < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.encoder_layers < 1 or self.tokens_per_region < 1:
            raise ValueError("encoder_layers and tokens_per_region must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.length_scale <= 0 or self.generation_scale <= 0:
            raise ValueError("length scales must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "RouterArchitecture":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["expert_ids"] = tuple(data["expert_ids"])
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
