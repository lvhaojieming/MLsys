"""Strict JSONL data loading and dynamic right-padding for Router training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    input_ids: tuple[int, ...]
    max_new_tokens: int
    expert_losses: tuple[float, ...]


class RequestDataset(Dataset[TrainingExample]):
    """Load request labels whose loss positions follow ``expert_ids`` exactly.

    Rows may include an ``expert_ids`` field. When present, it must exactly
    equal the configured sequence and provides an explicit order check. Without
    it, the positional ``expert_losses`` list is interpreted in configured
    order and is never sorted or remapped.
    """

    def __init__(
        self,
        path: str | Path,
        expert_ids: tuple[str, ...],
        max_prompt_tokens: int,
    ) -> None:
        if not expert_ids:
            raise ValueError("expert_ids cannot be empty")
        if max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")

        self.path = Path(path)
        self.examples: list[TrainingExample] = []
        seen_ids: set[str] = set()
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    example = self._parse_row(
                        row,
                        expert_ids=expert_ids,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                    if example.sample_id in seen_ids:
                        raise ValueError(f"duplicate id: {example.sample_id!r}")
                    seen_ids.add(example.sample_id)
                    self.examples.append(example)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
        if not self.examples:
            raise ValueError(f"{self.path}: no training examples")

    @staticmethod
    def _parse_row(
        row: Any,
        *,
        expert_ids: tuple[str, ...],
        max_prompt_tokens: int,
    ) -> TrainingExample:
        if not isinstance(row, dict):
            raise ValueError("each JSONL row must be an object")

        sample_id = row["id"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("id must be a nonempty string")

        input_ids = row["input_ids"]
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or any(type(token) is not int or token < 0 for token in input_ids)
        ):
            raise ValueError("input_ids must be a nonempty list of nonnegative integers")
        if len(input_ids) > max_prompt_tokens:
            raise ValueError(
                f"sample {sample_id!r} has {len(input_ids)} prompt tokens, "
                f"exceeding max_prompt_tokens={max_prompt_tokens}"
            )

        max_new_tokens = row["max_new_tokens"]
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")

        losses = row["expert_losses"]
        if not isinstance(losses, list) or len(losses) != len(expert_ids):
            raise ValueError(
                f"expert_losses must be a list of length {len(expert_ids)} "
                "in RouterArchitecture.expert_ids order"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in losses
        ):
            raise ValueError("expert_losses must contain only finite numbers")

        declared_experts = row.get("expert_ids")
        if declared_experts is not None and declared_experts != list(expert_ids):
            raise ValueError(
                "row expert_ids must exactly match RouterArchitecture.expert_ids order"
            )

        return TrainingExample(
            sample_id=sample_id,
            input_ids=tuple(input_ids),
            max_new_tokens=max_new_tokens,
            expert_losses=tuple(float(value) for value in losses),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


def collate_requests(examples: list[TrainingExample]) -> dict[str, Tensor | list[str]]:
    """Dynamically right-pad one batch while preserving sample IDs."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples)
    token_count = max(len(example.input_ids) for example in examples)
    input_ids = torch.zeros(batch_size, token_count, dtype=torch.int64)
    attention_mask = torch.zeros(batch_size, token_count, dtype=torch.bool)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.int64)
        attention_mask[row, :length] = True

    return {
        "ids": [example.sample_id for example in examples],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": torch.tensor(
            [example.max_new_tokens for example in examples], dtype=torch.int64
        ),
        "expert_losses": torch.tensor(
            [example.expert_losses for example in examples], dtype=torch.float32
        ),
    }
