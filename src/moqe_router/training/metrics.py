"""Request-level validation metrics for expert selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RoutingMetrics:
    top1_correct: int
    routing_regret_sum: float
    sample_count: int

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / self.sample_count

    @property
    def mean_routing_regret(self) -> float:
        return self.routing_regret_sum / self.sample_count


def routing_metrics(logits: Tensor, expert_losses: Tensor) -> RoutingMetrics:
    if logits.ndim != 2 or logits.shape != expert_losses.shape:
        raise ValueError("logits and expert_losses must have the same [batch, experts] shape")
    if logits.shape[0] < 1:
        raise ValueError("metrics require at least one sample")
    predicted = logits.argmax(dim=-1)
    oracle = expert_losses.argmin(dim=-1)
    selected_losses = expert_losses.gather(1, predicted[:, None]).squeeze(1)
    regrets = selected_losses - expert_losses.min(dim=-1).values
    return RoutingMetrics(
        top1_correct=int((predicted == oracle).sum().item()),
        routing_regret_sum=float(regrets.sum().item()),
        sample_count=logits.shape[0],
    )
