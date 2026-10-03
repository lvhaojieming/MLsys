"""The single supported loss-aware soft-target objective."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def build_loss_aware_targets(expert_losses: Tensor, temperature: float) -> Tensor:
    if expert_losses.ndim != 2 or expert_losses.shape[1] < 1:
        raise ValueError("expert_losses must have shape [batch, experts]")
    if not torch.is_floating_point(expert_losses):
        raise ValueError("expert_losses must be floating point")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not bool(torch.isfinite(expert_losses).all()):
        raise ValueError("expert_losses must be finite")
    expert_losses = expert_losses.detach().float()
    relative_losses = expert_losses - expert_losses.min(dim=-1, keepdim=True).values
    return F.softmax(-relative_losses / temperature, dim=-1)


def soft_target_cross_entropy(logits: Tensor, target_probs: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape != target_probs.shape:
        raise ValueError("logits and target_probs must have the same [batch, experts] shape")
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def gap_weighted_terms(logits: Tensor, expert_losses: Tensor, temperature: float,
                       alpha: float = 2.0, gap_scale: float = 0.1) -> tuple[Tensor, Tensor]:
    """Return per-sequence CE and bounded, detached weights for two experts."""
    if logits.ndim != 2 or logits.shape != expert_losses.shape or logits.shape[1] != 2:
        raise ValueError("gap weighting requires matching [batch, 2] tensors")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not math.isfinite(gap_scale) or gap_scale <= 0:
        raise ValueError("gap_scale must be finite and positive")
    targets = build_loss_aware_targets(expert_losses, temperature)
    losses = expert_losses.detach().float()
    gap = (losses[:, 0] - losses[:, 1]).abs()
    weights = 1.0 + alpha * gap / (gap + gap_scale)
    ce = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(-1)
    return ce, weights


def gap_weighted_router_loss(logits: Tensor, expert_losses: Tensor, temperature: float = 0.1,
                             alpha: float = 2.0, gap_scale: float = 0.1) -> Tensor:
    ce, weights = gap_weighted_terms(logits, expert_losses, temperature, alpha, gap_scale)
    return (ce * weights).sum() / weights.sum()
