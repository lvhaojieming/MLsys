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
    relative_losses = expert_losses - expert_losses.min(dim=-1, keepdim=True).values
    return F.softmax(-relative_losses / temperature, dim=-1)


def soft_target_cross_entropy(logits: Tensor, target_probs: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape != target_probs.shape:
        raise ValueError("logits and target_probs must have the same [batch, experts] shape")
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()
