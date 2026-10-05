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


def loss_aware_terms(logits: Tensor, expert_losses: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
    """Return per-sequence soft-target CE and unit sample weights.

    Unit weights preserve the distributed real-sample mask/reduction interface.
    Expert loss gaps affect the soft target, never the sample's contribution.
    """
    if logits.ndim != 2 or logits.shape != expert_losses.shape:
        raise ValueError("logits and expert_losses must have matching [batch, experts] shapes")
    targets = build_loss_aware_targets(expert_losses, temperature)
    ce = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(-1)
    return ce, torch.ones_like(ce)


def loss_aware_router_loss(logits: Tensor, expert_losses: Tensor, temperature: float = 0.1) -> Tensor:
    ce, _ = loss_aware_terms(logits, expert_losses, temperature)
    return ce.mean()


def accuracy_aware_terms(
    logits: Tensor,
    expert_losses: Tensor,
    expert_correctness: Tensor,
    temperature: float = 0.1,
    nll_aux_weight: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-sequence task-error CE plus a small NLL soft-target auxiliary term.

    Correctness must be 0/1 in architecture expert order, or -1/-1 to denote
    missing task labels. When both experts share an outcome, NLL soft targets
    provide the only expert preference signal. Every example receives unit
    weight; loss gaps never rescale sample contributions.
    """
    if logits.ndim != 2 or logits.shape != expert_losses.shape or logits.shape != expert_correctness.shape:
        raise ValueError("logits, losses, and correctness must have matching [batch, experts] shapes")
    if not math.isfinite(nll_aux_weight) or nll_aux_weight < 0:
        raise ValueError("nll_aux_weight must be finite and nonnegative")
    known = (expert_correctness >= 0).all(dim=-1)
    if bool(((expert_correctness < 0).any(dim=-1) & (expert_correctness >= 0).any(dim=-1)).any()):
        raise ValueError("correctness labels must mark every expert or none")
    if bool((expert_correctness > 1).any()):
        raise ValueError("correctness values must be 0/1 or -1 for missing")

    nll_targets = build_loss_aware_targets(expert_losses, temperature)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll_ce = -(nll_targets * log_probs).sum(-1)

    outcomes = expert_correctness.float().clamp_min(0)
    outcome_total = outcomes.sum(-1, keepdim=True)
    known_targets = outcomes / outcome_total.clamp_min(1)
    # If both experts are correct or both are wrong, accuracy alone cannot
    # prefer one. Use sequence NLL to break that tie.
    accuracy_targets = torch.where(outcome_total > 0, known_targets, nll_targets)
    correctness_ce = -(accuracy_targets * log_probs).sum(-1)
    task_ce = torch.where(known, correctness_ce, nll_ce)
    return task_ce + nll_aux_weight * nll_ce, correctness_ce, nll_ce, known
