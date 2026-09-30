"""Offline request-level training components for the L1 quality router."""

from .data import RequestDataset, TrainingExample, collate_requests
from .embedding import FrozenEmbeddingProvider
from .metrics import RoutingMetrics, routing_metrics
from .objective import build_loss_aware_targets, soft_target_cross_entropy

__all__ = [
    "FrozenEmbeddingProvider",
    "RequestDataset",
    "RoutingMetrics",
    "TrainingExample",
    "build_loss_aware_targets",
    "collate_requests",
    "routing_metrics",
    "soft_target_cross_entropy",
]
