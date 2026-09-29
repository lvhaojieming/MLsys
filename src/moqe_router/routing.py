"""Compose L1 quality scores and L2 physical routing into one request decision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import EmbeddingRouter
from .physical import PhysicalRouter, PoolRegistry, Replica, RouteUnavailable


@dataclass(frozen=True)
class RoutingDecision:
    model_family: str
    expert_id: str
    replica: Replica
    registry_epoch: int
    architecture_version: str
    expert_ranking: tuple[str, ...]


class TwoStageRouter:
    def __init__(
        self,
        quality_router: EmbeddingRouter,
        registry: PoolRegistry,
        physical_router: PhysicalRouter | None = None,
    ) -> None:
        self.quality_router = quality_router
        self.registry = registry
        self.physical_router = physical_router or PhysicalRouter()

    @torch.inference_mode()
    def route(
        self,
        embeddings: Tensor,
        attention_mask: Tensor,
        max_new_tokens: Tensor,
        *,
        original_prompt_lengths: Tensor | None = None,
    ) -> RoutingDecision:
        if embeddings.shape[0] != 1:
            raise ValueError("route() handles one request; batch the quality model separately")
        config = self.quality_router.config
        snapshot = self.registry.snapshot()
        visible_length = int(attention_mask[0].sum().item())
        required_context = visible_length + int(max_new_tokens[0].item())
        if original_prompt_lengths is not None:
            required_context = int(original_prompt_lengths[0].item()) + int(
                max_new_tokens[0].item()
            )
        eligible = snapshot.eligible_experts(config.model_family, required_context)
        if not eligible.intersection(config.expert_ids):
            raise RouteUnavailable("no compatible READY expert")

        self.quality_router.eval()
        logits = self.quality_router(
            embeddings,
            attention_mask,
            max_new_tokens,
            original_prompt_lengths=original_prompt_lengths,
        )[0]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("quality router produced non-finite logits")
        ranking = tuple(
            config.expert_ids[index]
            for index in torch.argsort(logits, descending=True).tolist()
        )
        for expert_id in ranking:
            if expert_id not in eligible:
                continue
            try:
                replica = self.physical_router.choose(
                    snapshot,
                    model_family=config.model_family,
                    expert_id=expert_id,
                    required_context_tokens=required_context,
                )
            except RouteUnavailable:
                continue
            return RoutingDecision(
                model_family=config.model_family,
                expert_id=expert_id,
                replica=replica,
                registry_epoch=snapshot.epoch,
                architecture_version=config.architecture_version,
                expert_ranking=ranking,
            )
        raise RouteUnavailable("all ranked experts lost their READY replicas")
