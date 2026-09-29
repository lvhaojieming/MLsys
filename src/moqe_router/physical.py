"""L2 physical routing over versioned, ready replicas.

This is deliberately a feasibility and placement layer. It contains no queue
prices, predicted latency, or cross-expert quality decisions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum


class RouteUnavailable(RuntimeError):
    """No compatible ready replica exists for the requested expert."""


class ReplicaState(str, Enum):
    LOADING = "LOADING"
    WARMING = "WARMING"
    READY = "READY"
    DRAINING = "DRAINING"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class Replica:
    model_family: str
    expert_id: str
    pool_id: str
    replica_id: str
    endpoint: str
    max_context_tokens: int
    state: ReplicaState
    checkpoint_version: str

    def __post_init__(self) -> None:
        if not all(
            (self.model_family, self.expert_id, self.pool_id, self.replica_id, self.endpoint)
        ):
            raise ValueError("replica identity and endpoint fields are required")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        if not isinstance(self.state, ReplicaState):
            raise ValueError("state must be a ReplicaState")
        if not self.checkpoint_version:
            raise ValueError("checkpoint_version is required")


@dataclass(frozen=True)
class RegistrySnapshot:
    epoch: int
    replicas: tuple[Replica, ...]

    def candidates(
        self, model_family: str, expert_id: str, required_context_tokens: int
    ) -> tuple[Replica, ...]:
        return tuple(
            replica
            for replica in self.replicas
            if replica.model_family == model_family
            and replica.expert_id == expert_id
            and replica.state is ReplicaState.READY
            and replica.max_context_tokens >= required_context_tokens
        )

    def eligible_experts(self, model_family: str, required_context_tokens: int) -> set[str]:
        return {
            replica.expert_id
            for replica in self.replicas
            if replica.model_family == model_family
            and replica.state is ReplicaState.READY
            and replica.max_context_tokens >= required_context_tokens
        }


class PoolRegistry:
    """Atomically publish whole snapshots from a separate deployment controller."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = RegistrySnapshot(epoch=0, replicas=())

    def snapshot(self) -> RegistrySnapshot:
        with self._lock:
            return self._snapshot

    def publish(self, epoch: int, replicas: tuple[Replica, ...]) -> RegistrySnapshot:
        if epoch < 1:
            raise ValueError("epoch must be positive")
        replicas = tuple(replicas)
        ids = [replica.replica_id for replica in replicas]
        if len(ids) != len(set(ids)):
            raise ValueError("replica IDs must be globally unique")
        endpoints = [replica.endpoint for replica in replicas]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("replica endpoints must be globally unique")
        pools: dict[str, tuple[str, str, str]] = {}
        experts: dict[tuple[str, str], str] = {}
        for replica in replicas:
            identity = (replica.model_family, replica.expert_id, replica.checkpoint_version)
            if replica.pool_id in pools and pools[replica.pool_id] != identity:
                raise ValueError("a pool cannot mix model families, experts or checkpoint versions")
            pools[replica.pool_id] = identity
            expert_key = (replica.model_family, replica.expert_id)
            if expert_key in experts and experts[expert_key] != replica.checkpoint_version:
                raise ValueError("an expert ID cannot refer to multiple checkpoint versions")
            experts[expert_key] = replica.checkpoint_version
        with self._lock:
            if epoch <= self._snapshot.epoch:
                raise ValueError("registry epoch must increase")
            self._snapshot = RegistrySnapshot(epoch=epoch, replicas=replicas)
            return self._snapshot


class PhysicalRouter:
    """Choose one pool, then one replica, with local round-robin counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool_counters: dict[tuple[str, str], int] = {}
        self._replica_counters: dict[str, int] = {}

    def choose(
        self,
        snapshot: RegistrySnapshot,
        *,
        model_family: str,
        expert_id: str,
        required_context_tokens: int,
    ) -> Replica:
        if required_context_tokens < 1:
            raise ValueError("required_context_tokens must be positive")
        candidates = snapshot.candidates(model_family, expert_id, required_context_tokens)
        if not candidates:
            raise RouteUnavailable(f"no READY replica for {expert_id}")
        pools: dict[str, list[Replica]] = {}
        for replica in candidates:
            pools.setdefault(replica.pool_id, []).append(replica)
        pool_ids = sorted(pools)
        with self._lock:
            pool_key = (model_family, expert_id)
            pool_index = self._pool_counters.get(pool_key, 0)
            pool_id = pool_ids[pool_index % len(pool_ids)]
            self._pool_counters[pool_key] = pool_index + 1
            replica_list = sorted(pools[pool_id], key=lambda item: item.replica_id)
            replica_index = self._replica_counters.get(pool_id, 0)
            chosen = replica_list[replica_index % len(replica_list)]
            self._replica_counters[pool_id] = replica_index + 1
            return chosen
