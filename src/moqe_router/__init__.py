"""MoQE routing primitives. Embedding extraction and model serving live upstream."""

from .config import RouterArchitecture
from .physical import PoolRegistry, Replica, ReplicaState, RouteUnavailable

__all__ = ["RouterArchitecture", "PoolRegistry", "Replica", "ReplicaState", "RouteUnavailable"]
