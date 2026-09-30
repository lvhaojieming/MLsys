"""MoQE Router architecture, offline training, and routing primitives."""

from .config import RouterArchitecture
from .physical import PoolRegistry, Replica, ReplicaState, RouteUnavailable

__all__ = ["RouterArchitecture", "PoolRegistry", "Replica", "ReplicaState", "RouteUnavailable"]
