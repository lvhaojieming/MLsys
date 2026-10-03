"""Validation cadence and best-only checkpoint selection for online NPU training."""
from dataclasses import dataclass
import math


@dataclass
class BestCheckpointPolicy:
    interval: int = 5000
    best_regret: float = float('inf')
    last_validation_step: int = 0

    def __post_init__(self):
        if type(self.interval) is not int or self.interval < 1:
            raise ValueError('validation interval must be a positive integer')

    def due(self, step: int, *, final: bool = False) -> bool:
        return step > self.last_validation_step and (final or step % self.interval == 0)

    def observe(self, step: int, regret: float) -> bool:
        if not math.isfinite(regret):
            raise ValueError('validation regret must be finite')
        if step <= self.last_validation_step:
            raise ValueError('validation steps must increase')
        self.last_validation_step = step
        if regret < self.best_regret:
            self.best_regret = regret
            return True
        return False
