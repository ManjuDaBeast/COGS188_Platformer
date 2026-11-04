"""Simple reference policies for environment testing and evaluation."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int: ...


class RandomPolicy:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        del observation, deterministic
        return int(self.rng.integers(0, 8))


class HeuristicPolicy:
    """Run toward the goal, jumping near gaps and visible threats."""

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        del deterministic
        on_ground = observation[4] > 0.5
        glide_available = observation[5] > 0.5
        goal_dx = float(observation[40])
        right_edge_dx = None
        next_platform_dx = None
        for i in range(4):
            offset = 6 + 4 * i
            dx, width = float(observation[offset]), float(observation[offset + 2])
            if width <= 0.0:
                continue
            edge = dx * 20.0 + width * 2.5
            if edge >= -0.5 and (right_edge_dx is None or edge < right_edge_dx):
                right_edge_dx = edge
            if dx > 0 and (next_platform_dx is None or dx < next_platform_dx):
                next_platform_dx = dx
        should_jump = on_ground and right_edge_dx is not None and right_edge_dx < 1.8
        for start in (22, 28):
            for i in range(2):
                dx = float(observation[start + i * 3])
                kind = float(observation[start + i * 3 + 2])
                if kind > 0 and 0.0 < dx < 0.07:
                    should_jump = on_ground
        if should_jump and (next_platform_dx is None or next_platform_dx < 0.29):
            return 5
        if not on_ground and observation[3] < -0.1 and glide_available:
            return 7 if goal_dx > 0 else 6
        return 2 if goal_dx > 0.01 else 1
