"""Episode-based Monte Carlo control and a state-independent UCB baseline."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from platformer.env import PlatformerEnv


class UCBPolicy:
    """UCB action selection with one reward estimate per action."""

    def __init__(self, actions: int = 8) -> None:
        self.counts = np.zeros(actions, dtype=np.int64)
        self.values = np.zeros(actions, dtype=np.float64)
        self.total = 0
        self._last_action: int | None = None

    @property
    def visualization_metadata(self) -> dict[str, object]:
        return {
            "method": "UCB",
            "training": "Online action-reward updates during this episode; no state conditioning.",
            "action_counts": self.counts.tolist(),
        }

    def prepare(self, env: PlatformerEnv) -> None:
        del env

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        del observation, deterministic
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            self._last_action = int(untried[0])
        else:
            bonus = np.sqrt(2.0 * math.log(max(self.total, 1)) / self.counts)
            self._last_action = int(np.argmax(self.values + bonus))
        return self._last_action

    def observe(self, reward: float) -> None:
        if self._last_action is None:
            raise RuntimeError("act() must be called before observe()")
        action = self._last_action
        self.total += 1
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]


class MonteCarloControlPolicy:
    """First-visit Monte Carlo control over coarse observation features."""

    def __init__(
        self,
        *,
        episodes: int = 12,
        seed: int = 0,
        epsilon: float = 0.2,
        gamma: float = 0.99,
    ) -> None:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        self.episodes = episodes
        self.seed = seed
        self.epsilon = epsilon
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.returns: defaultdict[tuple[tuple[int, ...], int], list[float]] = defaultdict(list)
        self.values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(8, dtype=np.float64)
        )
        self.trained_difficulty: str | None = None
        self.training_steps = 0

    @staticmethod
    def state_key(observation: np.ndarray) -> tuple[int, ...]:
        ground = int(observation[4] > 0.5)
        velocity = int(np.clip(np.floor((observation[2] + 1.0) * 2), 0, 3))
        nearest_dx = float(observation[6]) * 20.0
        distance = int(np.clip(np.floor((nearest_dx + 5.0) / 2.5), 0, 7))
        height = int(np.clip(np.floor((float(observation[7]) + 1.0) * 2), 0, 3))
        return ground, velocity, distance, height

    @property
    def visualization_metadata(self) -> dict[str, object]:
        return {
            "method": "Monte Carlo control",
            "training": (
                f"First-visit episode-return updates; {self.episodes} training episodes "
                f"on {self.trained_difficulty} levels."
            ),
            "training_steps": self.training_steps,
            "visited_states": len(self.values),
        }

    def prepare(self, env: PlatformerEnv) -> None:
        if self.trained_difficulty == env.difficulty:
            return
        training_env = PlatformerEnv(
            max_episode_steps=min(env.max_episode_steps, 300),
            reward_shaping=env.reward_shaping,
            difficulty=env.difficulty,
            layout_variant=env.layout_variant,
            hazards_enabled=env.hazards_enabled,
            enemies_enabled=env.enemies_enabled,
        )
        self.trained_difficulty = env.difficulty
        try:
            for episode in range(self.episodes):
                observation, _ = training_env.reset(
                    seed=self.seed + 10_000 + episode
                )
                trajectory: list[
                    tuple[tuple[int, ...], int, float]
                ] = []
                while True:
                    action = self._exploratory_action(observation)
                    next_observation, reward, terminated, truncated, _ = training_env.step(
                        action
                    )
                    trajectory.append((self.state_key(observation), action, reward))
                    self.training_steps += 1
                    observation = next_observation
                    if terminated or truncated:
                        break
                discounted_returns = [0.0] * len(trajectory)
                total_return = 0.0
                for index in range(len(trajectory) - 1, -1, -1):
                    total_return = trajectory[index][2] + self.gamma * total_return
                    discounted_returns[index] = total_return
                first_visit: set[tuple[tuple[int, ...], int]] = set()
                for (state, action, _), episode_return in zip(
                    trajectory, discounted_returns
                ):
                    pair = (state, action)
                    if pair in first_visit:
                        continue
                    first_visit.add(pair)
                    self.returns[pair].append(episode_return)
                    self.values[state][action] = float(np.mean(self.returns[pair]))
        finally:
            training_env.close()

    def _exploratory_action(self, observation: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(8))
        return int(np.argmax(self.values[self.state_key(observation)]))

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        if deterministic:
            return int(np.argmax(self.values[self.state_key(observation)]))
        return self._exploratory_action(observation)
