"""Course search methods on an abstract platform-transition graph."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Literal

import numpy as np

from platformer.env import Platform, PlatformerEnv


SearchMethod = Literal[
    "bfs",
    "dfs",
    "ucs",
    "greedy",
    "astar",
    "hill_climbing",
    "simulated_annealing",
    "value_iteration",
    "policy_iteration",
]


@dataclass(frozen=True)
class SearchResult:
    method: SearchMethod
    path: tuple[int, ...]
    expanded: int
    cost: float


class PlatformSearchPolicy:
    """Search platform centers, then steer the simulator along the chosen route.

    The search graph is an explicit abstraction: nodes are platforms and edges
    are estimated feasible jumps. The low-level controller uses simulator state
    to execute those jumps; this is a model-informed planning demonstration,
    not a policy learned from observation-only interaction.
    """

    MAX_JUMP_GAP = 4.2
    MAX_RISE = 1.65
    MAX_DROP = 3.0
    GLIDE_GAP = 2.4

    def __init__(self, method: SearchMethod, *, expansion_limit: int = 10_000) -> None:
        if method not in (
            "bfs",
            "dfs",
            "ucs",
            "greedy",
            "astar",
            "hill_climbing",
            "simulated_annealing",
            "value_iteration",
            "policy_iteration",
        ):
            raise ValueError(f"unknown search method: {method}")
        if expansion_limit <= 0:
            raise ValueError("expansion_limit must be positive")
        self.method = method
        self.expansion_limit = expansion_limit
        self.env: PlatformerEnv | None = None
        self.result: SearchResult | None = None
        self._route_position = 0

    def prepare(self, env: PlatformerEnv) -> SearchResult:
        """Build the level graph and search a route from the start to goal."""
        self.env = env
        count = len(env.platforms)
        centers = [(item.left + item.right) / 2.0 for item in env.platforms]
        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
        for source_index, source in enumerate(env.platforms):
            for target_index in range(source_index + 1, count):
                target = env.platforms[target_index]
                gap = target.left - source.right
                rise = target.height_at(0.0) - source.height_at(0.0)
                if gap > self.MAX_JUMP_GAP or rise > self.MAX_RISE or rise < -self.MAX_DROP:
                    continue
                horizontal_cost = centers[target_index] - centers[source_index]
                vertical_cost = 0.5 * abs(rise)
                obstacle_cost = self._landing_risk(target)
                adjacency[source_index].append(
                    (target_index, horizontal_cost + vertical_cost + obstacle_cost)
                )

        if self.method in ("hill_climbing", "simulated_annealing"):
            self.result = self._local_search(adjacency, centers)
        elif self.method in ("value_iteration", "policy_iteration"):
            self.result = self._dynamic_programming(adjacency, centers)
        else:
            self.result = search_platform_graph(
                adjacency,
                centers,
                start=0,
                goal=count - 1,
                method=self.method,
                expansion_limit=self.expansion_limit,
            )
        self._route_position = 0
        return self.result

    def _local_search(
        self, adjacency: list[list[tuple[int, float]]], centers: list[float]
    ) -> SearchResult:
        assert self.env is not None
        method = self.method
        seed = int(self.env._last_seed or 0)
        rng = np.random.default_rng(seed)

        def random_route() -> tuple[int, ...] | None:
            route = [0]
            while route[-1] < len(adjacency) - 1:
                options = adjacency[route[-1]]
                if not options:
                    return None
                route.append(options[int(rng.integers(len(options)))][0])
            return tuple(route)

        costs = {
            (left, right): cost
            for left, edges in enumerate(adjacency)
            for right, cost in edges
        }

        def score(route: tuple[int, ...]) -> float:
            return sum(costs[(a, b)] for a, b in zip(route, route[1:]))

        current = random_route()
        if current is None:
            raise ValueError(f"{method} found no route through the platform graph")
        best = current
        best_score = score(best)
        expansions = 0
        for iteration in range(self.expansion_limit):
            candidate = random_route()
            expansions += 1
            if candidate is None:
                continue
            candidate_score = score(candidate)
            if candidate_score < best_score:
                best, best_score = candidate, candidate_score
            if method == "hill_climbing":
                if candidate_score < score(current):
                    current = candidate
            else:
                temperature = max(0.01, 1.0 - iteration / self.expansion_limit)
                delta = candidate_score - score(current)
                if delta < 0 or rng.random() < math.exp(-delta / temperature):
                    current = candidate
        return SearchResult(method, best, expansions, best_score)

    def _dynamic_programming(
        self, adjacency: list[list[tuple[int, float]]], centers: list[float]
    ) -> SearchResult:
        count = len(adjacency)
        goal = count - 1
        values = [0.0] * count
        policy = [None] * count
        gamma = 0.95
        sweeps = 0

        if self.method == "value_iteration":
            for sweeps in range(1, self.expansion_limit + 1):
                largest_change = 0.0
                updated = values.copy()
                for node in range(goal - 1, -1, -1):
                    if adjacency[node]:
                        updated[node] = max(
                            -cost + gamma * values[target]
                            for target, cost in adjacency[node]
                        )
                    largest_change = max(largest_change, abs(updated[node] - values[node]))
                values = updated
                if largest_change < 1e-10:
                    break
            for node in range(goal):
                if adjacency[node]:
                    policy[node] = max(
                        adjacency[node],
                        key=lambda edge: -edge[1] + gamma * values[edge[0]],
                    )[0]
        else:
            for node in range(goal):
                if adjacency[node]:
                    policy[node] = adjacency[node][0][0]
            for sweeps in range(1, self.expansion_limit + 1):
                for node in range(goal - 1, -1, -1):
                    target = policy[node]
                    if target is not None:
                        edge_cost = next(
                            cost for candidate, cost in adjacency[node] if candidate == target
                        )
                        values[node] = -edge_cost + gamma * values[target]
                stable = True
                for node in range(goal):
                    if not adjacency[node]:
                        continue
                    best_target = max(
                        adjacency[node],
                        key=lambda edge: -edge[1] + gamma * values[edge[0]],
                    )[0]
                    if best_target != policy[node]:
                        policy[node] = best_target
                        stable = False
                if stable:
                    break

        route = [0]
        route_cost = 0.0
        while route[-1] != goal:
            node = route[-1]
            target = policy[node]
            if target is None or target in route:
                raise ValueError(f"{self.method} produced no route to the goal")
            route_cost += next(
                cost for candidate, cost in adjacency[node] if candidate == target
            )
            route.append(target)
        return SearchResult(self.method, tuple(route), sweeps * count, route_cost)

    def _landing_risk(self, platform: Platform) -> float:
        assert self.env is not None
        left = platform.left
        right = platform.right
        risk = sum(left <= hazard.x <= right for hazard in self.env.hazards)
        risk += sum(
            left <= enemy.origin_x <= right for enemy in self.env.enemies
        )
        return 2.0 * risk

    @property
    def visualization_metadata(self) -> dict[str, object]:
        if self.result is None:
            return {}
        return {
            "method": self.method.upper(),
            "planned_route": list(self.result.path),
            "expanded_nodes": self.result.expanded,
            "estimated_path_cost": round(self.result.cost, 3),
            "abstraction": (
                "Platforms are nodes; forward edges represent jumps within "
                f"{self.MAX_JUMP_GAP:g} horizontal units, up to {self.MAX_RISE:g} "
                f"units upward and {self.MAX_DROP:g} downward."
            ),
        }

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        del observation, deterministic
        if self.env is None or self.result is None:
            raise RuntimeError("prepare(env) must be called before requesting an action")

        env = self.env
        route = self.result.path
        if env.on_ground and self._route_position + 1 < len(route):
            target_index = route[self._route_position + 1]
            target = env.platforms[target_index]
            if target.left - 0.25 <= env.x <= target.right + 0.25:
                self._route_position += 1

        if self._route_position + 1 >= len(route):
            return 2 if env.x < env.goal_x else 0

        source = env.platforms[route[self._route_position]]
        target = env.platforms[route[self._route_position + 1]]
        target_center = (target.left + target.right) / 2.0
        if env.on_ground:
            if env.x < source.right - 0.65:
                return 2
            return 5

        horizontal_error = target_center - env.x
        direction = 1 if horizontal_error > 0.45 else -1 if horizontal_error < -0.45 else 0
        if (
            env.vy < -0.2
            and env.glide_available
            and target.left - source.right >= self.GLIDE_GAP
        ):
            return 7 if direction >= 0 else 6
        return 2 if direction > 0 else 1 if direction < 0 else 0


def search_platform_graph(
    adjacency: list[list[tuple[int, float]]],
    positions: list[float],
    *,
    start: int,
    goal: int,
    method: SearchMethod,
    expansion_limit: int = 10_000,
) -> SearchResult:
    """Run an uninformed or heuristic lecture search method on a weighted graph."""
    if len(adjacency) != len(positions):
        raise ValueError("adjacency and positions must have the same number of nodes")
    if not (0 <= start < len(adjacency) and 0 <= goal < len(adjacency)):
        raise ValueError("start and goal must be valid graph nodes")
    if expansion_limit <= 0:
        raise ValueError("expansion_limit must be positive")

    if start == goal:
        return SearchResult(method, (start,), 0, 0.0)

    def heuristic(node: int) -> float:
        return max(0.0, positions[goal] - positions[node])

    frontier: list[tuple[float, int, float]] = []
    sequence = itertools.count()
    initial_priority = (
        heuristic(start) if method in ("greedy", "astar") else 0.0
    )
    heapq.heappush(frontier, (initial_priority, next(sequence), start))
    parent: dict[int, int | None] = {start: None}
    path_cost = {start: 0.0}
    expanded = 0
    settled: set[int] = set()
    if method == "dfs":
        stack = [start]
        parents: dict[int, int | None] = {start: None}
        seen = {start}
        expanded = 0
        while stack:
            node = stack.pop()
            if node == goal:
                path = []
                cursor: int | None = goal
                while cursor is not None:
                    path.append(cursor)
                    cursor = parents[cursor]
                path.reverse()
                return SearchResult(
                    method,
                    tuple(path),
                    expanded,
                    float(len(path) - 1),
                )
            if expanded >= expansion_limit:
                raise RuntimeError(
                    f"DFS exceeded its {expansion_limit}-node expansion limit"
                )
            expanded += 1
            for neighbor, _ in reversed(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    parents[neighbor] = node
                    stack.append(neighbor)
        raise ValueError(f"DFS found no path from node {start} to node {goal}")

    while frontier:
        priority, _, node = heapq.heappop(frontier)
        expected_priority = (
            path_cost[node]
            if method == "ucs"
            else heuristic(node)
            if method == "greedy"
            else path_cost[node] + heuristic(node)
            if method == "astar"
            else path_cost[node]
        )
        if priority != expected_priority or node in settled:
            continue
        settled.add(node)
        current_cost = path_cost[node]
        if node == goal:
            path = []
            cursor: int | None = goal
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            path.reverse()
            return SearchResult(method, tuple(path), expanded, current_cost)
        if expanded >= expansion_limit:
            raise RuntimeError(
                f"{method.upper()} exceeded its {expansion_limit}-node expansion limit"
            )
        expanded += 1

        for neighbor, edge_cost in adjacency[node]:
            if edge_cost < 0:
                raise ValueError("search edge costs must be nonnegative")
            candidate_cost = current_cost + (1.0 if method == "bfs" else edge_cost)
            if method == "bfs":
                if neighbor in parent:
                    continue
                parent[neighbor] = node
                path_cost[neighbor] = candidate_cost
                heapq.heappush(frontier, (candidate_cost, next(sequence), neighbor))
                continue
            if candidate_cost >= path_cost.get(neighbor, float("inf")):
                continue
            parent[neighbor] = node
            path_cost[neighbor] = candidate_cost
            if method == "ucs":
                next_priority = candidate_cost
            elif method == "greedy":
                next_priority = heuristic(neighbor)
            else:
                next_priority = candidate_cost + heuristic(neighbor)
            heapq.heappush(frontier, (next_priority, next(sequence), neighbor))

    raise ValueError(f"{method.upper()} found no path from node {start} to node {goal}")
