"""Command-line evaluation and replay entry points for the platformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from platformer.agents import HeuristicPolicy, Policy, RandomPolicy
from platformer.course_agents import MonteCarloControlPolicy, UCBPolicy
from platformer.env import PlatformerEnv
from platformer.evaluation import evaluate_policy
from platformer.search import PlatformSearchPolicy, SearchMethod
from platformer.visualize import create_episode_visualization


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_policy(name: str, seed: int) -> Policy:
    if name == "random":
        return RandomPolicy(seed)
    if name == "heuristic":
        return HeuristicPolicy()
    methods: dict[str, SearchMethod] = {
        "bfs": "bfs",
        "dfs": "dfs",
        "ucs": "ucs",
        "greedy": "greedy",
        "astar": "astar",
        "hill-climbing": "hill_climbing",
        "simulated-annealing": "simulated_annealing",
        "value-iteration": "value_iteration",
        "policy-iteration": "policy_iteration",
    }
    if name in methods:
        return PlatformSearchPolicy(methods[name])
    if name == "ucb":
        return UCBPolicy()
    if name == "monte-carlo":
        return MonteCarloControlPolicy(seed=seed)
    raise ValueError(f"unsupported reference policy: {name}")


def _add_environment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=500)


def evaluate(args: argparse.Namespace) -> None:
    env = PlatformerEnv(
        max_episode_steps=args.max_episode_steps,
        reward_shaping=not args.no_shaping,
        difficulty=args.difficulty,
    )
    policy = _make_policy(args.algorithm, args.seed)
    try:
        summary = evaluate_policy(
            env,
            policy,
            seeds=range(args.seed_start, args.seed_start + args.episodes),
        )
        summary["algorithm"] = args.algorithm
        print(json.dumps(summary, indent=2))
        if args.output:
            _write_json(args.output, summary)
    finally:
        env.close()


def visualize(args: argparse.Namespace) -> None:
    env = PlatformerEnv(
        max_episode_steps=args.max_episode_steps,
        reward_shaping=not args.no_shaping,
        difficulty=args.difficulty,
    )
    policy = _make_policy(args.algorithm, args.seed)
    try:
        summary = create_episode_visualization(
            env,
            policy,
            seed=args.seed,
            output_path=args.output,
            algorithm=args.algorithm,
        )
        print(json.dumps({**summary, "visualization": str(args.output)}, indent=2))
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="platformer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    eval_parser = subparsers.add_parser(
        "evaluate", help="evaluate a simple reference policy on fixed level seeds"
    )
    eval_parser.add_argument(
        "--algorithm",
        choices=(
            "random",
            "heuristic",
            "bfs",
            "dfs",
            "ucs",
            "greedy",
            "astar",
            "hill-climbing",
            "simulated-annealing",
            "value-iteration",
            "policy-iteration",
            "monte-carlo",
            "ucb",
        ),
        required=True,
    )
    eval_parser.add_argument("--episodes", type=int, default=20)
    eval_parser.add_argument("--seed-start", type=int, default=1_000)
    eval_parser.add_argument("--output", type=Path)
    eval_parser.add_argument(
        "--difficulty", choices=("easy", "medium", "hard"), default="medium"
    )
    _add_environment_options(eval_parser)
    eval_parser.set_defaults(func=evaluate)

    visualize_parser = subparsers.add_parser(
        "visualize", help="save a playable HTML replay of a reference policy"
    )
    visualize_parser.add_argument(
        "--algorithm",
        choices=(
            "random",
            "heuristic",
            "bfs",
            "dfs",
            "ucs",
            "greedy",
            "astar",
            "hill-climbing",
            "simulated-annealing",
            "value-iteration",
            "policy-iteration",
            "monte-carlo",
            "ucb",
        ),
        default="heuristic",
    )
    visualize_parser.add_argument(
        "--output", type=Path, default=Path("platformer_episode.html")
    )
    visualize_parser.add_argument(
        "--difficulty", choices=("easy", "medium", "hard"), default="medium"
    )
    _add_environment_options(visualize_parser)
    visualize_parser.set_defaults(func=visualize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
