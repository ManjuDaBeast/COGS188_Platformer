"""Generate same-seed level images and course-method replay galleries."""

from __future__ import annotations

import html
import json
from pathlib import Path

from platformer.course_agents import MonteCarloControlPolicy, UCBPolicy
from platformer.env import PlatformerEnv
from platformer.search import PlatformSearchPolicy
from platformer.visualize import create_episode_visualization


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "lecture_replays"
LEVEL_SEEDS = {"easy": 6, "medium": 33, "hard": 6}
METHODS = {
    "bfs": "Breadth-first search",
    "dfs": "Depth-first search",
    "ucs": "Uniform-cost search",
    "greedy": "Greedy best-first search",
    "astar": "A*",
    "value-iteration": "Value iteration",
    "policy-iteration": "Policy iteration",
    "monte-carlo": "Monte Carlo control",
    "hill-climbing": "Hill climbing",
    "simulated-annealing": "Simulated annealing",
    "ucb": "UCB bandit",
}


def _level_svg(env: PlatformerEnv, difficulty: str, seed: int) -> str:
    platforms = env.platforms
    min_x = min(item.left for item in platforms) - 1
    max_x = env.goal_x + 2
    min_y = min(item.y - item.amplitude for item in platforms) - 1
    max_y = max(item.y + item.amplitude for item in platforms) + 2
    width, height = 1400, 480
    scale_x = (width - 100) / (max_x - min_x)
    scale_y = (height - 150) / max(max_y - min_y, 4)

    def x_pos(value: float) -> float:
        return 50 + (value - min_x) * scale_x

    def y_pos(value: float) -> float:
        return height - 90 - (value - min_y) * scale_y

    shapes = [
        f'<rect width="{width}" height="{height}" fill="#15243a"/>',
        f'<text x="44" y="42" fill="#f3f6fb" font-size="26" font-family="system-ui" font-weight="700">{difficulty.title()} level · seed {seed}</text>',
        f'<text x="46" y="70" fill="#bdcbe0" font-size="15" font-family="system-ui">Full level overview · platforms, gems, hazards, enemies, and goal</text>',
    ]
    for platform in platforms:
        px = x_pos(platform.left)
        py = y_pos(platform.y)
        pw = platform.width * scale_x
        fill = "#53a7a3" if platform.moving else "#5fc79c"
        shapes.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="15" rx="4" fill="{fill}"/>'
        )
        if platform.moving:
            top = y_pos(platform.y + platform.amplitude)
            bottom = y_pos(platform.y - platform.amplitude)
            shapes.append(
                f'<line x1="{px + pw / 2:.1f}" y1="{top:.1f}" x2="{px + pw / 2:.1f}" y2="{bottom:.1f}" stroke="#89d9d3" stroke-width="2" stroke-dasharray="6 5"/>'
            )
        shapes.append(
            f'<text x="{px + pw / 2:.1f}" y="{py + 34:.1f}" fill="#d5e5f3" font-size="13" text-anchor="middle" font-family="system-ui">P{platforms.index(platform)}</text>'
        )
    for gem in env.gems:
        gx, gy = x_pos(gem.x), y_pos(gem.y)
        shapes.append(
            f'<polygon points="{gx:.1f},{gy - 8:.1f} {gx + 7:.1f},{gy:.1f} {gx:.1f},{gy + 8:.1f} {gx - 7:.1f},{gy:.1f}" fill="#66d9ff" stroke="#d7f7ff" stroke-width="1"/>'
        )
    for hazard in env.hazards:
        hx, hy = x_pos(hazard.x), y_pos(hazard.y)
        shapes.append(
            f'<polygon points="{hx - 10:.1f},{hy + 9:.1f} {hx:.1f},{hy - 12:.1f} {hx + 10:.1f},{hy + 9:.1f}" fill="#f25f5c"/>'
        )
    for enemy in env.enemies:
        ex, ey = x_pos(enemy.origin_x), y_pos(enemy.y)
        shapes.append(
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="9" fill="#f69b52"/>'
        )
        shapes.append(
            f'<line x1="{x_pos(enemy.origin_x - enemy.amplitude):.1f}" y1="{ey + 16:.1f}" x2="{x_pos(enemy.origin_x + enemy.amplitude):.1f}" y2="{ey + 16:.1f}" stroke="#f69b52" stroke-width="2" stroke-dasharray="4 4"/>'
        )
    start = platforms[0]
    shapes.append(
        f'<circle cx="{x_pos(start.left + 1):.1f}" cy="{y_pos(start.y + 0.5):.1f}" r="11" fill="#73a7ff" stroke="#e0eaff" stroke-width="2"/>'
    )
    goal_x, goal_y = x_pos(env.goal_x), y_pos(env.goal_y)
    shapes.extend(
        [
            f'<line x1="{goal_x:.1f}" y1="{goal_y - 40:.1f}" x2="{goal_x:.1f}" y2="{goal_y + 3:.1f}" stroke="#f5f2d0" stroke-width="4"/>',
            f'<polygon points="{goal_x + 2:.1f},{goal_y - 39:.1f} {goal_x + 28:.1f},{goal_y - 31:.1f} {goal_x + 2:.1f},{goal_y - 23:.1f}" fill="#f2c14e"/>',
            f'<text x="45" y="{height - 25}" fill="#c6d3e2" font-size="14" font-family="system-ui">Blue: start · Green: static platform · Teal: moving platform · Cyan: gem · Red: hazard · Orange: moving enemy · Yellow: goal</text>',
        ]
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{difficulty} level overview">'
        + "".join(shapes)
        + "</svg>"
    )


def _make_policy(method: str, seed: int):
    if method == "ucb":
        return UCBPolicy()
    if method == "monte-carlo":
        return MonteCarloControlPolicy(seed=seed)
    search_names = {
        "bfs": "bfs",
        "dfs": "dfs",
        "ucs": "ucs",
        "greedy": "greedy",
        "astar": "astar",
        "value-iteration": "value_iteration",
        "policy-iteration": "policy_iteration",
        "hill-climbing": "hill_climbing",
        "simulated-annealing": "simulated_annealing",
    }
    return PlatformSearchPolicy(search_names[method])


def _write_index(summaries: list[dict[str, object]]) -> None:
    cards = []
    for difficulty in LEVEL_SEEDS:
        seed = LEVEL_SEEDS[difficulty]
        image_extension = (
            "png"
            if (OUTPUT / "levels" / f"{difficulty}.png").exists()
            else "svg"
        )
        cards.append(
            f'<section><h2>{difficulty.title()} level · seed {seed}</h2>'
            f'<a href="levels/{difficulty}.{image_extension}"><img class="level" src="levels/{difficulty}.{image_extension}" alt="{difficulty} level overview"></a>'
            "<div class=\"grid\">"
        )
        for item in summaries:
            if item["difficulty"] != difficulty:
                continue
            method = str(item["method"])
            status = "Completed" if item["completed"] else "Did not complete"
            cards.append(
                f'<a class="replay" href="{difficulty}/{method}.html">'
                f'<strong>{html.escape(METHODS[method])}</strong>'
                f'<span>{status} · {item["steps"]} steps · {item["gems_collected"]}/{item["gems_available"]} gems</span>'
                "</a>"
            )
        cards.append("</div></section>")

    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>COGS 188 Platformer: Lecture Methods by Difficulty</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101725;color:#edf3fa}
body{max-width:1180px;margin:32px auto;padding:0 20px;line-height:1.5}
h1{line-height:1.15}section{margin:34px 0;padding:20px;border:1px solid #40516a;border-radius:12px;background:#18243a}
.level{width:100%;height:auto;border-radius:8px;border:1px solid #40516a}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin-top:16px}
.replay{display:flex;flex-direction:column;gap:4px;padding:12px;border:1px solid #465a75;border-radius:8px;color:inherit;text-decoration:none;background:#202f49}
.replay:hover{border-color:#ffe66d}.replay strong{color:#ffe66d}.replay span{font-size:.88rem;color:#c6d3e2}
.note{padding:14px 18px;border-left:4px solid #65d6b1;background:#18243a}
</style></head><body>
<h1>Lecture Models on Easy, Medium, and Hard Levels</h1>
<p>Each difficulty uses one fixed, seeded level shared by all methods. Click a level image for a larger view, or open any replay to inspect the agent's attempt frame by frame.</p>
<div class="note"><strong>Interpretation:</strong> BFS/DFS/UCS/greedy/A* search platform routes; value/policy iteration solve a small approximate graph MDP; hill climbing and simulated annealing optimize candidate routes; Monte Carlo control trains on seeded episodes; UCB selects actions without using state. Their information and compute differ, so these one-level replays illustrate behavior, not a fair ranking or estimate of generalization.</div>
"""
    page += "".join(cards)
    page += '<p><a href="../../README.md">Project README</a></p></body></html>'
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    (OUTPUT / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "levels").mkdir(exist_ok=True)
    summaries: list[dict[str, object]] = []
    for difficulty, seed in LEVEL_SEEDS.items():
        env = PlatformerEnv(difficulty=difficulty, max_episode_steps=350)
        try:
            env.reset(seed=seed)
            (OUTPUT / "levels" / f"{difficulty}.svg").write_text(
                _level_svg(env, difficulty, seed), encoding="utf-8"
            )
        finally:
            env.close()

        (OUTPUT / difficulty).mkdir(exist_ok=True)
        for method in METHODS:
            env = PlatformerEnv(difficulty=difficulty, max_episode_steps=350)
            try:
                summary = create_episode_visualization(
                    env,
                    _make_policy(method, seed),
                    seed=seed,
                    output_path=OUTPUT / difficulty / f"{method}.html",
                    algorithm=METHODS[method],
                )
                summaries.append({"method": method, **summary})
                print(
                    f"{difficulty:6} {method:22} "
                    f"complete={summary['completed']} steps={summary['steps']} "
                    f"gems={summary['gems_collected']}/{summary['gems_available']}"
                )
            finally:
                env.close()
    _write_index(summaries)
    print(f"Replay gallery: {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
