# COGS 188 Platformer Project

Public repository: [ManjuDaBeast/COGS188_Platformer](https://github.com/ManjuDaBeast/COGS188_Platformer)

This project studies course-taught AI methods on a seed-deterministic
procedural platformer. The Gymnasium environment has a normalized
42-dimensional observation, eight discrete movement actions, and easy,
medium, and hard level tiers. A seeded reset makes a level reproducible for
paired evaluation.

## Course-aligned project scope

The allowed methods are restricted to the supplied COGS 188 lecture PDFs:

- **Uninformed search:** breadth-first search, depth-first search, and
  uniform-cost search.
- **Heuristic search:** greedy best-first search and A*.
- **Local search:** hill climbing and simulated annealing.
- **Bandits:** UCB action selection.
- **Dynamic programming:** value iteration and policy iteration.
- **Monte Carlo:** episode-based policy evaluation and control.
- **MDP / stochasticity / observability:** problem formulation and analysis,
  not competing algorithms.

The [revised proposal](./PROJECT_PROPOSAL.md) analyzes how these families fit
the platformer, separates methods that learn from methods that plan with a
model, defines the proposed evaluation, and lists the assumptions and limits
of each abstraction. Methods that need a finite state representation will use
a documented abstraction; they will not be described as exact solutions to
the original continuous-state problem.

The first four search methods have playable demonstrations on a documented
platform-transition abstraction. The other lecture-method implementations
and the full benchmark remain proposed work; these replays are demonstrations,
not comparative evidence.

## Current environment and checks

The environment implementation is in [`platformer/env.py`](./platformer/env.py).
Install the existing dependencies and run the current regression tests with:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

The environment supports replay visualization for the simple reference
policies and the four search demonstrations.

## Lecture search replays

Open the [replay index](./experiments/lecture_replays/index.html) for
individual BFS, UCS, greedy best-first, and A* replays. All four use easy
level seed 6 and complete the same episode. The search nodes are platforms;
edges are estimated feasible jumps. The replay highlights the route and shows
the search method's node expansions and estimated route cost.

These are demonstrations, not a performance comparison: on this level the
abstract graph has only one route, so the four methods choose the same
platform sequence. The planner uses the level model, and a simple simulator
state-based controller executes its route. This is not a learned,
observation-only policy.

To generate another replay locally:

```bash
python3 -m platformer visualize --algorithm astar --difficulty easy --seed 6 \
  --output experiments/lecture_replays/astar.html
```

## Proposal formats

- [Markdown proposal](./PROJECT_PROPOSAL.md)
- [PDF proposal](./PROJECT_PROPOSAL.pdf)
- [Editable DOCX for Google Docs](./PROJECT_PROPOSAL.docx)

The DOCX can be uploaded to Google Drive and opened with Google Docs.
