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

Playable demonstrations are available for the course search, local-search,
dynamic-programming, Monte Carlo, and bandit methods. These are illustrative
single episodes, not a held-out benchmark or a performance ranking.

## Current environment and checks

The environment implementation is in [`platformer/env.py`](./platformer/env.py).
Install the existing dependencies and run the current regression tests with:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

The environment supports replay visualization for simple reference policies
and course-method demonstrations.

## Lecture-method level gallery

View the rendered [hosted level and replay gallery](https://manjudabest.github.io/COGS188_Platformer/)
to see full-level images for easy, medium, and hard, and an individual replay
for every lecture method on each difficulty. Each difficulty uses one fixed
seed shared across methods. The gallery reports completion, episode length,
and collected gems; open a replay to inspect the attempt frame by frame.

The GitHub repository's file view shows HTML source code. Use the hosted
gallery link above for the rendered visuals.

The methods use different information: search and dynamic programming plan
over an abstract platform graph, local search optimizes candidate graph
routes, Monte Carlo control trains on seeded episodes, and UCB chooses actions
without conditioning on state. The route planners also use a simple
simulator-state controller to execute the selected route. Their replays
illustrate assumptions and behavior, not an apples-to-apples performance
comparison or evidence of generalization.

Regenerate the complete gallery locally:

```bash
python3 -m scripts.generate_lecture_replays
```

## Proposal formats

- [Markdown proposal](./PROJECT_PROPOSAL.md)
- [PDF proposal](./PROJECT_PROPOSAL.pdf)
- [Editable DOCX for Google Docs](./PROJECT_PROPOSAL.docx)

The DOCX can be uploaded to Google Drive and opened with Google Docs.
