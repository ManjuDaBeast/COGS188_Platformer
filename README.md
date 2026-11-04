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

The repository is being narrowed to the supplied lecture-method scope. The
course-method implementations and their new benchmark are proposed work in
[`PROJECT_PROPOSAL.md`](./PROJECT_PROPOSAL.md), not yet completed results.

## Current environment and checks

The environment implementation is in [`platformer/env.py`](./platformer/env.py).
Install the existing dependencies and run the current regression tests with:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

The environment supports replay visualization for inspecting a single
episode with either of the simple reference policies.

## Proposal formats

- [Markdown proposal](./PROJECT_PROPOSAL.md)
- [PDF proposal](./PROJECT_PROPOSAL.pdf)
- [Editable DOCX for Google Docs](./PROJECT_PROPOSAL.docx)

The DOCX can be uploaded to Google Drive and opened with Google Docs.
