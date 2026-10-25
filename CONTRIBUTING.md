# Contributing to PandaForge

Thanks for considering a contribution. This project is a portfolio-grade
reference implementation of a robotics motion-planning stack, so the bar for
changes is: **correct, measured, and honest**. The rules below exist because
the project's value is its evidence chain.

## Ground rules (from AGENTS.md — enforced in review)

1. **Physics state has exactly one owner**: the `SimulationEngine` thread.
   Everything else interacts via commands (`engine.submit`) or snapshots.
   Planning/checking threads use their own shadow `MjData` — never the
   engine's.
2. **Planners never import Qt.** The UI never contains core math
   (`np.linalg`, kinematics, collision logic). Benchmarks never patch the
   kernel to look good.
3. Every new module needs at least one unit test before it lands.
4. Random algorithms accept a `seed`; benchmarks fix seeds.
5. **Failure must be loud**: no exception swallowing, no fake successes, no
   results that were not physically measured. "Virtual snap" outcomes are
   never counted as physical grasps.
6. Third-party code requires a license check + an entry in
   `THIRD_PARTY_NOTICES.md`.

## Getting started

```bat
git clone <repository-url>
cd pandaforge
setup.cmd          :: Windows (Linux: pip install -r requirements-lock.txt && pip install -e . --no-deps)
test.cmd           :: or: .venv\Scripts\python -m pytest tests -q
```

All 75 tests must pass locally before you open a PR. CI runs the same suite
on `windows-latest` and `ubuntu-latest`.

## What makes a good PR

- A test that would fail without your change (bug fix) or a benchmark row
  that quantifies the improvement (optimization).
- For algorithm work: reference the paper you implemented from in
  `docs/ALGORITHMS.md`, and add the benchmark protocol you used.
- For UI work: an offscreen test in `tests/gui/`.

## What will be rejected

- Test criteria weakened to make something pass.
- "Improvements" without before/after measurements on identical seeds.
- Hardcoded expected values that encode test data into the algorithm.
- Changes that break the single-owner threading model.
- Untested TODO-comment code.

## Commit style

One logical change per commit; imperative subject line; mention the
evidence (test file or benchmark) in the body. The project history is part
of its story — `git log` should read like a development narrative.

## Reporting bugs

Open an issue with: what you did, what you expected, what happened, and the
relevant log lines. If the bug is in the simulation, include the scene JSON
(`Scene` tab → Save scene…). Security-sensitive reports: see SECURITY.md.
