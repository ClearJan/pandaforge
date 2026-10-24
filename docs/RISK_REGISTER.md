# Risk Register

Living document; reviewed at each phase gate.

| # | Risk | Impact | Likelihood | Mitigation | Status |
|---|---|---|---|---|---|
| R1 | pip installs stall on large wheels (PySide6) | build blocked | observed once | batched installs with `--timeout 60 --retries 8`; `PySide6-Essentials` halves the payload | CLOSED (D2) |
| R2 | MuJoCo mesh convex hulls differ from visual geometry (palm/finger grazing) | false collisions, failed grasps | observed twice | grasp constants MEASURED from the model; pad offset +5 mm; release clearance 10 mm | CLOSED |
| R3 | Servo lag frozen as hold target -> physical drift | wrong end pose, collisions | observed once | settle-based completion + static hold targets (D5); physics tests cover it | CLOSED |
| R4 | Single-tree RRT fails narrow 7-D passages | weak benchmark optics | expected (algorithmic) | RRT-Connect is the P0 planner; RRT results reported honestly with budget sensitivity | ACCEPTED |
| R5 | Viewer/model reload complexity on scene switch | crashes, dangling windows | designed out | in-place scene switching (D9); covered by GUI test | CLOSED |
| R6 | Thread races between UI/planner/engine | deadlocks, corruption | designed out | single-owner engine + command queue + shadow MjData; no shared mutable state | MONITORED |
| R7 | Packaging misses hidden imports (PySide6/mujoco plugins) | broken dist exe | likely | build.cmd self-test (`--selftest`) runs the packaged exe headless before acceptance | MONITORED |
| R8 | Contact-dynamics nondeterminism across runs | flaky grasp trials | observed (bounded) | fixed seeds, tolerance-based success criteria, per-trial raw CSV rows | MONITORED |
| R9 | Chaos in generated scene files vs vendored menagerie dir | license/attribution confusion | low | generated files prefixed `rs_scene_*`; THIRD_PARTY_NOTICES.md documents both | CLOSED |
