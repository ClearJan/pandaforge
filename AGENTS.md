# AGENTS.md — RobotStudio Multi-Agent Development Charter

This repository was developed by an autonomous agent team. This file records
the REAL constraints, role boundaries, shared interfaces and review rules that
governed the work, so future contributors (human or agent) can extend the
project coherently.

## Roles actually used

| Role | Responsibility | Owned files (write scope) |
|---|---|---|
| Main / Architecture & Integration | interfaces, dependency graph, integration, git | `src/robotstudio/**` (coordinator), root metadata |
| Kinematics engineer | FK/IK/Jacobian/SO(3) | `src/robotstudio/kinematics/**` |
| Motion-planning engineer | collision API, RRT/RRT-Connect, shortcut | `src/robotstudio/collision/**`, `src/robotstudio/planning/**` |
| Simulation & manipulation engineer | engine, scenes, gripper, grasp FSM | `src/robotstudio/simulation/**`, `src/robotstudio/manipulation/**`, `src/robotstudio/control/**` |
| UI engineer | PySide6 panels, workers, plots | `src/robotstudio/ui/**` |
| QA & benchmark engineer | tests, benchmark scripts, reports | `tests/**`, `src/robotstudio/benchmarks/**`, `reports/**` |
| Independent reviewer | audit, no implementation duties | `reports/FINAL_AUDIT.md` only |
| Documentation & release | docs, scripts, packaging | `docs/**`, `*.cmd`, `.github/**` |

## Hard rules

1. **Physics state has exactly one owner**: `SimulationEngine` (its thread).
   All other threads interact via command objects (`engine.submit`) or
   `engine.snapshot()`. Planning/checking threads use their own `MjData`
   (shadow) — never the engine's.
2. **Planners never import Qt.** UI never contains core math. Benchmarks
   never patch the kernel to look good.
3. Every new module needs at least one unit test before integration.
4. Random algorithms accept a `seed`; benchmarks fix seeds.
5. Failure must be loud: no exception swallowing, no fake successes, no
   "virtual snap" results counted as physical grasps.
6. Third-party code requires license check + attribution note.

## Shared interfaces (stable contract)

- `Pose`, `so3_error` — `robotstudio.kinematics.pose`
- `IKSolver.solve(target: Pose, q_init, n_restarts, validate) -> IKResult`
- `CollisionChecker.check_state / check_segment / check_path -> CollisionReport`
- `JointSpacePlanner.plan(start, goal, finger_opening, holding_cube) -> PlanningResult`
- `TrajectoryGenerator.generate(path) -> Trajectory`; `Trajectory.sample(t)`
- `SimulationEngine`: `submit(Command)`, `snapshot() -> SimSnapshot`,
  `set_controller(Controller)`, `on_estop(cb)`
- `GraspFSM(engine, spec)` — a Controller ticked by the engine thread; builds its own checker/IK/planner (shadow data)

## Integration gate

Before merging any module: `pytest tests/unit -x` must pass, and the module
must not regress `tests/integration`. The integration agent re-runs the full
suite and commits.
