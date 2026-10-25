# Architecture

## System overview

```
+--------------------------------------------------------------+
|                     RobotStudio (one product)                 |
|                                                               |
|  PySide6 main window (GUI thread)      MuJoCo viewer window   |
|  +-----------+ +-----------+ +--------+   (own render thread, |
|  | Robot     | | IK / Plan | | Scene  |    launched passively)|
|  | Control   | | tabs      | | editor |        ^              |
|  +-----------+ +-----------+ +--------+        | sync()       |
|        |              |              |         |              |
|        | signals      | QThread      |         |              |
|        v              v workers      v         |              |
|  +------------------------------------------------------------------+
|  |                SimulationEngine (engine thread)                  |
|  |  - sole owner of MjModel / MjData                                |
|  |  - command queue (SetJoints, SetController, ApplyScene, ...)     |
|  |  - control loop: Controller.update() -> ctrl -> mj_step          |
|  |  - snapshot() for lock-guarded state reads                       |
|  +------------------------------------------------------------------+
|         ^                  ^                    ^
|         |                  |                    |
|  TrajectoryExecutor   GraspFSM          CollisionChecker
|  (Controller)         (Controller;      (own shadow MjData,
|                       nests executors)   usable from any thread)
+--------------------------------------------------------------+
```

## Threading model

There are exactly four kinds of threads:

1. **Engine thread** - owns all MuJoCo physics state. Runs at 60 Hz wall
   pacing: drains the command queue, ticks the active
   `Controller` once per physics step (2 ms), steps MuJoCo for the requested
   realtime factor, syncs the viewer, and publishes state.
2. **Qt GUI thread** - widgets only. A 100 ms `QTimer` polls
   `engine.snapshot()` and updates panels/plots. Never touches MjData.
3. **Worker threads** (QThread) - IK solving, planning, trajectory
   generation. Pure computation on their own `MjData`; post results back via
   Qt signals. Planners never import Qt; the UI never contains math.
4. **Viewer render thread** - created internally by
   `mujoco.viewer.launch_passive`; the engine only calls `sync()`.

Cross-thread rules:

- Everything that mutates physics goes through `engine.submit(Command)`.
- Everything that reads physics uses `engine.snapshot()` (copies under a
  short lock) or `engine.state_copy()`.
- The `GraspFSM` and `TrajectoryExecutor` implement the `Controller`
  protocol and run INSIDE the engine thread - the FSM can therefore command
  the gripper, plan short moves and verify contacts without extra
  synchronization. Planning inside the FSM blocks the engine thread for the
  (bounded, seconds) planning duration; this trade-off is documented in
  docs/DECISIONS.md.

## Module map

| Module | Responsibility |
|---|---|
| `config/` | paths (frozen-aware), robot constants, units |
| `simulation/engine.py` | physics owner, command queue, controllers, scene switching in place |
| `simulation/scenes.py` | `SceneSpec` data model -> MJCF generation (union body layout), scripted moving obstacle, presets, JSON I/O |
| `simulation/viewer.py` | passive viewer lifecycle, graceful headless degradation |
| `robots/panda.py` | name-resolved joint/actuator/body indices, limits, gripper mapping |
| `kinematics/pose.py` | SO(3) math (quaternion <\-> matrix, axis-angle error) |
| `kinematics/fk.py` | FK + body Jacobian (world frame, FD-verified) |
| `kinematics/ik.py` | DLS IK with adaptive damping, nullspace bias, multi-start |
| `collision/checker.py` | whitelist collision checks, dense segment checking, carried-cube transform |
| `planning/base.py` | planner API + result dataclass |
| `planning/rrt.py` | RRT + RRT-Connect (vectorized nearest-neighbour trees) |
| `planning/linear.py` | straight-line baseline |
| `planning/shortcut.py` | randomized shortcutting |
| `trajectory/timing.py` | trajectory factory (TOPP-RA default, conservative spline fallback) |
| `trajectory/toppra.py` | TOPP-RA time-optimal parameterization with dominance guarantee |
| `control/monitor.py` | guarded execution + online replanning (park state machine) |
| `control/executor.py` | servo-streaming execution with settle-based completion |
| `manipulation/grasp.py` | grasp/place geometry (measured from the model) |
| `manipulation/fsm.py` | 15-state pick&place state machine |
| `ui/` | PySide6 panels, workers, theme |
| `benchmarks/` | IK / planning / grasp benchmark runners |

## Scene switching without viewer reload

All preset obstacles, the table, the grasp cube and the place marker exist in
every generated scene XML. `ApplySceneCmd` toggles them **in place** (static
body poses, geom `contype`, free-joint `qpos`, mocap positions), so switching
scenes never reloads the model and the viewer window stays open. Generated
scene files are written next to `panda.xml` because MuJoCo resolves
`include`/`meshdir` relative to the main XML file.

## Data flow of the flagship demo (pick & place)

```
UI grasp panel -> GraspTaskConfig -> GraspFSM (controller)
   PREGRASP:  IK (collision-filtered) -> RRT-Connect -> executor -> settle
   APPROACH:  IK descent -> 2-point trajectory -> settle
   CLOSE:     gripper servo command, static hold
   VERIFY:    >=2 fingertip-cube contacts (MuJoCo contact list)
   LIFT:      vertical segment (break-away tolerant validation)
              + physical check: cube actually rose
   TRANSFER:  RRT-Connect with carried-cube transform
   PLACE:     IK descent (10 mm release hover)
   RELEASE:   gripper open, cube free-falls and settles
   VERIFY:    xy tolerance + resting z + gripper open + low velocity
   RETREAT:   vertical pull-out, fallback RRT path near table edges
   SUCCESS
```

Every success claim is a physical measurement; every motion phase streams
joint targets through the position servos. There is no teleporting and no
scripted animation anywhere in the pipeline.

## Key trade-offs

- **FSM in the engine thread** vs a separate orchestration thread: chosen for
  state-ownership simplicity; bounded planning pauses are visible in the
  viewer but never deadlock. See docs/DECISIONS.md D4.
- **Shadow `MjData` for planning** vs locking the live data: chosen to keep
  the engine loop lock-free during worker planning.
- **In-place scene switching** vs model reload: chosen to keep the viewer
  window alive; obstacle geometry is limited to primitives in V1.
- **pyqtgraph curves** fed from engine history ring buffers (last 4000
  samples) instead of a streaming bus: simple, sufficient at 10 Hz UI refresh.
