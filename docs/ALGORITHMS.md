# Algorithms

All algorithms in this file are implemented in `src/robotstudio/` from the
original references - no robotics framework code is reused. Units: meters,
radians, seconds. Robot: 7-DoF Franka Panda (menagerie MJCF).

## 1. Forward kinematics

FK uses MuJoCo's own kinematics: `mj_forward` on a dedicated `MjData`, then
read `body("hand").xpos / xquat`. The `hand` body is the end-effector frame
(its +z axis points toward the fingertips - measured, see §6).

Complexity: one MuJoCo position-stage pass, O(model). Correctness is pinned
by `tests/unit/test_kinematics.py` (agreement with `data.body()` after
`mj_forward`, and finite-difference consistency of the Jacobian below).

## 2. Geometric Jacobian

`mj_jacBody(model, data, jacp, jacr, hand_id)` restricted to the 7 arm DoFs.
Both blocks are expressed in **world axes**; this was settled empirically by
central finite differences of the body pose (`test_jacobian_finite_difference`,
tolerance 1e-4) rather than assumed from documentation.

- jacp: d(EE position)/d(q) - world frame
- jacr: d(EE angular velocity)/d(q) - world frame

## 3. Orientation error (SO(3))

For current rotation R and target R* the error is the matrix logarithm of

    E = R* R^T        (world-frame error)

returning an axis-angle vector with angle in [0, pi]. Near pi the axis is
extracted from the quaternion (off-diagonal degeneracy). Euler-angle
differences are never used (they are discontinuous and wrong across
wraps). Implementation: `kinematics/pose.py::so3_error`.

## 4. Damped Least Squares IK (`kinematics/ik.py`)

Given the stacked error e = [p* - p; log(R* R^T)] and stacked Jacobian J:

    dq = J^T (J J^T + lambda^2 I)^-1 e        (6x6 solve, nv=7)

Features:

- **Adaptive damping**: lambda increases x4 when the combined error grows,
  decays x0.7 on progress, bounded to [1e-3, 10].
- **Step clamping**: Cartesian progress per iteration is capped at 0.1 m /
  0.5 rad to avoid overshoot near singularities.
- **Joint limits**: clamped after every step; nullspace bias
  `dq_null = (I - J+ J) k (q_center - q)`, k = 0.05, pulls redundant DoFs
  toward mid-range.
- **Convergence**: ||p* - p|| <= 5 mm AND angle(log E) <= 0.05 rad (~2.9 deg).
- **Numerical guards**: non-finite Jacobian/error aborts with a diagnostic;
  singular `solve` raises damping and retries.
- **Multi-start**: up to N restarts (home + Halton-ish random seeds); a
  candidate can additionally be rejected by an external validator (used for
  collision-free IK).

Assumptions/limits: local method - narrow basin poses may need restarts;
measured success 100% on 120 FK-generated reachable targets, max
5.0 mm / 2.8 deg, mean 9.9 ms / p95 57 ms (reports/BENCHMARKS.md).

Complexity: O(iters * 216) flops for the 6x6 solves + O(model) per
`mj_forward`.

## 5. Collision checking (`collision/checker.py`)

- Private ("shadow") `MjData` per checker -> thread-safe alongside the live
  engine. Per state: `mj_kinematics` + `mj_comPos` + `mj_collision`
  (measured 0.05 ms per check).
- **Whitelist semantics**: every contact pair (by body names) not explicitly
  allowed is a violation. Allowed pairs are mode-dependent:
  - grasp pairs (fingers/palm vs cube): only while `holding_cube=True`
  - rest pairs (cube vs table/floor): only while NOT holding it, or when a
    break-away is explicitly permitted (`allow_cube_rest=True`, used while
    lifting off / retreating)
- **Segment checking**: joint-space interpolation with max 0.02 rad between
  samples; the whole path is re-checkable independently
  (`check_path`), which the planner tests and the benchmark do with a FRESH
  checker instance.
- **Carried-object transform**: while grasping, every sampled configuration
  places the shadow cube at (hand pose ∘ grasp offset), so transfer paths are
  validated against the cube actually moving with the gripper.

Known trap this design prevents: endpoint-only checks miss mid-segment
penetration - covered by `test_mid_segment_penetration_caught`.

## 6. Grasp geometry (`manipulation/grasp.py`)

Measured from the menagerie collision model (not assumed):

- fingertip pad centres: 0.1029 m along hand +z from the hand origin
- finger mesh convex hulls extend to 0.1126 m (9.7 mm beyond the pads) and
  the hand hull reaches the cube top at a centred grasp - hence pads grip
  5 mm ABOVE cube mid-height (GRASP_PAD_OFFSET) so the descent is verified
  clear.
- squeeze depth 20 mm: commanded finger opening = cube_half - 20 mm. The
  tendon actuator is a 100 N/m position servo; measured grip force vs command
  depth: 11.6 mm -> 1.34 N (slips), 5.3 mm -> 1.96 N (holds, 3.7x friction
  margin for the 100 g cube). These numbers are in the squeeze-test log in
  reports/BENCHMARKS.md.

Top-down grasp: hand +z aligned with world -z (quaternion (0,1,0,0));
fingers close along world y gripping two opposite cube faces.

## 6b. Antipodal grasp candidate generation (`manipulation/grasp.py`)

For an axis-aligned cube the valid pad oppositions are the 6 face normals;
the bottom face is unreachable through the support. Candidates are generated
in preference order:

1. top-down (a = world -z; most stable, keeps the wrist clear of the table)
2. four horizontal side grasps (a = ±x/±y; hand +z = a, fingers closing
   along the world horizontal axis)

Hand orientation: hand +z is pinned to the approach direction a; for
horizontal approaches the hand x-axis is set to world up so the fingers
close horizontally (R = [x_h, y_h, z_h], quaternion via Shepperd). Grasp
pose: pads on the cube centre-line, hand origin one GRASP_LIFT back along
-a; pregrasp = grasp + PREGRASP_CLEARANCE back along -a.

Selection is declarative: the FSM takes the FIRST candidate whose pregrasp
IK is collision-free AND reachable by a collision-free plan. In the
``roof_grasp`` scene a low roof makes every top-down configuration collide,
so the planner automatically falls back to a horizontal side grasp - and
completes the shelf-to-shelf transfer physically (measured in the grasp
benchmark category ``roof_side``).

Break-away semantics (found the hard way, kept honest): at lift-off the cube
still rests on its support and the gripper grazes that support (measured
~0.1 mm hand-pedestal contact on side grasps). The support pairs are
discovered from the LIVE engine contact list (the shadow checker's list is
unstable exactly at the contact boundary, dist ~ 0) and temporarily
whitelisted for the lift validation only, via try/finally.

## 7. Sampling-based planning (`planning/rrt.py`)

Both planners share the unified API (start, goal, checker, seed, time
budget, max iterations) and the same state-validity predicate.

**RRT** (LaValle & Kuffner 2001): single tree from start; goal-biased
uniform sampling (10%); bounded extension step 0.3 rad; edge validity =
dense segment check; goal connection when within one step and the connecting
segment is valid.

**RRT-Connect** (Kuffner & LaValle 2000): two trees rooted at start and
goal; EXTEND toward a random sample, then CONNECT greedily toward the new
node until "reached" or "trapped"; trees swap roles each iteration. Path =
backtrack(tree1) + reverse(backtrack(tree2)).

Implementation notes:

- trees are flat preallocated arrays with **vectorized nearest-neighbour**
  (einsum); a pure-Python per-node scan was profiled as the dominant cost
  (O(n) Python iterations per sample) and vectorizing gave a measured 3.2x
  planner speed-up (reports/BENCHMARKS.md, §optimization).
- deterministic per seed (`numpy.random.default_rng(seed)`).
- **shortcutting** (`planning/shortcut.py`): up to 200 random i<j waypoint
  bypasses, accepted only if the bypass segment passes the dense check; the
  final path is re-validated independently.

Measured behaviour (8 s budget, fixed seeds 100-129, 3 scenarios):

| scenario | linear | RRT | RRT-Connect |
|---|---|---|---|
| empty | 100% (1.9 ms) | 100% (23 ms) | 100% (7 ms) |
| single obstacle | 28% | 45% | 100% (0.19 s) |
| narrow passage | 0% | 47% | 100% (0.23 s) |

Zero successful paths failed the independent re-validation (261 trials).
Single-tree RRT's weakness in 7-D narrow passages is a known property of the
algorithm and is reported honestly rather than tuned away.

## 7b. Optimal planners: RRT* and Informed RRT* (`planning/rrt_star.py`)

**RRT\*** (Karaman & Frazzoli, IJRR 2011 - k-nearest variant): same growth
loop as RRT plus, per iteration: (1) parent selection among the k(n) nearest
neighbours minimizing root-path cost through a collision-free segment, and
(2) rewiring those neighbours through the new node when cheaper, with the
cost delta propagated to the affected subtree (child-adjacency walk). The
connection bound is k(n) = ceil(e·(1+1/d)·ln n), d=7, which caps
per-iteration collision checks - critical at 0.05 ms/check.

**Informed RRT\*** (Gammell et al., IROS 2014): once a first solution of
cost cbest exists, all samples are drawn directly from the prolate
hyperspheroid (ellipsoid) that exactly contains all states able to improve
the current best path: centre = (start+goal)/2, major axis cbest/2 along the
start→goal direction (SVD-derived rotation), minor axes
sqrt(cbest²-cmin²)/2. Rejection handles joint-limit clipping. Convergence
toward the optimum therefore accelerates as cbest shrinks.

Both record best-cost-per-iteration (`stats["cost_history"]`) for honest
convergence benchmarking, and report raw path length separately from the
shortcut post-processing (`stats["raw_path_length"]`).

Measured behaviour (reports/BENCHMARKS.md §6): on solvable tasks RRT\* raw
paths are consistently shorter than RRT's and Informed RRT\* converges
faster; on the blocked 7-D scene single-tree optimal search rarely finds a
first solution within a 20 s budget - the classic feasibility/optimality
trade-off against RRT-Connect, reported as measured, not tuned away.

## 7c. Online replanning under a scripted moving obstacle (`control/monitor.py`)

The system layer that turns the static pipeline into a closed loop:

- **Scripted obstacle**: a mocap panel sweeps the workspace on a
  deterministic sinusoid (pure function of sim time), so planning stays
  reproducible.
- **Unified swept-volume semantics**: with a moving obstacle set, every
  collision check validates the obstacle's swept volume over
  [t, t + window] (3 samples) - planners and the runtime guard share ONE
  semantics, so a validated path cannot be instantly rejected by its own
  guard (semantic mismatch was the source of early replan thrash).
- **Guard**: each step validates the reference AND the physical arm (the
  arm lags the reference; the panel swept into that gap in forensics runs)
  against the obstacle over a 1.2 s horizon - long enough to detect,
  decelerate and retreat before contact (0.6 s fired with the hand already
  15 mm inside the panel).
- **Park configuration**: an EE pose above/behind the sweep plane, IK-solved
  and verified against the ENTIRE sweep cycle offline. On violation the arm
  retreats to park via a GUARDED planned path (a straight servo can cut
  through the panel; EE-Jacobian escapes were tried and are wrong - the
  panel hits arbitrary links and pulling the EE back swings the elbow into
  the sweep plane, measured 44 simultaneous contacts).
- **Replan loop**: parked, the controller replans to the goal with the
  obstacle pinned at NOW (the checker's swept window then covers exactly the
  guard's horizon); failures wait out the sweep and retry.

Measured (reports/BENCHMARKS.md §7): 100% zero-contact completion at both
benchmark speeds with double-digit replan counts; the unguarded baseline
executes the same initial plan into the panel.

## 8. Trajectory time parameterization (`trajectory/timing.py`)

Method: **time-scaled cubic spline**.

1. Per-segment synchronized trapezoidal lower bound on duration:
   `t = max_j (d_j/v_j + v_j/a_j)` (or triangular `2 sqrt(d/a)` when short).
2. Clamped cubic spline through the time-stamped waypoints (zero endpoint
   velocities).
3. **Numeric verification** of joint velocity/acceleration limits on a dense
   grid (max 2000 samples); on violation ALL durations are scaled by
   `max(v_ratio, a_ratio) * 1.05` and the spline is refit (<= 30 iterations).

The numeric check is authoritative - visual smoothness is never used as
evidence. Result is C2 within segments, C1 across waypoints (bounded
velocity jump, asserted <= 15% of vmax in tests).

Limits used: vmax = (1.0, 1.0, 1.5, 1.25, 1.5, 1.5, 1.5) rad/s, amax half of
the Panda spec values - conservative, documented in `config/settings.py`.

## 9. Execution (`control/executor.py`)

The executor samples the spline at simulation time and commands joint
targets to the position servos every physics step. Completion requires BOTH
the timeline to finish AND the physical arm to settle within 0.03 rad of the
final waypoint (2 s budget) - otherwise downstream states would freeze servo
lag as their hold target (this was a real bug found by the grasp trials:
gravity droop ratcheted the EE 22 mm down when states held the *actual*
configuration instead of the static reference).

Infeasible trajectories (violating the numeric limits) are refused before
execution (`InfeasibleTrajectoryError`).

## 10. Grasp state machine (`manipulation/fsm.py`)

Chain (each state has entry action, success exit, timeout, failure class):

```
IDLE -> PREGRASP -> APPROACH -> CLOSE -> VERIFY_GRASP -> LIFT
     -> TRANSFER -> PLACE -> RELEASE -> VERIFY_PLACE -> RETREAT -> SUCCESS
```

with `RETRY_DECISION` (internal routing) and terminal `FAILED`, `TIMEOUT`,
`ABORTED`. One retry loop: a failed VERIFY_GRASP reopens the gripper, pulls
out (validated in holding mode so break-away contact is tolerated) and
re-approaches once.

Physical success criteria (no shortcuts):

| state | criterion |
|---|---|
| VERIFY_GRASP | >= 2 fingertip-cube contacts in the live contact list |
| LIFT | cube centre actually rose >= 50% of the lift height |
| VERIFY_PLACE | xy within 35 mm, resting z band, gripper > 96% open, cube speed < 0.05 m/s |

Timeouts per state (4-40 s) plus per-phase planning budgets; the e-stop hook
aborts the FSM from any state.
