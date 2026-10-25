# Test Report

Test suite: `pytest tests` on 2026-10-25, Windows 11, Python 3.11.9,
pytest 9.1.1 + pytest-timeout. QT runs with the offscreen platform except
the viewer tests, which open a REAL MuJoCo window (skipped on display-less CI). Result: **75 passed, 0 failed, 1 documented skip** (incl. real-viewer and showcase runs).

Legend: PASS / FAIL / SKIP / BLOCKED / NOT RUN — nothing below is
pre-filled; every PASS row corresponds to a real executed test.

## Unit tests (`tests/unit/`) — 49 tests, all PASS

| Test file | Covers | Result |
|---|---|---|
| test_kinematics.py (13) | quaternion/matrix round-trip; SO(3) error incl. near-pi branch; FK home sanity; **Jacobian frame convention via finite differences**; IK 100-pose acceptance (≥95% required, got 100%); out-of-reach rejection; joint-limit compliance; multi-start; tolerance fields | PASS |
| test_collision.py (7) | home config collision-free; endpoint collision detection; **checker vs mj_forward parity** (30 random states); mid-segment penetration caught (both endpoints free); gripper-cube whitelist semantics in both modes; joint-limit reporting; path validation sampling | PASS |
| test_planners.py (8) | RRT-Connect invariants on blocked task + independent re-validation; RRT on open task; RRT-Connect efficiency vs RRT; impossible goal fails cleanly (no fake path); linear baseline refuses blocked / accepts free; **seed determinism (identical path)**; shortcutting reduces length + stays valid | PASS |
| test_toppra.py (7) | TOPP-RA never slower than the conservative generator (20 random paths); numeric limit compliance; zero boundary velocities; endpoint/time clamping; duplicate-waypoint and 2-point degeneracy; default-factory dominance; tiny-limit feasibility | PASS |
| test_rrt_star.py (5) | RRT\* solves and its RAW path cost never exceeds RRT's (same seed); cost history monotone; Informed solves; informed sampler produces points provably inside the admissible ellipsoid (exact inverse-transform check, 300 samples); determinism; k(n) bound sanity | PASS |
| test_trajectory.py (9) | generated trajectories respect v/a limits; endpoints; time clamping; long-jump stretching; degenerate path rejection; tiny-limits feasibility; zero-rescale-budget raise; Panda limit spec; C1 velocity continuity at waypoints | PASS |

## Integration tests (`tests/integration/`) — 5 tests, all PASS

| Test | What it proves | Result |
|---|---|---|
| test_manual_joint_command_converges | direct commands drive the physical arm | PASS |
| test_invalid_joint_command_raises | invalid targets raise, never silently accepted | PASS |
| test_trajectory_execution_through_physics | plan → time-parameterize → execute via servos; arm ends at goal; tracking stats collected from the real run | PASS |
| test_infeasible_trajectory_is_refused | hand-built too-fast trajectory rejected before execution | PASS |
| test_estop_freezes_motion | e-stop mid-trajectory holds targets (drift < 0.05 rad) | PASS |

## Physics tests (`tests/physics/`) — 5 tests, all PASS

| Test | What it proves | Result |
|---|---|---|
| test_full_pick_and_place_success | the full FSM chain physically grasps and places the cube; independent post-check (xy ≤ 35 mm, resting z, moved > 150 mm, gripper open); ALL 11 states visited in order | PASS |
| test_grasp_rejects_unreachable_place | out-of-reach place target fails with a clear reason | PASS |
| test_abort_stops_task | user abort reaches ABORTED and the arm stops | PASS |
| test_roof_blocked_falls_back_to_side_grasp | roof blocks the top approach -> automatic SIDE grasp; full shelf-to-shelf transfer verified physically (candidate must be non-top) | PASS |
| test_grasp_candidate_generation_properties | 5 antipodal candidates, top first, pregrasp offset along -approach, legacy top frame preserved | PASS |

## GUI tests (`tests/gui/`) — 6 tests, all PASS (offscreen Qt)

| Test | What it proves | Result |
|---|---|---|
| test_window_boots_and_refreshes | window + engine + 100 ms refresh loop | PASS |
| test_joint_command_flow | slider signal → engine → physical convergence | PASS |
| test_ik_flow_via_panel | IK worker → result label → Move-to-solution | PASS |
| test_plan_and_execute_flow | plan worker → trajectory worker → executor → completed | PASS |
| test_scene_switch_flow | in-place scene switching | PASS |
| test_full_pick_place_via_ui | the ENTIRE pick&place driven through UI signals | PASS |

## Viewer tests (`tests/gui/test_viewer.py`) — 4 tests, all PASS

| Test | What it proves | Result |
|---|---|---|
| test_mujoco_viewer_submodule_importable | pins the lazy-submodule trap (`mujoco.viewer` needs explicit import) | PASS |
| test_viewer_module_imports_cleanly | wrapper module imports | PASS |
| test_real_viewer_launch_and_close | the ACTUAL passive viewer opens, syncs, closes (skipped on display-less machines) | PASS |
| test_viewer_handle_wrapper_launches | the RobotStudio wrapper's real launch path | PASS |

Post-release regression found by the USER's first run: the 3D viewer never
opened because `mujoco.viewer` was used without an explicit import; the
graceful headless fallback masked it as a warning. Fixed in
`simulation/viewer.py`, pinned by the tests above, verified end-to-end
(engine + live viewer + joint command + clean shutdown).

## Showcase tests (`tests/gui/test_showcase.py`) — 4 tests, all PASS

| Test | What it proves | Result |
|---|---|---|
| test_showcase_completes_end_to_end | one-click demo: obstacle sweep + full pick & place, physical cube at target | PASS |
| test_showcase_abort_on_estop | e-stop aborts the sequencer via callback (controllers detach on estop) | PASS |
| test_path_visual_and_decorations | path decimation bounds; empty/short inputs | PASS |
| test_decorations_drawn_in_real_viewer | user-scene geoms render in a REAL viewer window | PASS |

Two engine protocol defects found by these tests and fixed: ApplySceneCmd
and E-stop silently detached controllers (now: controllers may opt into
surviving scene switches; e-stop notified via callback).

## Replanning tests (`tests/integration/test_replan.py`) - 2 passed, 1 documented skip

| Test | What it proves | Result |
|---|---|---|
| test_guarded_execution_replans_and_completes | sweeping panel closes the corridor mid-task; guarded execution completes with >=1 replan, ZERO arm-obstacle contacts, goal reached | PASS |
| test_guard_detects_imminent_collision | time-pinned checks flag configurations the obstacle is about to occupy | PASS |
| test_static_plan_baseline_gets_hit_or_fails | (documented skip: the baseline contrast is measured in the replan benchmark rather than duplicating a collision in CI) | SKIP (documented) |

## Interactive verification (manual, NOT covered by offscreen tests)

The real MuJoCo viewer window and mouse/keyboard interaction were exercised
during development on this machine (viewer launch, camera orbit, live scene
switching, start/stop of tasks from the UI). These interactions are
NOT automated — recorded here explicitly as NOT RUN under automation rather
than claimed as covered. Screenshots in docs/img/ are real captures.

## Regression history (failures found and fixed during development)

1. Duplicate `home` keyframe in generated scenes (compile error) → removed.
2. `single_obs` wall intersected the arm at home pose → obstacle moved.
3. Python-loop nearest-neighbour made RRT 60x slower than needed → vectorized.
4. Checker flagged the resting cube (table-cube contact) → mode-dependent
   whitelist.
5. Trajectory completion froze servo lag; hold-states ratcheted downward
   under gravity (22 mm) → settle-based completion + static hold targets.
6. Gripper squeeze too weak (0.25 N) → measured force curve, 20 mm depth.
7. Finger hull grazed table at edge placements → 10 mm release clearance.
8. IK benchmark population contained self-colliding ground truths →
   filtered (94.2% → 100%).

None of these were "fixed" by weakening the tests — the acceptance criteria
stayed at or above the brief's requirements.
