# Technical Decisions Log

Append-only. Each entry: context, decision, reason, consequences.

## D1 - Python 3.11 instead of 3.12

- Date: 2026-10-25
- Context: The machine has Python 3.11.9 and 3.13 but not 3.12. The task
  brief prefers 3.12 but allows a verified 3.11.
- Decision: Use 3.11.9 (the default `python`).
- Consequences: All pinned deps (mujoco 3.17, PySide6 6.12, numpy 2.4)
  provide 3.11 wheels; CI runs 3.11 on windows-latest.

## D2 - PySide6-Essentials instead of full PySide6

- Date: 2026-10-25
- Context: The first pip install stalled at 0 KB/s on the large PySide6
  combined wheels; only QtWidgets/QtGui/QtCore are used.
- Decision: Pin `PySide6-Essentials` (LGPL-3.0, dynamically linked,
  unmodified).
- Consequences: ~half the download; no QtWebEngine/Qt3D. Import name and
  behaviour identical for what we use.

## D3 - Generated scene XMLs are siblings of panda.xml

- Date: 2026-10-25
- Context: MuJoCo resolves `include` and `meshdir` relative to the MAIN XML
  file's directory. Generating scenes elsewhere breaks mesh resolution.
- Decision: `SceneSpec -> rs_scene_<name>.xml` written into
  `assets/robots/panda/` (the same mechanism menagerie's own scene.xml uses).
- Consequences: scenes are regenerable (`ensure_preset_scene_files()`),
  offline-loadable, and the menagerie directory is clearly marked as
  partly-generated in THIRD_PARTY_NOTICES.md.

## D4 - Grasp FSM runs inside the engine thread

- Date: 2026-10-25
- Context: The FSM must command the gripper, plan short moves and read
  contacts at physics rate. Alternatives: (a) FSM thread + commands,
  (b) FSM as Controller in the engine thread.
- Decision: (b). The FSM implements the `Controller` protocol and nests
  TrajectoryExecutors.
- Consequences: single-owner physics, no FSM<->engine races; planning
  (0.1-2 s) briefly pauses stepping - acceptable for V1 and documented in
  the UI (the viewer freezes, the GUI stays responsive). Bounded by
  per-state planning budgets.

## D5 - Settle-based trajectory completion

- Date: 2026-10-25
- Context: Found by grasp trials: marking execution done at `t >= duration`
  froze the servo lag as the next state's hold target; holding the ACTUAL
  configuration instead ratcheted the EE 22 mm downward under gravity.
- Decision: (1) executor is "finished" only when the physical arm is within
  0.03 rad of the final waypoint (2 s settle budget); (2) FSM hold states
  return the STATIC final waypoint, never the current actual position.
- Consequences: honest end-of-motion criterion; slight per-phase time
  overhead; a blocked arm is reported as a failure instead of silently
  continuing.

## D6 - Measured gripper squeeze (20 mm command depth)

- Date: 2026-10-25
- Context: The menagerie gripper actuator is a tendon position servo
  (stiffness 100 N/m). Commanding the object width exactly produces ~0.25 N
  grip force - the cube slips at lift.
- Decision: command `cube_half - 20 mm` (measured 1.96 N normal force, 3.7x
  friction margin for the 100 g cube; the contact solver stops the fingers
  at the faces).
- Consequences: reliable grasp; force-limited so the object cannot be
  crushed; numbers recorded in docs/ALGORITHMS.md §6.

## D7 - Carried-cube collision transform

- Date: 2026-10-25
- Context: During transfer planning the shadow cube would otherwise stay at
  its grasp-time pose while the arm moves - validation would be wrong on
  both sides (phantom collisions and missed ones).
- Decision: `CollisionChecker.cube_carry_offset` places the shadow cube at
  (hand pose ∘ captured relative transform) for every sampled config.
- Consequences: correct transfer-path validation; the offset is captured
  from the live grasp (tolerates the cube sitting a few mm off-nominal).

## D8 - Vectorized nearest-neighbour trees

- Date: 2026-10-25
- Context: Profiled RRT: 3.2 ms/check overall with collision checks at
  0.05 ms - the Python per-node nearest scan dominated.
- Decision: preallocated (N,7) point arrays + einsum nearest queries.
- Consequences: measured 3.2x planner speed-up (20000 iterations in ~8 s);
  identical algorithmic behaviour, deterministic per seed.

## D9 - In-place scene switching

- Date: 2026-10-25
- Context: `mujoco.viewer.launch_passive` binds model+data at launch;
  reloading the model would close/reopen the viewer window.
- Decision: every scene carries all bodies; switching toggles poses and
  geom `contype`/`conaffinity` on the live model.
- Consequences: instant scene switching with a stable window; V1 obstacles
  limited to primitives (acceptable for the product scope).

## D10 - Honest IK benchmark population

- Date: 2026-10-25
- Context: The first IK run reported 94.2%; 6 of 7 "failures" had <1.1 mm
  error - they were FK targets whose ONLY realizations are self-colliding
  configurations, rejected by the collision validator.
- Decision: the benchmark samples ground-truth configurations filtered for
  collision-freeness (128 sampled -> 120 valid); self-colliding targets are
  not "valid reachable poses".
- Consequences: 100% success on the valid population; the sampling rule is
  documented in the benchmark script.

## D11 - Union scene layout (position-only hiding)

- Date: 2026-10-25
- Context: engine construction on 'empty' crashed (missing cube body) and
  preset switching was silently half-applied. Root cause: MuJoCo bakes
  contype=0 geoms out of the collision pipeline at COMPILE time - runtime
  flag flips cannot re-enable them (verified experimentally).
- Decision: every scene XML contains every body (table, all preset
  obstacles, cube, place marker, moving obstacle); disabled bodies are
  parked far below the floor by POSITION only, collisions always enabled.
- Consequences: true in-place preset switching, one model per session, no
  viewer reloads; ResetCmd re-applies the current spec after the keyframe
  reset (the keyframe restores the LOADED XML's world, which may be another
  preset).

## D12 - Shadow-cube live sync + semantic contact modes

- Date: 2026-10-25
- Context: position-jittered grasp trials dropped to 50% - the shadow
  checker was judging contacts against a cube at its XML spawn pose, not
  the live one; and the enumerated rest-pair whitelist could not handle
  arbitrary supports (pedestals).
- Decision: the FSM syncs the shadow cube to the live pose every tick
  (unless carrying); contact semantics became mode rules instead of pair
  lists (free mode: cube-vs-environment is physics, allowed; holding mode:
  forbidden unless breaking away; grasp pairs only while holding).
- Consequences: correct validation for any support geometry; the
  roof side-grasp scene became feasible at all.

## D13 - TOPP-RA geometry/time-law split + dominance guarantee

- Date: 2026-10-25
- Context: TOPP on raw waypoint polylines compared different GEOMETRY than
  the executed spline (unfair, sometimes 5x slower); refit overshoots broke
  the numeric check (a_ratio 74).
- Decision: geometry = the legacy clamped spline, sampled by ARC LENGTH
  (uniform-time sampling decouples parameter from geometry); TOPP-RA
  re-times that exact geometry; an explicit dominance guarantee returns
  the faster of TOPP/legacy, so the result is never slower by construction.
- Consequences: honest -58% on long paths; 0% rows in the benchmark are
  the guarantee working where the conservative allocation is already
  near-optimal.

## D14 - Guarded replanning: one semantics, two states, verified park

- Date: 2026-10-25
- Context: five failed designs (single-time guards missing transiting
  panels; plan/guard semantic mismatch thrash; EE escapes swinging the
  elbow into the sweep plane - 44 simultaneous contacts; waiting inside
  the band; unguarded retreats getting grazed).
- Decision: swept-volume semantics live IN the checker (planner and guard
  share one definition); the guard checks the reference AND the physical
  arm over a 1.2 s horizon; violations retreat to a park configuration
  verified against the ENTIRE sweep cycle, via planned+guarded paths;
  replans pin the obstacle at NOW so the checker's swept window equals the
  guard's horizon.
- Consequences: 100% zero-contact completion at design speed; at the
  envelope edge the guard degrades to fewer, grazing contacts with 100%
  goal reach (vs 0% unguarded) - the operating envelope is measured and
  documented, not hidden.
