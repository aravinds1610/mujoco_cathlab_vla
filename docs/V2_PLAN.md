# v2: realistic surgical instruments, both arms (autonomous run plan)

*Started 2026-10-23 21:16. Runs unattended for ~6 h. The progress log at the bottom is updated after every phase.*

## Goal

The same "Hand me the {object}" task, with **realistic surgical instruments** from NVIDIA Isaac for Healthcare
(Lightwheel, CC BY-NC 4.0), picked by **whichever arm is nearer** and placed on a central tray. Arm-to-arm
handover comes later. Then generate a dataset, continue SmolVLA training (batch 12) and evaluate it.

| | v1 | v2 |
|---|---|---|
| Objects | bottle, vial, syringe as cylinders/capsules | **scissors, tweezers, trocar, puncture device** as textured meshes with convex collision pieces |
| Arms | right only | **left and right**: the arm on the object's side picks |
| Target | grey box tray at +y | **instrument tray in the middle of the table**, reachable by both arms |
| Layouts | 8 train / 2 held-out | **~50: 45 train / 5 held-out**, 2 objects on each side |
| Start pose | home | home **and** near-target (recovery examples), 50/50 |

## Phases (times are targets)

| # | Phase | Done when (pass/fail check) | Target |
|---|---|---|---|
| 1 | **Assets and scene.** Bake each instrument lying flat, real masses, convex collision pieces; a tray reachable by both arms; v2 scene builder | **Load check** in CPU MuJoCo *and* Newton: every object at rest drifts ≤ 2 mm over 300 frames on the table and in the tray, a 5 cm drop settles, no NaN, renders look right (images saved) | 22:20 |
| 2 | **Expert, both arms.** Side-aware simulators, stage checks and expert; a grasp point and orientation per object; reach probe for both arms | **Every scenario:** {left, right} × {4 objects} × {home, near-start} × 5 layouts = 80 episodes, run in Newton and in CPU MuJoCo; **≥ 80 % per cell** (aim 100 %); contact sheet of one success per cell | 23:40 |
| 3 | **Data.** v2 variation config (~50 layouts); generate 360 home + 360 near-start episodes; convert to LeRobot; check variety | 720 episodes, balanced over object, arm, layout, table colour and start; loads in LeRobot with the right shapes; variety grid image | 00:50 |
| 4 | **Train.** SmolVLA from the v1 16k checkpoint, batch 12, checkpoint every 2k steps | the run finishes without errors; loss falls | 02:40 |
| 5 | **Evaluate.** Held-out layouts and table colour, per object and arm; blending on; swap test | an eval report per checkpoint | 03:05 |
| 6 | **Wrap up.** Docs, commit and push. Publish the v2 dataset; publish the v2 model **only if it places ≥ 1 object** | pushed; HF links checked | 03:15 |

## Rules for the unattended run

- Each phase ends with its check; nothing moves on after a failed check without a logged reason.
- **Fallbacks:**
  - An instrument stays below 80 % after 2 grasp attempts → drop it (keep at least 3) and log why.
  - Newton can't show the textures → flat average colour per part, logged.
  - Training runs out of memory at batch 12 → batch 10.
  - The tray isn't reachable by both arms → move or resize the target area, logged.
- The 16k v1 model and dataset stay as they are; v2 goes into new folders and new HF repos.
- Lightwheel meshes stay out of git (`external/`, gitignored). The dataset card credits them as CC BY-NC 4.0, non-commercial.
- Near-start episodes: the gripper starts 6–18 cm above and ±8 cm around the target (same as v1 near-start).

## Progress log

- 21:16 Plan written.
- 21:36 **Phase 1 done (pass).** Instruments baked lying flat with real masses (scissors 60 g, tweezers 25 g, trocar 90 g,
  puncture device 50 g; estimated) and 4–16 convex collision pieces each. Tray = Lightwheel plate (427 × 303 mm, floor
  2.6 mm) in the middle with box colliders; both arms reach it (reach map). Load check (`reports/v2_assets_check.json`):
  table layouts, each instrument on the tray, 5 cm drops: max drift **CPU 0.18 mm, Newton 0.03 mm**, no NaN.
  Fixes needed: (1) CPU MuJoCo creep of ~0.14 mm/s: MjSpec.attach had dropped the Robotiq model's own contact settings
  (elliptic cone, impratio 10); restored for v2 only. (2) Newton drops textures: the 2 textured parts get their
  texture's mean colour in Newton. v1 regression after the task refactor: 12/12 in Newton.
