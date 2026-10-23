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
- 21:44 **Phase 2 done (pass).** Expert check (`reports/v2_expert_check.json`): {left, right} × {scissors, trocar,
  puncture device} × {home, near-start} × 5 layouts = **60/60 in Newton and 60/60 in CPU MuJoCo**, every cell 5/5;
  contact sheet `reports/v2_expert_contact_sheet.png`. **Deviation:** tweezers dropped as a target and kept as a
  distractor on the table. Their thin rigid prongs slip out of the grasp during the lift in Newton (2/20 grasped
  mid-length; 4/20 near the joint, `reports/v2_expert_check_tweezers.json`). Both engines use the same contact
  settings (elliptic cone, impratio 10, checked), and slip during the slow lift rules out carry speed, so I applied the
  fallback after one attempt instead of two. Train grid: 45 layouts × 3 targets × 2 table colours = 270.
- 21:44 Phase 3 started: 270 home (seed 0) + 270 near-start (seed 1) episodes.
- 21:46 **Paused for handoff to a new chat** (owner's request). Phases 1–2 done; phase 3 generation running unattended.

## Handoff status (read this first in the new chat)

**State at 21:46, 2026-10-23.** Repo `main` is ahead of `origin/main` with local commits only (not pushed).

| Phase | State |
|---|---|
| 1 Assets and scene | ✅ done, passes (`reports/v2_assets_check.json`) |
| 2 Expert, both arms | ✅ done, 60/60 Newton + 60/60 CPU (`reports/v2_expert_check.json`) |
| 3 Data | 🔄 **generation running in the background** (started 21:44): 270 home-start → `datasets/newton_raw/v2_home`, then 270 near-start → `datasets/newton_raw/v2_near`. ~25 min each. Converter ready but not run |
| 4 Train | ⏳ not started |
| 5 Evaluate | ⏳ not started; the v2 evaluation harness is not written yet (see below) |
| 6 Wrap up | ⏳ not started |

**Check or stop the running generation:**
```bash
tail -2 /tmp/claude-1000/-home-aravindsuresh16/7bc47717-7358-4c7a-99e7-2feb03ecd51a/scratchpad/gen_v2_home.log   # "[n/270] saved ..." / "done:"
tail -2 /tmp/claude-1000/-home-aravindsuresh16/7bc47717-7358-4c7a-99e7-2feb03ecd51a/scratchpad/gen_v2_near.log
ls datasets/newton_raw/v2_home/episodes | wc -l; ls datasets/newton_raw/v2_near/episodes | wc -l
pgrep -fa "[r]unner generate --task v2"        # stop: kill <the python PIDs>  (never `pkill -f`, it matches its own shell)
```
If the logs were lost (the /tmp path is session scratch), the episode counts and `meta/episodes_extra.jsonl` in each
raw folder are the source of truth. Re-running with the same arguments into a new folder reproduces the same episodes.

### Next steps (in order)

1. **Verify phase 3 data:** both raw folders have ~270 episodes; `meta/episodes_extra.jsonl` shows success per
   attempt (expect ~100 %); balance over side × object × start. Then convert (in `.venv`, LeRobot), about 3 s per episode:
   ```bash
   source .venv/bin/activate
   python -m cathlab_sim.newton.convert --raw datasets/newton_raw/v2_home datasets/newton_raw/v2_near \
     --out datasets/cathlab_fr3_instruments_v1 --repo-id aravindsuresh16/cathlab_fr3_instruments_v1 \
     --version v1 --scene-id cathlab_fr3_instruments \
     --notes "FR3 Duo, both arms (nearer arm picks), surgical instruments (NVIDIA Isaac for Healthcare / Lightwheel, CC BY-NC 4.0), central tray, 50 layouts (45 train / 5 held-out), home + near-target starts. "
   ```
   Then load-check with `LeRobotDataset(repo_id, root=...)` and build a first-frame variety grid (like
   `reports/dataset_variety_near_s3.png`).
2. **Train (phase 4)** in `.venv`, from the v1 16k checkpoint, **batch 12** (owner's choice; batch 16 measured at
   7.3 GB / 8 GB, batch 10 at 5.2 GB). LeRobot uses the new dataset's normalization stats when continuing from a
   checkpoint (checked in `lerobot_train.py`):
   ```bash
   lerobot-train --policy.path=outputs/smolvla_newton_v1_16k/checkpoints/016000/pretrained_model --policy.device=cuda \
     --policy.push_to_hub=false --dataset.repo_id=aravindsuresh16/cathlab_fr3_instruments_v1 \
     --dataset.root=datasets/cathlab_fr3_instruments_v1 \
     --rename_map='{"observation.images.external_camera":"observation.images.camera1","observation.images.left_wrist_camera":"observation.images.camera2","observation.images.right_wrist_camera":"observation.images.camera3"}' \
     --batch_size=12 --steps=<~2.5 h worth; measure steps/s first> --save_freq=2000 --log_freq=200 \
     --wandb.enable=false --output_dir=outputs/smolvla_instruments_v1
   ```
   Run a 100-step test first (memory + steps/s), as done for batch 10/16.
3. **Evaluate (phase 5):** `cathlab_sim/newton/watch.py` and `diag_policy.py` are still **v1-only**. They build
   `NewtonTaskSim(1)` with the default V1Task and v1 `eval_configs`. Add `--task v2` (use
   `runner.make_task`, `task.eval_configs(tier, seeds)`, `task.instruction(cfg)`) and an eval loop over the
   `V2_TIERS` (`instruments.py`), per object and per arm. Use `SmolVLAPolicyAdapter(..., blend=10)` (removes
   chunk-switch jumps). The owner wants a live MuJoCo window run (`watch.py`, DISPLAY=:1) to see it.
4. **Wrap up (phase 6):** push the dataset to HF (dataset card: credit Lightwheel/NVIDIA i4h, **CC BY-NC 4.0,
   non-commercial**); push the model to HF **only if it places ≥ 1 object** (owner's rule); README + this file;
   `git push`.

### What changed in the code (all committed locally)

- **Task interface:** `task_scene.V1Task` and `instruments.V2Task` provide objects, scene, episode params, grasp
  point and plan, acting arm, tray, instruction and configs. `cpu_sim`, `scene` (NewtonTaskSim), `expert_fr3`
  (TaskExpert(model, task, side)) and `runner` (`--task v1|v2`, `arm_q` start poses) use it. v1 regression after the
  refactor: 12/12 in Newton.
- **v2 assets:** `usd_assets.py` (USD → OBJ + texture), `collision_hulls.py` (CoACD), `instruments.py` (bake
  lying flat, grasp from geometry, V2Task, 50 layouts). Assets in `external/i4h_assets` (downloads) and
  `external/i4h_mujoco` (converted + hulls + `laid/`), gitignored.
- **v2 scene settings:** elliptic cone + impratio 10 (the Robotiq model's own; stops CPU creep). Newton gets
  texture-mean colours for textured parts (`V2Task.newton_shape_colors`).
- **Targets:** scissors, trocar, puncture device; **tweezers = distractor only** (slip; see the log above).
- **Checks:** `check_v2.py assets|expert [--objects ...]`.
- **Converter:** `convert.py --raw` accepts several folders (interleaved), plus `--scene-id` and `--notes`.

### Gotchas

- Three venvs: `.venv-newton` (generate), `.venv` (convert, train, tests), `.venv-eval` (Newton + LeRobot: watch/eval).
  Python scripts that use `make_pool` must be files, not stdin (spawn re-imports `__main__`).
- `pkill -f` / `pgrep -f` patterns match the calling shell (exit 144); use the `[x]yz` bracket trick or PIDs.
- Newton import quirks already handled: textures/materials dropped, collision shapes invisible (marked VISIBLE except
  `_hull`), short solref padded, `njmax` raised.
- The v1 SmolVLA (`aravindsuresh16/smolvla_cathlab_fr3_newton_v1`) approaches to 6–8 cm, then stalls (0/6); v2 adds
  near-target starts to address that.
