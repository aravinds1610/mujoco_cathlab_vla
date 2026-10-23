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
| Start pose | home | home, near-target and **drop-recovery** (270 / 270 / 100) |

## Phases (times are targets)

| # | Phase | Done when (pass/fail check) | Target |
|---|---|---|---|
| 1 | **Assets and scene.** Bake each instrument lying flat, real masses, convex collision pieces; a tray reachable by both arms; v2 scene builder | **Load check** in CPU MuJoCo *and* Newton: every object at rest drifts ≤ 2 mm over 300 frames on the table and in the tray, a 5 cm drop settles, no NaN, renders look right (images saved) | 22:20 |
| 2 | **Expert, both arms.** Side-aware simulators, stage checks and expert; a grasp point and orientation per object; reach probe for both arms | **Every scenario:** {left, right} × {4 objects} × {home, near-start} × 5 layouts = 80 episodes, run in Newton and in CPU MuJoCo; **≥ 80 % per cell** (aim 100 %); contact sheet of one success per cell | 23:40 |
| 3 | **Data.** v2 variation config (~50 layouts); generate 270 home + 270 near-start + 100 drop-recovery episodes; convert to LeRobot; check variety | 640 episodes, balanced over object, arm, layout, table colour and start; loads in LeRobot with the right shapes; variety grid image | 00:50 |
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
- Drop-recovery episodes (owner's request, capped at 100): see the 22:30 log entry.
- **One generation run at a time, `--worlds 16 --procs 8`, launched with `setsid nohup`** (see the 22:03 log entry).

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
- 22:00 Resumed. `watch.py` now does v2 evaluation (`--task v2 --tiers ... --report`); expert 3/3 held-out through it, v1 1/1. Variation check (owner asked): varies = object x/y + full yaw (45 layouts, distractors too), table colour (4+1), light intensity and colour temperature, mass, friction, start pose (half near-target, random); **fixed** = place pose and tray position, light direction, camera poses, home pose, instrument textures. Owner: train on the current data. Removed outdated docs (STATUS.md, AR2_robot_model.md).

- 22:03 **Out of memory (my mistake).** A drop-mode test run started next to the near-start generation; together
  they used more than 32 GB, the kernel killed both, and the owner's desktop froze and VS Code crashed. One
  generation at 32 worlds + 12 workers uses ~24 GB (the main process holds every camera frame of a batch, ~13 GB).
  Fix: one run at a time, `--worlds 16 --procs 8` (~15 GB, same speed, ~1,100 episodes/h), detached. Near-start
  resumed from episode 96 with `--start`; the pose jitter is now drawn for the whole grid before slicing, so a resumed
  run equals a full run (checked against the 64 episodes logged before the crash).
- 22:16 **Home and near-start done:** 270/270 and 270/270 succeeded, no duplicates; 40–50 episodes per
  (instrument, arm).
- 22:30 **Drop-recovery episodes: 100 done** (`datasets/newton_raw/v2_drop`, 100 saved of 112 attempts). On CPU the
  expert picks the instrument and the gripper opens at a random frame: 40 % during the lift (a slip), 60 % during
  the carry (it lands between its spot and the tray). The episode starts from where it came to rest, arm raised and
  gripper open, and records the re-pick and place in Newton (successes only). The drop itself is not recorded, so
  the policy never learns to open mid-carry. Rejected starts: not lifted, on or within 3 cm of the tray, crossed to
  the other arm's side, long axis tilted > 15° (propped up), or moves when the scene is rebuilt. Smoke test first
  (24 configs): 22/24 saved; 10/22 moved > 4 cm (up to 33 cm), turned up to 29°;
  `reports/v2_drop_smoke_start_frames.png`. Two fixes from the smoke test: the trocar rests rolled ~32° (puncture
  device ~15°) even when placed normally, so the tilt check now looks only at the long axis; and drops were sampled
  over time, most fell during the slow lift, so they are now split lift/carry.
- 22:31 Phase 3 conversion started: all 640 episodes → `datasets/cathlab_fr3_instruments_v1`.
- 22:36 Conversion stopped at 48 and restarted: it counted 647. Bug: `--max-saved` logged 7 successes after the cap
  as saved without writing their files (fixed in `runner.py`; the 7 lines in `v2_drop` now say `episode: null,
  over_cap: true`, original log kept as `episodes_extra.jsonl.bak`). The partial output was moved to the session
  scratchpad, not deleted. Restarted conversion counts 640.

- 22:45 **Change of plan (owner):** the team is presenting tomorrow. After conversion: publish the v2 dataset to HF
  (`aravindsuresh16/cathlab_fr3_instruments_v1`, CC BY-NC 4.0), link it in the README, push everything. **Phases 4–5
  (fine-tuning, evaluation) are deferred** to a later session; the commands below still apply. README now opens with
  the project framing (learning SmolVLA fine-tuning and how the dataset shapes the policy) and a pipeline flowchart.
- 23:00 Conversion crashed at 273/640: 4 near-start `.npz` files were truncated (all written 22:06:42, when the
  32-world run was stopped mid-write). Moved them to the scratchpad, regenerated grid positions 92–95 (identical
  names, start poses and frame counts), and `convert.py` now opens every raw file before encoding. Restarted.
- 23:15 **Phase 3 done, dataset published:** https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_instruments_v1
  (640 episodes, 230,558 frames; tags v1, v3.0; licence cc-by-nc-4.0; checked from the Hub). Variety grid
  `reports/v2_dataset_variety.png`. `watch.py --task v2 --expert --start home|near|drop` replays the recorded episodes
  live (6/6 succeeded headless).

## Handoff status (read this first in the new chat)

**State at 23:15, 2026-10-23.** Repo `main` is ahead of `origin/main` with local commits only (not pushed).

| Phase | State |
|---|---|
| 1 Assets and scene | ✅ done, passes (`reports/v2_assets_check.json`) |
| 2 Expert, both arms | ✅ done, 60/60 Newton + 60/60 CPU (`reports/v2_expert_check.json`) |
| 3 Data | ✅ 640 episodes → `datasets/cathlab_fr3_instruments_v1`, published on HF (`aravindsuresh16/cathlab_fr3_instruments_v1`, v1) |
| 4 Train | ⏸ deferred (owner, 22:45); 100-step test first, then batch 12 from the v1 16k checkpoint |
| 5 Evaluate | ⏳ harness ready: `watch.py --task v2 --tiers ... --report` (expert 3/3 through it) |
| 6 Wrap up | ✅ dataset published, README linked, pushed (model: after phases 4–5) |

**Check or stop the running conversion:**
```bash
tail -3 /tmp/claude-1000/-home-aravindsuresh16/7bc47717-7358-4c7a-99e7-2feb03ecd51a/scratchpad/convert_v2.log  # "converted n/640" / "wrote 640 episodes"
pgrep -fa "[c]onvert --raw"        # stop: kill <the python PID>  (never `pkill -f`, it matches its own shell)
```
If the log is lost (the /tmp path is session scratch), the raw folders' `meta/episodes_extra.jsonl` are the source of
truth. A stopped conversion leaves a partial `datasets/cathlab_fr3_instruments_v1`; delete it (ask the owner) and rerun.

### Next steps (in order)

1. **Finish phase 3:** the conversion (running) merges the three raw folders round-robin, so any first-N slice mixes
   home, near and drop starts. Command used (in `.venv`):
   ```bash
   python -m cathlab_sim.newton.convert --raw datasets/newton_raw/v2_home datasets/newton_raw/v2_near datasets/newton_raw/v2_drop \
     --out datasets/cathlab_fr3_instruments_v1 --repo-id aravindsuresh16/cathlab_fr3_instruments_v1 \
     --version v1 --scene-id cathlab_fr3_instruments --notes "..."
   ```
   Then load-check with `LeRobotDataset(repo_id, root=...)` (640 episodes, shapes) and build a first-frame variety
   grid (like `reports/dataset_variety_near_s3.png`).
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
3. **Evaluate (phase 5)** in `.venv-eval`, 12 episodes per tier (covers both arms), blending on by default:
   ```bash
   python -m cathlab_sim.newton.watch --task v2 --ckpt outputs/smolvla_instruments_v1/checkpoints/last/pretrained_model \
     --no-viewer --episodes 12 --tiers in_distribution held_out_layout held_out_table_color all_held_out \
     --report reports/v2_eval_<ckpt>.json
   ```
   Then a live window run for the owner (`watch.py --task v2 --ckpt ... --episodes 3`, DISPLAY=:1).
   `diag_policy.py` is still v1-only (not needed for the eval).
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
- **Drop-recovery:** `runner.py generate --drop [--max-saved N]` (`_sample_drop_start`), `V2Task.start_override`,
  expert phase marks (`TaskExpert.marks`). `--start` resumes give the same configs and jitter as a full run.
- **Eval:** `watch.py --task v2 --tiers ... --report` (per tier, object and arm).
- **Upload:** `convert.py --push --license cc-by-nc-4.0 --description "..."` (`--skip-convert` to push only).

### Gotchas

- Three venvs: `.venv-newton` (generate), `.venv` (convert, train, tests), `.venv-eval` (Newton + LeRobot: watch/eval).
  Python scripts that use `make_pool` must be files, not stdin (spawn re-imports `__main__`).
- `pkill -f` / `pgrep -f` patterns match the calling shell (exit 144); use the `[x]yz` bracket trick or PIDs.
- **RAM:** one generation at 32 worlds + 12 workers is ~24 GB of 32; two at once froze the desktop (22:03). One run at a
  time, `--worlds 16 --procs 8`, `setsid nohup` so a Claude Code restart does not kill it.
- Newton import quirks already handled: textures/materials dropped, collision shapes invisible (marked VISIBLE except
  `_hull`), short solref padded, `njmax` raised.
- The v1 SmolVLA (`aravindsuresh16/smolvla_cathlab_fr3_newton_v1`) approaches to 6–8 cm, then stalls (0/6); v2 adds
  near-target starts to address that.
