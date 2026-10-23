# v2: surgical instruments, both arms

*Status as of 2026-10-23 23:45. Read "Where things stand" and "How to resume" first.*

**In one sentence:** the "Hand me the {instrument}" task with realistic surgical instruments, picked by the arm on the instrument's side and placed on a central tray. The 640-episode dataset is built and published. Fine-tuning SmolVLA on it, and evaluating the result, are the next steps.

## Where things stand

| # | Phase | State | Evidence |
|---|---|---|---|
| 1 | Assets and scene | ✅ Done | Every instrument stays at rest in both engines; max drift CPU 0.18 mm, Newton 0.03 mm (`reports/v2_assets_check.json`) |
| 2 | Scripted expert, both arms | ✅ Done | 60/60 in Newton and 60/60 in CPU MuJoCo, every arm × instrument × start (`reports/v2_expert_check.json`) |
| 3 | Dataset | ✅ Done, published | 640 episodes on HF: [`aravindsuresh16/cathlab_fr3_instruments_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_instruments_v1), tag `v1` |
| 4 | Fine-tune SmolVLA | ⏸ Deferred | Owner's decision (23 Oct, team presentation). Command ready below |
| 5 | Evaluate | ⏸ Deferred | Tool ready and tested (`watch.py --task v2 ... --report`) |
| 6 | Publish | ✅ Dataset, README, GitHub | The v2 model is published only if it places at least 1 instrument (owner's rule) |

Git: `main` is pushed and clean. Nothing is running in the background.

## What changed from v1

| | v1 | v2 |
|---|---|---|
| Objects | bottle, vial, syringe (simple shapes) | **scissors, trocar, puncture device** (textured meshes); tweezers lie on the table as clutter |
| Arms | right only | **left and right**: the arm on the instrument's side picks |
| Target | box tray to one side | **instrument tray in the middle**, reachable by both arms |
| Layouts | 8 train / 2 held out | **45 train / 5 held out**, two instruments per side |
| Start pose | home only | **home, near-target, drop-recovery** |
| Episodes | 384 | 640 |

Why: the v1 model gets its first moves right, then stalls 6–8 cm from the object (0/6 placed). Its data never shows the arm near an object, so it has no example of correcting small errors. v2 adds those examples.

## The dataset

**Hugging Face:** `aravindsuresh16/cathlab_fr3_instruments_v1`, tag `v1`, licence CC BY-NC 4.0 (because of the instrument models). 640 episodes, 230,558 frames, 30 fps. Format: 16-D joint state and action, 3 cameras at 224 × 224.

| Start type | Episodes | What the episode shows |
|---|---|---|
| Home | 270 | Both arms start at the home pose |
| Near-target | 270 | The acting arm starts 6–18 cm above the instrument and up to 8 cm to either side, gripper at a random rotation |
| Drop-recovery | 100 | The instrument was dropped during the lift or the carry and lies where it came to rest; the arm starts raised with the gripper open, then re-picks and places it |

Every arm and instrument pair has 95–118 episodes. Episodes are interleaved (home, near, drop, …), so any first-N slice mixes all three start types.

**What varies per episode:** layout (position and full rotation of every instrument), table colour (4 for training), light brightness (0.7–1.3×) and colour temperature (4000–7500 K), instrument mass (±30 %), friction (0.5–1.0), start pose, and one of 3 instruction phrasings.

**What stays fixed:** the tray position and the place spot for each arm, light direction, camera poses, the home pose, and each instrument's texture.

**Held out for evaluation (not in the dataset):** layouts 45–49 and table colour 4.

**On disk** (not in git):

| Path | What | Size |
|---|---|---|
| `datasets/cathlab_fr3_instruments_v1` | The LeRobot dataset (same as on HF) | 1.8 GB |
| `datasets/newton_raw/v2_home`, `v2_near`, `v2_drop` | Raw episodes (`.npz`) + `meta/episodes_extra.jsonl` with every attempt's settings | 6.0 + 5.3 + 1.9 GB |
| `external/i4h_assets`, `external/i4h_mujoco` | Downloaded instrument models and their MuJoCo conversions | — |

The raw folders can be regenerated exactly, so they could be deleted to free 13 GB. Ask the owner first.

## How to resume

There are three virtual environments: `.venv` (LeRobot: convert and train), `.venv-newton` (generate data) and `.venv-eval` (Newton + LeRobot: run a trained policy).

### Phase 4: fine-tune SmolVLA

Continue from the v1 model at **batch 12** (owner's choice). Measured memory on the 8 GB GPU: batch 10 at 5.2 GB, batch 16 at 7.3 GB. When training continues from a checkpoint, LeRobot uses the new dataset's normalisation statistics (checked in `lerobot_train.py`).

1. Run 100 steps first, to check memory and measure steps per second.
2. Set `--steps` to fill about 2.5 hours, then run:

```bash
source .venv/bin/activate
lerobot-train --policy.path=outputs/smolvla_newton_v1_16k/checkpoints/016000/pretrained_model --policy.device=cuda \
  --policy.push_to_hub=false --dataset.repo_id=aravindsuresh16/cathlab_fr3_instruments_v1 \
  --dataset.root=datasets/cathlab_fr3_instruments_v1 \
  --rename_map='{"observation.images.external_camera":"observation.images.camera1","observation.images.left_wrist_camera":"observation.images.camera2","observation.images.right_wrist_camera":"observation.images.camera3"}' \
  --batch_size=12 --steps=<from the 100-step test> --save_freq=2000 --log_freq=200 \
  --wandb.enable=false --output_dir=outputs/smolvla_instruments_v1
```

If batch 12 runs out of memory, use batch 10.

### Phase 5: evaluate

12 episodes per test set covers both arms. Chunk blending is on by default (`--blend 10`); it removes the jumps between predicted action chunks.

```bash
source .venv-eval/bin/activate
python -m cathlab_sim.newton.watch --task v2 --ckpt outputs/smolvla_instruments_v1/checkpoints/last/pretrained_model \
  --no-viewer --episodes 12 --tiers in_distribution held_out_layout held_out_table_color all_held_out \
  --report reports/v2_eval_<checkpoint>.json
```

The report gives success per test set, instrument and arm, plus the stage each failure happened at. Afterwards, show the owner a live window run: the same command without `--no-viewer`, with `--episodes 3` and `DISPLAY=:1`.

To watch the recorded expert episodes live:

```bash
python -m cathlab_sim.newton.watch --task v2 --expert --start home    # or near / drop
```

### Phase 6: publish the model

Push the model to Hugging Face only if it places at least one instrument. Then update the README's "Dataset versions and what they taught" table and this file, and `git push`.

## Decisions and why

- **Tweezers are clutter, not a target.** Their thin, rigid prongs slipped out of the grasp during the lift in Newton: 2/20 grasped mid-length, 4/20 near the joint (`reports/v2_expert_check_tweezers.json`). Both engines used the same contact settings, and the slip happened during the slow lift, so carry speed wasn't the cause.
- **Gripper contact settings restored.** Attaching the Robotiq gripper to the arm had dropped the gripper model's own settings (elliptic friction cone, impratio 10). Without them, light instruments crept ~0.14 mm/s on CPU MuJoCo. Restored for v2 only.
- **Flat colours in Newton.** Newton's importer drops textures, so textured parts get their texture's average colour. The training images are therefore less detailed than the MuJoCo viewer.
- **Drop-recovery episodes start after the drop.** LeRobot can't exclude frames from training, so recording the gripper opening mid-carry would teach the policy to drop things. The drop is simulated on CPU, and the episode starts from where the instrument came to rest.
- **How drop starts are chosen:**
  - **Timing:** 40 % of drops happen during the lift (a slip just after grasping), 60 % during the carry (the instrument lands between its spot and the tray).
  - **Rejected starts:** the instrument wasn't lifted; it landed on or within 3 cm of the tray; it crossed to the other arm's side; its long axis tilts more than 15° (propped up on something); or it moves when the scene is rebuilt.
  - **Tilt is measured on the long axis only,** because the trocar naturally rests rolled ~32° and the puncture device ~15°.
- **Capped at 100 drop-recovery episodes** (owner's request).

## Where things are in the code

| File | What it does |
|---|---|
| `cathlab_sim/newton/instruments.py` | v2 task: instruments, tray, 50 layouts, held-out split, grasp plan, `start_override` for drop starts |
| `cathlab_sim/newton/task_scene.py` | v1 task and the shared robot / table / camera scene |
| `cathlab_sim/newton/runner.py` | Plans on CPU, runs in Newton, saves episodes. `generate --task v2 [--near / --drop] [--max-saved N] [--start i]` |
| `cathlab_sim/newton/expert_fr3.py` | Scripted expert for either arm; records where each phase starts (`marks`) |
| `cathlab_sim/newton/cpu_sim.py`, `scene.py` | CPU MuJoCo twin and Newton simulator, both aware of the acting arm |
| `cathlab_sim/newton/convert.py` | Raw episodes → LeRobot. Merges several raw folders, checks every file first, `--push --license --description` |
| `cathlab_sim/newton/watch.py` | Live viewer and evaluation: `--task v2`, `--tiers`, `--report`, `--expert --start` |
| `cathlab_sim/newton/check_v2.py` | Phase 1 and 2 checks: `assets` and `expert` |
| `cathlab_sim/newton/usd_assets.py`, `collision_hulls.py` | Instrument model conversion: USD → OBJ, convex collision pieces |

## Pitfalls

- **RAM.**
  - One generation run at 32 worlds and 12 workers uses ~24 GB of the 32 GB, and two runs at once froze the desktop.
  - Run one at a time with `--worlds 16 --procs 8` (~15 GB, same speed).
  - Start it with `setsid nohup`, so a Claude Code restart doesn't kill it.
- **Don't stop a generation run mid-batch.** It can leave half-written `.npz` files. The converter now catches them before encoding.
- **Resuming:** `--start i` reproduces exactly the same configs, position shifts and start poses as a full run.
- **Process search:** `pkill -f` / `pgrep -f` match the shell that runs them (exit 144). Use the `[x]yz` bracket trick or PIDs.
- **Scripts with `make_pool`** must be files, not piped to Python (the spawn start method re-imports the main module).
- **`diag_policy.py` is still v1-only.** It isn't needed for the v2 evaluation.

## History (23 Oct 2026)

| Time | Event |
|---|---|
| 21:16 | Plan written for an unattended ~6 h run |
| 21:36 | Phase 1 passed. Contact-settings fix; texture → average-colour fallback; v1 still 12/12 after the refactor |
| 21:44 | Phase 2 passed (60/60 both engines); tweezers became clutter. Generation started |
| 21:46 | Paused for a handoff to a new chat |
| 22:00 | Resumed. v2 evaluation tool added. Variation check done; owner chose to train on the current data. Outdated docs removed |
| 22:03 | **Out of memory:** a test run next to the generation froze the desktop. Switched to one run at a time, 16 worlds |
| 22:16 | Home and near-target starts done: 270/270 each |
| 22:30 | Smoke test, then 100 drop-recovery episodes (100 saved of 112 attempts) |
| 22:36 | Conversion restarted: `--max-saved` had logged 7 unsaved episodes as saved. Fixed |
| 22:45 | **Plan change:** team presenting tomorrow. Publish the dataset; defer fine-tuning and evaluation |
| 23:00 | Conversion restarted: 4 half-written files from the 22:06 stop. Regenerated identically; converter now checks files first |
| 23:15 | Dataset published and checked from the Hub. README reframed with a pipeline diagram; everything pushed |
