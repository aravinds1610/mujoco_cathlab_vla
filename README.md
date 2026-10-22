# FR3 Duo cathlab simulation (MuJoCo)

A MuJoCo simulation of a Franka FR3 Duo, a two-arm robot, handing instruments over in a cardiac cath lab. It is built to generate synthetic demonstrations and evaluate policies for the task "Hand me the `<object>`": pick a named instrument from a cluttered table and place it on a handover tray.

Status: work in progress. Working and tested: the robot model, the randomized scene, the simulator backend, a scripted IK expert, a LeRobot v3 dataset generator (dataset v1 published), dataset QA, an evaluation harness, and a first SmolVLA fine-tune with its `PolicyAdapter`. See [Results](#results-so-far) and [Not done yet](#not-done-yet).

## What's here

| Path | What it does |
|---|---|
| `cathlab_sim/scene.py` | Builds the scene: two Menagerie FR3 v2 arms with Robotiq 2F-85 grippers on the Duo's tilted mount, a head camera, two wrist cameras, a table and a handover tray |
| `cathlab_sim/robot_io.py` | Converts between the 16-number action/state format and MuJoCo controls, at 30 Hz. Order: left joints 1–7, left gripper, right joints 1–7, right gripper. Gripper 1 = open, 0 = closed |
| `cathlab_sim/layout.py` | Seeded random placement of the target object plus 2–5 distractors, with no overlaps |
| `cathlab_sim/backend.py` | `MujocoBackend`: `reset(scene, seed, target)`, `step(action16)` returning observation and `StepInfo`, and `render()` |
| `cathlab_sim/stages.py` | Task stage checks (reach, grasp, lift, transport, place) and collision detection |
| `cathlab_sim/contracts.py` | Shared interfaces: `SimBackend`, `SceneSpec`, `StepInfo`, `PolicyAdapter`, … |
| `cathlab_sim/ik.py` | Inverse kinematics for each arm's grasp point |
| `cathlab_sim/expert.py` | Scripted expert: grasp from the true object pose, waypoints, IK at 30 Hz (left arm). `expert_policy.py` wraps it as a `PolicyAdapter` |
| `cathlab_sim/generate.py` | Writes expert demos as a LeRobot v3 dataset, plus `manifest.yaml` and per-episode metadata. Accepts a `GapSpec` |
| `cathlab_sim/evaluate.py` | Runs a policy (`--policy expert` or `smolvla`) on seeded episodes. Reports success rate with a 95% Wilson CI, failure stages, per-object results and latency |
| `cathlab_sim/smolvla_policy.py` | A fine-tuned SmolVLA checkpoint wrapped as a `PolicyAdapter` (predicts 50 actions, executes the first 25, re-plans) |
| `cathlab_sim/watch.py` | Watch the expert or SmolVLA do the task live in the MuJoCo viewer |
| `cathlab_sim/dataset_qa.py` | Dataset QA report: balance, position/yaw coverage, gripper chatter, action jumps, tracking |
| `cathlab_sim/tools/diag_*.py` | Diagnostics behind the findings below (expert grasp misses, SmolVLA offline check) |
| `assets/meshes/` | Downloaded CC BY 4.0 meshes for the mesh pipeline (not used yet), see `ATTRIBUTION.md` |
| `configs/` | Object definitions (syringe, catheter, sheath, guidewire, scissors, gauze) and the scene spec with randomization ranges |
| `docs/AR2_robot_model.md` | Robot model and the 16-number action/state mapping |
| `tests/` | CPU tests: schema, mapping, layouts, backend determinism, stage checks |

Observations use LeRobot dataset key names: `observation.state` (float32, 16 values), and `observation.images.{external,left_wrist,right_wrist}_camera` (224×224 RGB).

## Setup

Tested on Ubuntu 24.04 with Python 3.12 and an NVIDIA GPU (CUDA 13 driver). PyTorch is pulled in by lerobot.

```bash
git clone <this repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Robot models (arms and grippers) come from MuJoCo Menagerie:
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie external/mujoco_menagerie
```

If you already have a Menagerie checkout, point to it with `export CATHLAB_MENAGERIE=/path/to/mujoco_menagerie` instead of cloning.

## Run

```bash
pytest -q tests                                    # all tests (headless, uses EGL)
python -m cathlab_sim.check_scene --png scene.png  # scene sanity checks + 3-camera image
python -m cathlab_sim.view --seed 3 --target catheter   # interactive viewer
```

In the viewer, **Rendering → Camera** switches between the three cameras, and the **Control** panel moves each of the 16 actuators. Without a GPU, set `MUJOCO_GL=osmesa` for the tests.

Generate a dataset and evaluate the expert:

```bash
python -m cathlab_sim.generate --n 200 --out datasets/handover_v1 [--gap gap.json] [--objects syringe sheath]
python -m cathlab_sim.evaluate --policy expert --n 60 --seed0 10000 --out reports/expert_eval.json
```

Load the dataset back with `LeRobotDataset("cathlab/handover_mujoco_scripted", root="datasets/handover_v1")`. Without system ffmpeg, add `video_backend="pyav"`.

## Datasets

| Dataset | What it is |
|---|---|
| [aravindsuresh16/cathlab_handover_mujoco_v1](https://huggingface.co/datasets/aravindsuresh16/cathlab_handover_mujoco_v1) | 200 scripted-expert episodes (91,375 frames, 30 fps), 5 objects as primitive shapes (catheter left out), 2–5 random distractors, 3 cameras at 224×224. LeRobot v3 |

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("aravindsuresh16/cathlab_handover_mujoco_v1")   # downloads from the Hugging Face Hub
```

Its QA report is in `reports/qa_v1.json` and `reports/qa_v1.png` (made with `python -m cathlab_sim.dataset_qa`). Known gaps: gauze has 22 episodes against 42–48 for the others, and only the left arm moves.

Watch a policy live (seeds from 10000 are held out from training):

```bash
python -m cathlab_sim.watch --episodes 5                                   # scripted expert
python -m cathlab_sim.watch --policy smolvla --ckpt outputs/smolvla_v1_3k/checkpoints/last/pretrained_model
```

Fine-tune SmolVLA on the dataset (8 GB GPU: batch 8 uses about 4.5 GB, about 2.3 steps/s on an RTX 4060 laptop GPU), then evaluate:

```bash
lerobot-train --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=aravindsuresh16/cathlab_handover_mujoco_v1 \
  --rename_map='{"observation.images.external_camera":"observation.images.camera1","observation.images.left_wrist_camera":"observation.images.camera2","observation.images.right_wrist_camera":"observation.images.camera3"}' \
  --batch_size=8 --steps=3000 --save_freq=1500 --wandb.enable=false --output_dir=outputs/smolvla_v1_3k
python -m cathlab_sim.evaluate --policy smolvla --ckpt outputs/smolvla_v1_3k/checkpoints/last/pretrained_model --n 60 --seed0 10000
```

The `rename_map` maps our camera names to the three camera slots of `smolvla_base`. Our 16-D state and action are padded to SmolVLA's 32.

Minimal backend use:

```python
import numpy as np
from cathlab_sim.backend import MujocoBackend
from cathlab_sim.scene import HOME_Q
from cathlab_sim.spec_io import load_scene_spec

be = MujocoBackend()
obs = be.reset(load_scene_spec(), seed=0, target_object="syringe")
hold = np.concatenate([HOME_Q, [1.0], HOME_Q, [1.0]])   # stay at the ready pose, grippers open
obs, info = be.step(hold)
print(info.stage_flags, info.success, info.collision)
```

## Results so far

Held-out evaluation, seeds 10000–10059, 10 episodes per object (all 6 objects):

| Policy | Success | 95% CI | Failures | Report |
|---|---|---|---|---|
| Scripted IK expert (ground-truth pose) | 47/60 = 78% | 66–87% | all at lift | `reports/expert_v1_eval_seed10000_n60.json` |
| SmolVLA, 3,000 steps on v1 (26% of one pass) | 0/60 = 0% | 0–6% | 58 at reach | `reports/smolvla_v1_3k_eval_seed10000_n60.json` |

Findings:

- **Expert lift failures are missed grasps, not slips.** In failing episodes the gripper closes almost fully (0.75–0.78 of 0.8) on air with 0–6 N grip; in successes it stops at 0.18–0.27 with about 100 N. The object is not pushed beforehand. Still open: finger orientation (for example along the 7 × 7 cm gauze's diagonal, wider than the 85 mm opening) versus a sideways offset. Run `cathlab_sim/tools/diag_expert_grasp.py`.
- **SmolVLA learned the motions but not yet where to go.** On dataset frames its 25-step action error is 0.035 rad, against 0.087 rad for holding still, and it is worst at the start of an episode, where it must find the target in the images. Live images and state match the dataset (about 1 level of video compression), so the adapter is not the problem. Likely fixes: longer training (one pass is about 11k steps), re-planning more often, and more varied data. Run `cathlab_sim/tools/diag_vla_offline.py`.
- **Idle right arm.** The expert uses only the left arm, so 8 of the 16 action columns are constant. Their std is rounding noise, which made the normalized SmolVLA loss about 700. `generate.py` now sets std = 1 for constant columns.

## Not done yet

- **Mesh objects** (`add_object(name, mesh_path)` with convex decomposition). The meshes are downloaded; the pipeline is not written yet.
- **GraspGen** grasps and camera-based target selection (SAM 3). The expert uses the ground-truth object pose.
- **Catheter:** physically unstable in MuJoCo (NaN accelerations), so it is left out of the v1 dataset (5 of 6 objects).
- **MuJoCo vs Isaac bake-off report.**
- **Expert fixes:** the grasp-miss root cause, closed-loop retry, and using the nearer arm.
- **Dataset v2:** meshes, visual randomization, more gauze episodes (22 against 42–48 for the others), and recovery demos. Then a longer SmolVLA run.
- **Tests without a GPU:** the tests render with EGL. OSMesa (`libosmesa6`, `MUJOCO_GL=osmesa`) is untested.
- **Licences** of ContactPilot and franka-vla-generalization: neither has a licence file, so nothing is copied from them.

## Credits

- Meshes: CC BY 4.0 from Sketchfab via Objaverse, authors listed in `assets/meshes/ATTRIBUTION.md`.
- Base model: [SmolVLA](https://huggingface.co/lerobot/smolvla_base) via [LeRobot](https://github.com/huggingface/lerobot).
- Robot models: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), `franka_fr3_v2` and `robotiq_2f85`. Each model carries its own licence in its folder.
- Approach inspired by [ContactPilot](https://github.com/VivekSai07/ContactPilot) and [franka-vla-generalization](https://github.com/VivekSai07/franka-vla-generalization). No code is copied from either.
