# FR3 Duo cathlab simulation (MuJoCo)

A MuJoCo simulation of a Franka FR3 Duo, a two-arm robot, handing instruments over in a cardiac cath lab. It is built to generate synthetic demonstrations and evaluate policies for the task "Hand me the `<object>`": pick a named instrument from a cluttered table and place it on a handover tray.

Status: work in progress. Working and tested: the robot model, the randomized scene, the simulator backend, a scripted IK expert, a LeRobot v3 dataset generator, and an evaluation harness. Next: SmolVLA fine-tuning, GraspGen grasps, and more randomization.

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
| `cathlab_sim/evaluate.py` | Runs a policy on seeded episodes. Reports success rate with a 95% Wilson CI, failure stages, per-object results and latency |
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

## Credits

- Robot models: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), `franka_fr3_v2` and `robotiq_2f85`. Each model carries its own licence in its folder.
- Approach inspired by [ContactPilot](https://github.com/VivekSai07/ContactPilot) and [franka-vla-generalization](https://github.com/VivekSai07/franka-vla-generalization). No code is copied from either.
