# FR3 Duo cathlab simulation (MuJoCo + Newton)

Synthetic demonstrations for a Franka FR3 Duo (two arms, Robotiq 2F-85 grippers) handing instruments over in a cardiac cath lab: "Hand me the `<object>`" = pick the named object from the table and place it on the handover tray. The output format is the Siemens `fr3_duo` LeRobot v3 schema (16-D joint state/action, 3 cameras at 224×224, 30 fps).

## Datasets and models

| Dataset / model | Contents |
|---|---|
| [**aravindsuresh16/cathlab_fr3_newton_v1**](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1) | **Right-arm FR3 Duo, bottle / vial / syringe**, physics and rendering in Newton (MuJoCo-Warp). 384 episodes (159,568 frames) = one per cell of the train grid (3 objects × 8 layouts × 4 palettes × 4 table colours; held-out splits left out for evaluation). Per-episode config in `meta/episodes_extra.jsonl`, preview in `preview.gif` |
| [**aravindsuresh16/smolvla_cathlab_fr3_newton_v1**](https://huggingface.co/aravindsuresh16/smolvla_cathlab_fr3_newton_v1) (model) | SmolVLA fine-tuned on the dataset above (16k steps, batch 10, 2 h 19 min on an RTX 4060 laptop). Research checkpoint: reaches within 6–8 cm of the target, **0/6 placements** so far (see Status) |
| [aravindsuresh16/cathlab_handover_mujoco_v1](https://huggingface.co/datasets/aravindsuresh16/cathlab_handover_mujoco_v1) | Earlier MuJoCo track: left arm, 5 instruments, 200 episodes |

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("aravindsuresh16/cathlab_fr3_newton_v1", revision="v1")
```

For quantile-normalized policies (π0.5), use `revision="v1.0.1"`: same episodes, but the idle left arm's `q01`/`q99` are widened so its sim jitter isn't amplified.

## Status

| Track | State | Details |
|---|---|---|
| Newton FR3 data (N1–N5) | ✅ Port, expert, variation, dataset and report done. Expert 100 % on train and all held-out tiers; CPU MuJoCo vs Newton agree on 60/60 episodes | [docs/NEWTON_REPORT.md](docs/NEWTON_REPORT.md) |
| SmolVLA on the Newton data | ⚠️ Trained (v1 model above). Inputs verified identical live vs training; first actions match the expert (0.02–0.04 rad). Closed loop it approaches to 6–8 cm, then stalls: no recovery examples in the data. Chunk-switch jitter fixed by blending (max jump 0.40 → 0.05 rad). Next: near-target start poses in the data, then continue training | `cathlab_sim/newton/watch.py`, `diag_policy.py` |
| MuJoCo track (v1, left arm) | Backend, expert (78 %), generator, eval harness, first SmolVLA run (0/60, undertrained) | [docs/STATUS.md](docs/STATUS.md) |

Throughput on this laptop (RTX 4060 8 GB, i9-14900HX), with cameras: Newton ~1,580 episodes/h at 32 worlds; CPU MuJoCo ~1,170/h per process, ~2,990/h with 12 processes. See the report.

## Layout

| Path | What it does |
|---|---|
| `cathlab_sim/newton/` | Newton track: right-arm task scene, `NewtonTaskSim` (parallel worlds, batched cameras), CPU twin, shared stage checks, expert, generator, LeRobot converter, N1–N5 checks |
| `configs/variation/cathlab_fr3_newton.yaml` | Variation grid (team format): geometry, palettes, table colours, splits, mass/friction/light ranges |
| `cathlab_sim/` (top level) | MuJoCo track: robot model and 16-D mapping, scene, backend, IK, v1 expert, generator, evaluation, SmolVLA adapter, viewer |
| `docs/` | `NEWTON_REPORT.md`, `STATUS.md` (MuJoCo track), `AR2_robot_model.md` (robot model, 16-D mapping) |
| `reports/` | Check outputs and preview images |
| `tests/` | MuJoCo-track tests (schema, mapping, layouts, backend, stages) |

## Setup

Ubuntu 24.04, Python 3.12, NVIDIA GPU. Two virtual environments, because Newton needs MuJoCo 3.12 and LeRobot 0.6.1 is used with MuJoCo 3.3.7. Never install both into one env.

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie external/mujoco_menagerie
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt                       # MuJoCo track + LeRobot
python3.12 -m venv .venv-newton && .venv-newton/bin/pip install "newton[sim]==1.6.0" trimesh scipy pydantic pyyaml pillow pytest
python3.12 -m venv .venv-eval && .venv-eval/bin/pip install "newton[sim]==1.6.0" trimesh scipy pydantic pyyaml pillow "lerobot[dataset,av-dep,smolvla]==0.6.1"   # run a policy in Newton
```

ROS users: unset `PYTHONPATH` inside both venvs (ROS packages otherwise shadow the venv's).

## Run

Newton track (details and the rest of the commands in the report):

```bash
source .venv-newton/bin/activate
python -m cathlab_sim.newton.check_n1                 # Newton port checks
python -m cathlab_sim.newton.runner check             # expert success per object
python -m cathlab_sim.newton.runner generate --out datasets/newton_raw/v1 --count 384 --worlds 32
source .venv/bin/activate
python -m cathlab_sim.newton.convert --raw datasets/newton_raw/v1 --out datasets/cathlab_fr3_newton_v1
source .venv-eval/bin/activate
python -m cathlab_sim.newton.watch --ckpt aravindsuresh16/smolvla_cathlab_fr3_newton_v1   # live MuJoCo window, Newton physics
python -m cathlab_sim.newton.watch --expert                                                 # the scripted expert, for comparison
```

MuJoCo track:

```bash
source .venv/bin/activate
pytest -q tests
python -m cathlab_sim.generate --n 200 --out datasets/handover_v1
python -m cathlab_sim.evaluate --policy expert --n 60 --seed0 10000
python -m cathlab_sim.watch --episodes 5               # live viewer
```

## Credits

- Robot models: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `franka_fr3_v2` and `robotiq_2f85` (licences in their folders).
- Physics and rendering: [Newton](https://github.com/newton-physics/newton), [MuJoCo](https://github.com/google-deepmind/mujoco) and [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp) (Apache-2.0).
- `cathlab_sim/newton/variation.py`: copied unchanged from the team repo (Rakeshdharan/vla-cathlab-companion@9deeb21, `dataengine/variation.py`).
- Meshes (not used yet): CC BY 4.0 from Sketchfab via Objaverse, see `assets/meshes/ATTRIBUTION.md`.
- Base model: [SmolVLA](https://huggingface.co/lerobot/smolvla_base) via [LeRobot](https://github.com/huggingface/lerobot).
- Approach inspired by [ContactPilot](https://github.com/VivekSai07/ContactPilot) and [franka-vla-generalization](https://github.com/VivekSai07/franka-vla-generalization); no code copied.
