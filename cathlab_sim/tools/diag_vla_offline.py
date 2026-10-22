"""Diagnostic: (A) does a fine-tuned SmolVLA predict the expert actions on dataset frames (vs a hold-still
baseline)? (B) do live MujocoBackend images match the dataset images?

    PYTHONPATH=. python cathlab_sim/tools/diag_vla_offline.py
"""
import json
import numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from cathlab_sim.backend import MujocoBackend
from cathlab_sim.scene import CAMERAS
from cathlab_sim.smolvla_policy import SmolVLAPolicyAdapter
from cathlab_sim.spec_io import load_scene_spec

R = "datasets/handover_mujoco_scripted_v1"
ds = LeRobotDataset("cathlab/handover_mujoco_scripted", root=R)
extra = {e["episode_index"]: e for e in map(json.loads, open(f"{R}/meta/cathlab_episodes_extra.jsonl")) if e["saved"]}
p = SmolVLAPolicyAdapter("outputs/smolvla_v1_3k/checkpoints/last/pretrained_model")
L = list(range(7))  # left arm joints

print("(A) offline: mean |error| over the next 25 actions, left joints (rad) / left gripper")
errs, holds = [], []
for ep in (0, 50, 120, 199):
    start = int(ds.meta.episodes["dataset_from_index"][ep])
    for t in (0, 60, 150, 250):
        s = ds[start + t]
        gt = np.stack([ds.hf_dataset[start + t + k]["action"].numpy() for k in range(25)])
        frame = {k: s[k] for k in ["observation.state"] + [f"observation.images.{c}" for c in CAMERAS]}
        frame["task"] = s["task"]
        p.policy.reset()
        with torch.inference_mode():
            pred = p.post(p.policy.predict_action_chunk(p.pre(frame)))[0, :25].cpu().numpy()
        e = np.abs(pred[:, L] - gt[:, L]).mean(); h = np.abs(s["observation.state"].numpy()[L] - gt[:, L]).mean()
        errs.append(e); holds.append(h)
        print(f"  ep {ep:3d} t {t:3d}: model {e:.3f}  hold-still {h:.3f}  | gripper model {np.abs(pred[:, 7] - gt[:, 7]).mean():.2f}")
print(f"  MEAN model {np.mean(errs):.3f} rad vs hold-still {np.mean(holds):.3f} rad")

print("(B) live render vs dataset frame 0 (uint8 levels)")
be = MujocoBackend(); scene = load_scene_spec()
for ep in (0, 120):
    e = extra[ep]; obs = be.reset(scene, e["seed"], e["target"])
    s = ds[int(ds.meta.episodes["dataset_from_index"][ep])]
    print(f"  ep {ep} seed {e['seed']} {e['target']}: state max diff {np.abs(obs['observation.state'] - s['observation.state'].numpy()).max():.4f}")
    for c in CAMERAS:
        live = obs[f"observation.images.{c}"].astype(np.float32)
        dset = (s[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255)
        swap = np.abs(live[..., ::-1] - dset).mean()
        print(f"    {c:20s} mean|diff| {np.abs(live - dset).mean():5.1f}  (if RGB<->BGR swapped: {swap:5.1f})  live mean {live.mean():5.1f} dataset mean {dset.mean():5.1f}")
