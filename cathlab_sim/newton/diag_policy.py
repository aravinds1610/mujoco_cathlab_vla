"""Diagnostic: does a trained policy see the same inputs live as in training, and does it predict the expert?

For dataset episodes, rebuilds the exact scene in Newton from meta/episodes_extra.jsonl and compares, at t = 0:
images (live vs dataset frame), joint state, and the policy's first actions on (a) the dataset frame and
(b) the live observation, against the expert's recorded actions. Run in .venv-eval:

    python -m cathlab_sim.newton.diag_policy --ckpt outputs/smolvla_newton_v1_16k/checkpoints/last/pretrained_model
"""
import argparse
import json

import numpy as np
import torch

from cathlab_sim.scene import CAMERAS

DATASET = "datasets/cathlab_fr3_newton_v1"
N = 25  # actions executed per policy call
RIGHT = slice(8, 15)  # right arm joints (the moving arm)


def predict(policy, images_chw: dict, state: np.ndarray, task: str) -> np.ndarray:
    """Same call as SmolVLAPolicyAdapter.predict, but taking CHW float images in [0, 1]."""
    frame = {"observation.state": torch.as_tensor(state, dtype=torch.float32), "task": task}
    for c in CAMERAS:
        frame[f"observation.images.{c}"] = torch.as_tensor(images_chw[c], dtype=torch.float32)
    policy.policy.reset()
    with torch.inference_mode():
        chunk = policy.post(policy.policy.predict_action_chunk(policy.pre(frame)))
    return chunk[0, :N].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from cathlab_sim.newton.scene import NewtonTaskSim
    from cathlab_sim.newton.variation import EpisodeConfig
    from cathlab_sim.smolvla_policy import SmolVLAPolicyAdapter

    ds = LeRobotDataset("aravindsuresh16/cathlab_fr3_newton_v1", root=DATASET)
    log = {e["episode_index"]: e for e in map(json.loads, open(f"{DATASET}/meta/episodes_extra.jsonl"))
           if e["episode_index"] is not None}
    policy = SmolVLAPolicyAdapter(args.ckpt)
    sim = NewtonTaskSim(1)
    fields = set(EpisodeConfig.model_fields)
    for ep in args.episodes:
        e = log[ep]
        cfg = EpisodeConfig(**{k: e[k] for k in fields})
        i0 = int(ds.meta.episodes["dataset_from_index"][ep])
        f0 = ds[i0]
        recorded = np.stack([ds[i0 + t]["action"].numpy() for t in range(N)])
        task = f0["task"]

        sim.reset([cfg])
        live_imgs = sim.images()
        live_state = sim.state()[0]
        img_diff = {c: float(np.abs(live_imgs[c][0].astype(float) - f0[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).mean())
                    for c in CAMERAS}
        state_diff = float(np.abs(live_state - f0["observation.state"].numpy()).max())

        pred_ds = predict(policy, {c: f0[f"observation.images.{c}"].numpy() for c in CAMERAS}, f0["observation.state"].numpy(), task)
        pred_live = predict(policy, {c: live_imgs[c][0].transpose(2, 0, 1) / 255.0 for c in CAMERAS}, live_state, task)
        hold = np.abs(recorded[:, RIGHT] - f0["observation.state"].numpy()[RIGHT]).mean()
        err = lambda p: float(np.abs(p[:, RIGHT] - recorded[:, RIGHT]).mean())
        print(f"episode {ep} ({e['object']}, layout {e['layout_id']}): \"{task}\"")
        print(f"  image diff live vs dataset (mean |px|, 0-255): " + ", ".join(f"{c.split('_')[0]} {v:.1f}" for c, v in img_diff.items()))
        print(f"  state diff (max |rad|): {state_diff:.4f}")
        print(f"  right-arm action error over first {N} steps (rad): dataset frame {err(pred_ds):.4f} | live frame {err(pred_live):.4f}"
              f" | hold-still baseline {hold:.4f}")
        print(f"  t=24 right-arm targets: expert {np.round(recorded[-1, RIGHT], 2)}")
        print(f"                          pred(ds) {np.round(pred_ds[-1, RIGHT], 2)}")
        print(f"                          pred(live) {np.round(pred_live[-1, RIGHT], 2)}")


if __name__ == "__main__":
    main()
