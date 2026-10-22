"""Dataset QA report (challenge stage 02, "dataset intelligence") for a generated LeRobot dataset.

    python -m cathlab_sim.dataset_qa datasets/handover_mujoco_scripted_v1 --out reports/qa_v1

Read-only. Checks balance (objects, instructions), episode lengths, coverage of target
position / yaw / clutter (layouts are rebuilt from each episode's seed, no physics), gripper
chatter, action jumps, idle action columns, and how well the arm tracks its commands.
Writes <out>.json and <out>.png (target positions and yaw histogram).
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cathlab_sim.layout import sample_layout
from cathlab_sim.robot_io import NAMES
from cathlab_sim.spec_io import load_object_defs, load_scene_spec

GRID = 4                 # target position coverage: GRID x GRID cells over the source area
YAW_BINS = 8
JUMP_RAD = 0.1           # an action step bigger than this (rad per frame = 3 rad/s) is a jump
LEFT_GRIPPER = NAMES.index("left_gripper")
LEFT_JOINTS = list(range(7))


def load(root: Path):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)))
    extra = [json.loads(l) for l in (root / "meta/cathlab_episodes_extra.jsonl").read_text().splitlines()]
    return df.sort_values(["episode_index", "frame_index"]), extra


def coverage(saved, scene):
    r = scene.randomization
    objs = load_object_defs(scene)
    (x0, x1), (y0, y1), (w0, w1) = r["area_x"], r["area_y"], r["yaw"]
    rows = []
    for e in saved:
        lay = sample_layout(np.random.default_rng(e["seed"]), r, objs, e["target"])
        x, y, yaw = lay[e["target"]]
        rows.append({"target": e["target"], "x": x, "y": y, "yaw": yaw, "n_distractors": len(lay) - 1})
    pos = pd.DataFrame(rows)
    cx = np.clip(((pos.x - x0) / (x1 - x0) * GRID).astype(int), 0, GRID - 1)
    cy = np.clip(((pos.y - y0) / (y1 - y0) * GRID).astype(int), 0, GRID - 1)
    pos["cell"] = cx * GRID + cy
    pos["yaw_bin"] = np.clip(((pos.yaw - w0) / (w1 - w0) * YAW_BINS).astype(int), 0, YAW_BINS - 1)
    per_obj = {o: {"n": len(g), "cells_covered": f"{g.cell.nunique()}/{GRID * GRID}",
                   "yaw_bins_covered": f"{g.yaw_bin.nunique()}/{YAW_BINS}"} for o, g in pos.groupby("target")}
    return pos, {"area": {"x": [x0, x1], "y": [y0, y1], "yaw": [w0, w1]},
                 "all_cells_covered": f"{pos.cell.nunique()}/{GRID * GRID}",
                 "min_episodes_in_a_cell": int(pos.cell.value_counts().reindex(range(GRID * GRID), fill_value=0).min()),
                 "n_distractors": dict(sorted(Counter(pos.n_distractors).items())), "per_object": per_obj}


def motion(df):
    chatter, jumps, track = [], [], []
    for _, ep in df.groupby("episode_index"):
        a = np.stack(ep["action"].values)
        s = np.stack(ep["observation.state"].values)
        closed = a[:, LEFT_GRIPPER] < 0.5
        chatter.append(int(np.count_nonzero(closed[1:] != closed[:-1])))  # a clean grasp is 2: close, open
        jumps.append(float(np.abs(np.diff(a[:, LEFT_JOINTS], axis=0)).max()))
        track.append(float(np.abs(a[:-1, LEFT_JOINTS] - s[1:, LEFT_JOINTS]).mean()))
    a_all = np.stack(df["action"].values)
    idle = [NAMES[j] for j in range(16) if a_all[:, j].std() < 1e-3]
    return {"gripper_transitions": dict(sorted(Counter(chatter).items())),
            "episodes_with_chatter": sum(c > 2 for c in chatter),
            "max_joint_step_rad": {"median": float(np.median(jumps)), "max": float(np.max(jumps))},
            "episodes_with_jumps": sum(j > JUMP_RAD for j in jumps),
            "tracking_error_rad": {"mean": float(np.mean(track)), "max": float(np.max(track))},
            "idle_action_columns": idle}


def plot(pos, cov, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.5))
    for o, g in pos.groupby("target"):
        ax.scatter(g.y, g.x, s=14, label=f"{o} ({len(g)})")
    (x0, x1), (y0, y1) = cov["area"]["x"], cov["area"]["y"]
    for k in range(GRID + 1):
        ax.axhline(x0 + k * (x1 - x0) / GRID, c="0.85", lw=0.8, zorder=0)
        ax.axvline(y0 + k * (y1 - y0) / GRID, c="0.85", lw=0.8, zorder=0)
    ax.set(xlabel="y (m, left is +)", ylabel="x (m, away from robot)", title="target positions")
    ax.legend(fontsize=7)
    bx.hist([g.yaw for _, g in pos.groupby("target")], bins=YAW_BINS, stacked=True, label=list(pos.groupby("target").groups))
    bx.set(xlabel="target yaw (rad)", ylabel="episodes", title="target yaw")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output prefix (writes .json and .png)")
    args = ap.parse_args()

    df, extra = load(args.root)
    saved = sorted((e for e in extra if e["saved"]), key=lambda e: e["episode_index"])
    lengths = df.groupby("episode_index").size()
    assert len(saved) == len(lengths), "episode metadata and parquet disagree"
    pos, cov = coverage(saved, load_scene_spec())
    report = {
        "dataset": str(args.root), "episodes": len(saved), "frames": int(len(df)),
        "attempts": len(extra), "expert_success": sum(e["success"] for e in extra),
        "per_object_saved": dict(Counter(e["target"] for e in saved)),
        "per_object_failed_at": {o: dict(Counter(e["failure_stage"] for e in extra if e["target"] == o and not e["success"]))
                                 for o in sorted({e["target"] for e in extra})},
        "instruction_templates": dict(Counter(e["instruction"].rsplit(" the ", 1)[0] for e in saved)),
        "unique_seeds": len({e["seed"] for e in saved}) == len(saved),
        "episode_seconds": {k: round(float(v) / 30, 2) for k, v in lengths.describe()[["min", "50%", "max"]].items()},
        "coverage": cov, "motion": motion(df),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    plot(pos, cov, args.out.with_suffix(".png"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
