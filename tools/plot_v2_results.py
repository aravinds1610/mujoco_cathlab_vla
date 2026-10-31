"""Plots for docs/V2_RESULTS.md from the training logs, eval reports and eval traces (run in .venv-eval).

    python tools/plot_v2_results.py
"""
import json
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [  # label, report, trace
    ("v2, home start", "reports/v2_eval_038400.json", "outputs/eval_traces/v2_038400.npz"),
    ("v2.5, home start", "reports/v2_5_eval_019200.json", "outputs/eval_traces/v2_5_019200.npz"),
    ("v2.5, near start", "reports/v2_5_eval_019200_near.json", "outputs/eval_traces/v2_5_019200_near.npz"),
]
STAGES = ["reach", "grasp", "lift", "transport", "place"]
BLUE, ORANGE, GREEN, GREY = "#2a6fdb", "#e8833a", "#3a9e5f", "#888888"


def losses(path, every=200):
    text = open(path, errors="replace").read()
    vals = [float(v) for v in re.findall(r" loss:([0-9.]+) grdn", text)]
    return np.arange(1, len(vals) + 1) * every, np.array(vals)


def plot_loss():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, log, colour in [("v2 (2 epochs)", "outputs/smolvla_instruments_v1.log", BLUE),
                               ("v2.5 (1 epoch, side in prompt)", "outputs/smolvla_v2_5.log", ORANGE)]:
        s, v = losses(log)
        ax.plot(s, v, color=colour, lw=1.5, label=label)
    ax.axvline(19_213, color=GREY, ls="--", lw=1)
    ax.text(19_600, 0.3, "end of epoch 1", color=GREY, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("training step (batch 12)")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title("Training loss kept falling; success did not follow", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("reports/v2_results_loss.png", dpi=130)


def plot_funnel():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.27
    for k, ((label, rep, _), colour) in enumerate(zip(RUNS, [BLUE, ORANGE, GREEN])):
        eps = json.load(open(rep))["episodes"]
        passed = []
        for i, stage in enumerate(STAGES):
            passed.append(sum(e["success"] or (e["failure_stage"] in STAGES and
                                               STAGES.index(e["failure_stage"]) > i) for e in eps))
        x = np.arange(len(STAGES)) + (k - 1) * width
        ax.bar(x, passed, width, color=colour, label=label)
        for xi, p in zip(x, passed):
            ax.text(xi, p + 0.5, str(p), ha="center", fontsize=7)
    ax.set_xticks(range(len(STAGES)), [f"passed\n{s}" for s in STAGES])
    ax.set_ylabel("episodes (of 48)")
    ax.set_title("Stage funnel: most episodes never complete the reach", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("reports/v2_results_funnel.png", dpi=130)


def plot_gripper_timing():
    """When does the gripper first close, and how far from the instrument is it then?"""
    import glob
    root = "datasets/cathlab_fr3_instruments_v1"
    extra = {e["episode_index"]: e for e in map(json.loads, open(f"{root}/meta/episodes_extra.jsonl"))
             if e["episode_index"] is not None}
    data = pd.concat(pd.read_parquet(p) for p in glob.glob(f"{root}/data/*/*.parquet"))
    expert = []
    for ep, g in data.groupby("episode_index"):
        if extra[ep]["start"] == "home":
            a = np.stack(g.action.to_numpy())
            expert.append(np.argmax(a[:, 7 if extra[ep]["side"] == "left" else 15] < 0.5))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axvspan(np.percentile(expert, 5), np.percentile(expert, 95), color=GREY, alpha=0.25)
    ax.text(np.median(expert), 30.5, "expert (home start) closes here,\nat the grasp point (~0 cm)", ha="center",
            fontsize=8, color="#444444")
    for (label, rep, trace), colour in zip(RUNS[1:], [ORANGE, GREEN]):
        eps, z = json.load(open(rep))["episodes"], np.load(trace)
        t, d = [], []
        for i, e in enumerate(eps):
            a, gap = z[f"ep{i}_action"], z[f"ep{i}_gap"] * 100
            closed = np.where(a[:, 7 if e["side"] == "left" else 15] < 0.5)[0]
            if len(closed):
                t.append(closed[0])
                d.append(gap[closed[0]])
        ax.scatter(t, d, s=16, color=colour, alpha=0.8, label=f"{label}: model's first close")
    ax.set_xlabel("frame of the first gripper close (30 fps)")
    ax.set_ylabel("gripper-to-instrument distance (cm)")
    ax.set_title("The model closes the gripper on a clock, centimetres away from the instrument", fontsize=10)
    ax.set_ylim(0, 34)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("reports/v2_results_gripper_timing.png", dpi=130)


if __name__ == "__main__":
    plot_loss()
    plot_funnel()
    plot_gripper_timing()
    print("wrote reports/v2_results_{loss,funnel,gripper_timing}.png")
