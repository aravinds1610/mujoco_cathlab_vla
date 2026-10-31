"""Copy a v2 LeRobot dataset with the acting arm appended to every instruction ("... with the left arm").

    python tools/relabel_side.py datasets/cathlab_fr3_instruments_v1 datasets/cathlab_fr3_instruments_v2_5

Videos are hard-linked (no extra disk); only the task strings change. The side comes from
meta/episodes_extra.jsonl and is cross-checked against which arm moves in the recorded actions.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    assert not os.path.exists(args.dst), f"{args.dst} exists"

    with open(f"{args.src}/meta/episodes_extra.jsonl") as f:
        lines = [json.loads(line) for line in f]  # unsaved attempts have episode_index None
    extra = {r["episode_index"]: r for r in lines if r["episode_index"] is not None}
    assert len(extra) == sum(r["episode_index"] is not None for r in lines), "repeated episode_index"
    eps = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(f"{args.src}/meta/episodes/*/*.parquet")))
    assert set(eps.episode_index) <= set(extra), "episodes without a side in episodes_extra.jsonl"

    old_tasks = pd.read_parquet(f"{args.src}/meta/tasks.parquet")
    old_names = {int(i): t for t, i in zip(old_tasks.index, old_tasks.task_index)}
    new_task = {}  # episode -> new string
    for ep, tasks in zip(eps.episode_index, eps.tasks):
        assert len(tasks) == 1 and tasks[0] == extra[ep]["instruction"], f"episode {ep}: task mismatch"
        new_task[int(ep)] = f"{tasks[0]} with the {extra[ep]['side']} arm"
    names = sorted(set(new_task.values()))
    index = {t: i for i, t in enumerate(names)}

    shutil.copytree(args.src, args.dst, copy_function=os.link, ignore=shutil.ignore_patterns("data", "meta"))
    shutil.copytree(f"{args.src}/meta", f"{args.dst}/meta")
    pd.DataFrame({"task_index": list(range(len(names)))}, index=pd.Index(names, name="task")).to_parquet(
        f"{args.dst}/meta/tasks.parquet")

    checked = mismatched = 0
    for p in sorted(glob.glob(f"{args.src}/data/*/*.parquet")):
        d = pd.read_parquet(p)
        for ep, g in d.groupby("episode_index"):
            assert old_names[int(g.task_index.iloc[0])] == extra[ep]["instruction"]
            a = np.stack(g.action.to_numpy())
            moved = "left" if np.ptp(a[:, 0:7], 0).max() > np.ptp(a[:, 8:15], 0).max() else "right"
            checked += 1
            mismatched += moved != extra[ep]["side"]
        d["task_index"] = d.episode_index.map(lambda e: index[new_task[int(e)]]).astype(d.task_index.dtype)
        out = p.replace(args.src, args.dst, 1)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        d.to_parquet(out)

    for p in sorted(glob.glob(f"{args.dst}/meta/episodes/*/*.parquet")):
        e = pd.read_parquet(p)
        e["tasks"] = e.episode_index.map(lambda ep: [new_task[int(ep)]])
        e.to_parquet(p)
    with open(f"{args.dst}/meta/info.json") as f:
        info = json.load(f)
    info["total_tasks"] = len(names)
    with open(f"{args.dst}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)
    with open(f"{args.dst}/meta/episodes_extra.jsonl", "w") as f:
        for r in lines:
            if r["episode_index"] in new_task:
                r = {**r, "instruction": new_task[r["episode_index"]]}
            f.write(json.dumps(r) + "\n")

    print(f"{len(new_task)} episodes, {len(names)} tasks; side vs moving arm: {checked - mismatched}/{checked} agree")
    for t in names:
        print(f"  {sum(v == t for v in new_task.values()):4d}  {t}")


if __name__ == "__main__":
    main()
