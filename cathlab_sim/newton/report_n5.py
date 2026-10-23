"""N5 numbers: FR3 sim2sim (CPU MuJoCo vs Newton/MuJoCo-Warp on the same planned trajectories), expert
success on train vs held-out tiers, and execution throughput without rendering.

    python -m cathlab_sim.newton.report_n5 [--sim2sim 60] [--heldout-seeds 10] [--worlds 20]

Writes reports/newton_n5.json. Rendering throughput (episodes/hour with cameras) comes from the
generate logs; see docs/NEWTON_REPORT.md.
"""
import argparse
import json
import time

import numpy as np

from cathlab_sim.newton.runner import configs, make_pool, plan_one, run_newton_batch
from cathlab_sim.newton.task_scene import load_variation
from cathlab_sim.newton.variation import TIERS, eval_configs
from cathlab_sim.spec_io import REPO

OBJECTS = ("bottle", "vial", "syringe")


def plan_and_run(sim, pool, cfgs):
    t0 = time.time()
    plans = pool.map(plan_one, [{"cfg": c.model_dump(), "run_cpu": True} for c in cfgs])
    cpu_s = time.time() - t0
    t0 = time.time()
    res = []
    for i in range(0, len(cfgs), sim.n):
        res += run_newton_batch(sim, cfgs[i:i + sim.n], [p["actions"] for p in plans[i:i + sim.n]])
    return plans, res, cpu_s, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim2sim", type=int, default=60, help="configs (from the end of the train grid)")
    ap.add_argument("--heldout-seeds", type=int, default=10, help="seeds per object per held-out tier")
    ap.add_argument("--worlds", type=int, default=20)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    from cathlab_sim.newton.scene import NewtonTaskSim
    v = load_variation()
    pool = make_pool(args.procs)
    sim = NewtonTaskSim(args.worlds, variation=v, render=False)
    report = {"worlds": args.worlds, "cpu_processes": args.procs}

    # sim2sim on train configs not used by the N2 check (which took the first 60)
    cfgs = configs(0)[-args.sim2sim:]
    plans, res, cpu_s, gpu_s = plan_and_run(sim, pool, cfgs)
    rows = []
    for c, p, r in zip(cfgs, plans, res):
        d = np.linalg.norm(np.array(p["cpu_final_target_pos"]) - np.array(r["final_target_pos"])) * 1000
        rows.append({"object": c.object, "cpu": p["cpu_success"], "newton": r["success"], "final_pos_diff_mm": d,
                     "cpu_stage": p["cpu_failure_stage"], "newton_stage": r["failure_stage"]})
    frames = sum(len(p["actions"]) for p in plans)
    report["sim2sim"] = {
        "configs": len(cfgs),
        "per_object": {o: {"cpu": f"{sum(x['cpu'] for x in rows if x['object'] == o)}/{sum(x['object'] == o for x in rows)}",
                           "newton": f"{sum(x['newton'] for x in rows if x['object'] == o)}/{sum(x['object'] == o for x in rows)}",
                           "median_final_pos_diff_mm": round(float(np.median([x['final_pos_diff_mm'] for x in rows if x['object'] == o])), 2),
                           "max_final_pos_diff_mm": round(float(max(x['final_pos_diff_mm'] for x in rows if x['object'] == o)), 2)}
                       for o in OBJECTS},
        "agreement_rate": round(float(np.mean([x["cpu"] == x["newton"] for x in rows])), 4),
        "disagreements": [x for x in rows if x["cpu"] != x["newton"]],
        "frames": frames,
        # execution only (no rendering): planning+CPU execution in the pool vs Newton execution
        "cpu_pool_plan_and_exec_s": round(cpu_s, 1), "newton_exec_s": round(gpu_s, 1),
        "newton_world_frames_per_s_no_render": round(frames / gpu_s, 1)}

    # held-out tiers (team eval_configs), seeds disjoint from training
    seeds = list(range(100_000, 100_000 + args.heldout_seeds))
    tiers = {}
    for tier in TIERS:
        cfgs = eval_configs(v, tier, seeds)
        plans, res, _, _ = plan_and_run(sim, pool, cfgs)
        tiers[tier] = {o: f"{sum(r['success'] for c, r in zip(cfgs, res) if c.object == o)}/"
                          f"{sum(c.object == o for c in cfgs)}" for o in OBJECTS}
        tiers[tier]["all"] = round(float(np.mean([r["success"] for r in res])), 3)
        tiers[tier]["cpu_all"] = round(float(np.mean([p["cpu_success"] for p in plans])), 3)
        tiers[tier]["newton_failure_stages"] = {s: sum(r["failure_stage"] == s for r in res)
                                                for s in {r["failure_stage"] for r in res} if s}
    report["tiers_newton"] = tiers
    pool.close()
    for x in report["sim2sim"]["disagreements"]:
        x["final_pos_diff_mm"] = round(float(x["final_pos_diff_mm"]), 2)
    (REPO / "reports/newton_n5.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
