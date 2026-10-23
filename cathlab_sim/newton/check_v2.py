"""v2 checks.

    python -m cathlab_sim.newton.check_v2 assets      # phase 1: objects load and rest in CPU MuJoCo and Newton
    python -m cathlab_sim.newton.check_v2 expert      # phase 2: every scenario (arm x object x start x layouts)

assets: every instrument rests (drift <= 2 mm over 300 frames) on the table and on the tray, settles after a 5 cm
drop, no NaN, in both engines; Newton renders saved next to CPU MuJoCo renders. -> reports/v2_assets_check.json
expert: {left, right} x {4 instruments} x {home, near-start} x layouts, planned on CPU, executed in CPU MuJoCo and
Newton; per-cell success and failure stages; contact sheet. -> reports/v2_expert_check.json
"""
import argparse
import json
import time

import numpy as np
from PIL import Image

from cathlab_sim.newton.instruments import INSTRUMENTS, PLACE_Y, TRAY_CENTER, TRAY_FLOOR, V2Task
from cathlab_sim.newton.task_scene import TABLE_TOP
from cathlab_sim.scene import CAMERAS
from cathlab_sim.spec_io import REPO

DRIFT_MAX_MM = 2.0
REST_FRAMES = 300


class _PoseOverride(V2Task):
    """V2Task whose episode poses can be replaced (tray and drop tests)."""
    override: dict = {}

    def episode_params(self, cfg, jitter=(0.0, 0.0)):
        p = super().episode_params(cfg, jitter)
        p["poses"].update(self.override.get(cfg.seed, {}))
        return p


def _scenarios(task):
    """(label, cfg, pose override) per world: each layout on the table; each object alone on the tray; each dropped."""
    base = task.configs(0)
    out = []
    for i in range(4):  # four layouts as recorded
        out.append((f"table_layout{base[i].layout_id}", base[i], {}))
    park = {n: (np.array([-1.5, -0.6 + 0.4 * k, 0.001]), np.array([1.0, 0, 0, 0])) for k, n in enumerate(INSTRUMENTS)}
    for k, name in enumerate(INSTRUMENTS):  # on the tray, the others parked
        cfg = base[0].model_copy(update={"seed": 900_000 + k, "object": name})
        pose = dict(park)
        pose[name] = (np.array([TRAY_CENTER[0], PLACE_Y * (1 if k % 2 else -1), TABLE_TOP + TRAY_FLOOR + 0.001]),
                      np.array([1.0, 0, 0, 0]))
        out.append((f"tray_{name}", cfg, pose))
    for k, name in enumerate(INSTRUMENTS):  # dropped from 5 cm onto the table
        cfg = base[1].model_copy(update={"seed": 910_000 + k, "object": name})
        x, y, yaw = task.layouts[cfg.layout_id][name]
        out.append((f"drop_{name}", cfg, {name: (np.array([x, y, TABLE_TOP + 0.05]),
                                                 np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))}))
    return out


def check_assets():
    from cathlab_sim.newton.cpu_sim import CpuTaskSim, home_action
    from cathlab_sim.newton.scene import NewtonTaskSim
    task = _PoseOverride()
    sc = _scenarios(task)
    task.override = {cfg.seed: pose for _, cfg, pose in sc}
    res = {"scenarios": [s[0] for s in sc], "cpu": {}, "newton": {}}

    # CPU MuJoCo, one scenario at a time
    cpu = CpuTaskSim(task=task, render=True)
    for label, cfg, _ in sc:
        cpu.reset(cfg)
        if label.startswith("drop"):
            for _ in range(30):
                cpu.step(home_action())
        p0 = {n: cpu.task_obs()["objects"][n][0].copy() for n in INSTRUMENTS}
        for _ in range(REST_FRAMES):
            cpu.step(home_action())
        p1 = cpu.task_obs()["objects"]
        drift = max(float(np.linalg.norm(p1[n][0] - p0[n])) * 1000 for n in INSTRUMENTS)
        nan = bool(np.isnan(cpu.data.qpos).any())
        res["cpu"][label] = {"max_drift_mm": round(drift, 3), "nan": nan,
                             "z_mm": {n: round(float(p1[n][0][2] - TABLE_TOP) * 1000, 1) for n in INSTRUMENTS}}
        if label == sc[0][0]:
            cpu_imgs = cpu.images()

    # Newton, all scenarios in parallel worlds
    t0 = time.time()
    sim = NewtonTaskSim(len(sc), task=task)
    res["newton_build_s"] = round(time.time() - t0, 1)
    sim.reset([cfg for _, cfg, _ in sc])
    hold = np.tile(home_action(), (len(sc), 1))
    drop = np.array([lab.startswith("drop") for lab, _, _ in sc])
    for _ in range(30):
        sim.step(hold)
    imgs = sim.images()
    p0 = {n: sim.task_obs()["objects"][n][0].copy() for n in INSTRUMENTS}
    for _ in range(REST_FRAMES):
        st = sim.step(hold)
    p1 = sim.task_obs()["objects"]
    for w, (label, _, _) in enumerate(sc):
        drift = max(float(np.linalg.norm(p1[n][0][w] - p0[n][w])) * 1000 for n in INSTRUMENTS)
        res["newton"][label] = {"max_drift_mm": round(drift, 3), "nan": bool(np.isnan(st[w]).any()),
                                "z_mm": {n: round(float(p1[n][0][w][2] - TABLE_TOP) * 1000, 1) for n in INSTRUMENTS}}

    # images: Newton worlds (external + both wrists) and the CPU render of world 0
    rows = [np.concatenate([imgs[c][w] for c in CAMERAS], axis=1) for w in range(min(4, len(sc)))]
    Image.fromarray(np.concatenate(rows, axis=0)).save(REPO / "reports/v2_assets_newton.png")
    pair = np.concatenate([np.concatenate([cpu_imgs[c] for c in CAMERAS], axis=1),
                           np.concatenate([imgs[c][0] for c in CAMERAS], axis=1)], axis=0)
    Image.fromarray(pair).save(REPO / "reports/v2_assets_cpu_vs_newton.png")
    ok = lambda r: r["max_drift_mm"] <= DRIFT_MAX_MM and not r["nan"]
    res["pass_cpu"] = all(ok(r) for r in res["cpu"].values())
    res["pass_newton"] = all(ok(r) for r in res["newton"].values())
    res["drop_settled"] = bool(drop.any())
    res["pass"] = res["pass_cpu"] and res["pass_newton"]
    (REPO / "reports/v2_assets_check.json").write_text(json.dumps(res, indent=2))
    for eng in ("cpu", "newton"):
        print(eng, {k: (v["max_drift_mm"], v["nan"]) for k, v in res[eng].items()})
    print("PASS" if res["pass"] else "FAIL", "| cpu", res["pass_cpu"], "| newton", res["pass_newton"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["assets", "expert"])
    ap.add_argument("--objects", nargs="+", default=None, help="expert: only these instruments")
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--out-name", default="v2_expert_check")
    args = ap.parse_args()
    if args.what == "assets":
        check_assets()
    else:
        check_expert(args.per_cell, objects=args.objects, out_name=args.out_name)


def _cells(task, per_cell: int):
    """{(side, object, start): [cfg, ...]} from the train grid, per_cell configs each."""
    cells = {}
    for cfg in task.configs(0):
        for start in ("home", "near"):
            key = (task.side(cfg), cfg.object, start)
            if len(cells.setdefault(key, [])) < per_cell:
                cells[key].append(cfg)
    return cells


def check_expert(per_cell: int = 5, worlds: int = 20, procs: int = 12, objects=None, out_name="v2_expert_check"):
    from cathlab_sim.newton.runner import make_pool, plan_one, run_newton_batch
    from cathlab_sim.newton.scene import NewtonTaskSim
    task = V2Task()
    cells = {k: v for k, v in _cells(task, per_cell).items() if objects is None or k[1] in objects}
    jobs, keys, cfgs = [], [], []
    for key, cl in sorted(cells.items()):
        for cfg in cl:
            jobs.append({"cfg": cfg.model_dump(), "run_cpu": True, "near": key[2] == "near"})
            keys.append(key)
            cfgs.append(cfg)
    t0 = time.time()
    pool = make_pool(procs, task_name="v2")
    plans = pool.map(plan_one, jobs)
    pool.close()
    plan_s = time.time() - t0
    sim = NewtonTaskSim(worlds, task=task, render=False)
    res = []
    t0 = time.time()
    for i in range(0, len(cfgs), worlds):
        res += run_newton_batch(sim, cfgs[i:i + worlds], [p["actions"] for p in plans[i:i + worlds]],
                                arm_qs=[p["arm_q"] for p in plans[i:i + worlds]])
    exec_s = time.time() - t0
    table, fails = {}, []
    for key in sorted(cells):
        idx = [i for i, k in enumerate(keys) if k == key]
        n_ok = sum(res[i]["success"] for i in idx)
        c_ok = sum(plans[i]["cpu_success"] for i in idx)
        near_ok = sum(plans[i]["arm_q"] is not None for i in idx) if key[2] == "near" else len(idx)
        table["/".join(key)] = {"newton": f"{n_ok}/{len(idx)}", "cpu": f"{c_ok}/{len(idx)}",
                                "newton_rate": n_ok / len(idx), "near_start_found": f"{near_ok}/{len(idx)}",
                                "newton_failures": [res[i]["failure_stage"] for i in idx if not res[i]["success"]],
                                "cpu_failures": [plans[i]["cpu_failure_stage"] for i in idx if not plans[i]["cpu_success"]],
                                "sides_used": sorted({plans[i]["side"] for i in idx})}
        fails += [{"cell": "/".join(key), "layout": cfgs[i].layout_id, "seed": cfgs[i].seed,
                   "newton": res[i]["failure_stage"], "cpu": plans[i]["cpu_failure_stage"]}
                  for i in idx if not (res[i]["success"] and plans[i]["cpu_success"])]
    report = {"episodes": len(cfgs), "plan_s": round(plan_s, 1), "newton_exec_s": round(exec_s, 1), "cells": table,
              "newton_total": f"{sum(r['success'] for r in res)}/{len(res)}",
              "cpu_total": f"{sum(p['cpu_success'] for p in plans)}/{len(plans)}",
              "pass": all(v["newton_rate"] >= 0.8 for v in table.values()), "failed_episodes": fails}
    (REPO / f"reports/{out_name}.json").write_text(json.dumps(report, indent=2))
    for k, v in table.items():
        print(f"{k:32s} newton {v['newton']:5s} cpu {v['cpu']:5s} near {v['near_start_found']:5s} "
              f"fails {v['newton_failures'] or ''} {v['cpu_failures'] or ''}")
    print("TOTAL newton", report["newton_total"], "| cpu", report["cpu_total"], "| PASS" if report["pass"] else "| FAIL")
    if objects is None:
        _contact_sheet(task, cells, plans, keys, cfgs)


def _contact_sheet(task, cells, plans, keys, cfgs):
    """One mid-carry frame (external + acting wrist) per (side, object), home start, recorded in Newton."""
    from cathlab_sim.newton.runner import run_newton_batch
    from cathlab_sim.newton.scene import NewtonTaskSim
    pick = [next(i for i, k in enumerate(keys) if k == (s, o, "home")) for s in ("left", "right") for o in task.targets]
    sim = NewtonTaskSim(len(pick), task=task)
    res = run_newton_batch(sim, [cfgs[i] for i in pick], [plans[i]["actions"] for i in pick], record=True)
    tiles = []
    for j, i in enumerate(pick):
        r = res[j]["record"]
        t = int(len(r["action"]) * 0.42)  # lifted, on the way to the tray
        wrist = "left_wrist_camera" if keys[i][0] == "left" else "right_wrist_camera"
        tiles.append(np.concatenate([r["external_camera"][t], r[wrist][t]], axis=1))
    n = len(task.targets)
    rows = [np.concatenate(tiles[k:k + n], axis=1) for k in (0, n)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(REPO / "reports/v2_expert_contact_sheet.png")


if __name__ == "__main__":
    main()
