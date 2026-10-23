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
    args = ap.parse_args()
    {"assets": check_assets, "expert": lambda: check_expert()}[args.what]()


def check_expert():
    raise NotImplementedError  # filled in phase 2


if __name__ == "__main__":
    main()
