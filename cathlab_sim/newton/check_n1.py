"""N1 go/no-go: the right-arm task scene in Newton.

    python -m cathlab_sim.newton.check_n1 [--worlds 4]

Checks, on the first `worlds` train configs (generation order):
  1. forward kinematics matches CPU MuJoCo (right flange + pinch point at home);
  2. objects rest: drift <= 2 mm after 300 frames holding HOME_Q, no NaN;
  3. the right gripper closes and opens on an empty hand;
  4. 3-camera images per world, saved next to the CPU MuJoCo renders of the same config.
Writes reports/newton_n1_check.json and reports/newton_n1_*.png.
"""
import argparse
import json
import time

import numpy as np
from PIL import Image

from cathlab_sim.newton.cpu_sim import CpuTaskSim, home_action
from cathlab_sim.newton.scene import NewtonTaskSim
from cathlab_sim.newton.task_scene import OBJECTS
from cathlab_sim.newton.variation import generation_order, train_configs
from cathlab_sim.scene import CAMERAS
from cathlab_sim.spec_io import REPO

DRIFT_MAX_MM = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=4)
    args = ap.parse_args()
    out = REPO / "reports"
    t0 = time.time()
    sim = NewtonTaskSim(args.worlds)
    build_s = time.time() - t0
    cfgs = generation_order(sim.variation, train_configs(sim.variation))[:args.worlds]
    sim.reset(cfgs)
    res = {"worlds": args.worlds, "build_s": round(build_s, 1), "configs": [c.model_dump() for c in cfgs]}

    # 1. FK vs CPU MuJoCo, world 0
    cpu = CpuTaskSim(render=True)
    cpu.reset(cfgs[0])
    ng, cg = sim.task_obs(), cpu.task_obs()
    res["fk_err_mm"] = {"tcp": round(1000 * float(np.linalg.norm(ng["tcp_pos"][0] - cg["tcp_pos"][0])), 3),
                        "pinch": round(1000 * float(np.linalg.norm(ng["pinch_pos"][0] - cg["pinch_pos"][0])), 3)}
    res["state_err_rad"] = round(float(np.abs(sim.state()[0] - cpu.state()).max()), 5)

    # 4a. images right after the reset (same instant as the CPU render)
    imgs = sim.images()
    cpu_imgs = cpu.images()
    rows = [np.concatenate([imgs[c][w] for c in CAMERAS], axis=1) for w in range(args.worlds)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / "newton_n1_worlds.png")
    pair = np.concatenate([np.concatenate([cpu_imgs[c] for c in CAMERAS], axis=1),
                           np.concatenate([imgs[c][0] for c in CAMERAS], axis=1)], axis=0)
    Image.fromarray(pair).save(out / "newton_n1_cpu_vs_newton.png")
    res["image_mean_abs_diff"] = {c: round(float(np.abs(imgs[c][0].astype(int) - cpu_imgs[c].astype(int)).mean()), 1)
                                  for c in CAMERAS}

    # 2. rest: hold home for 300 frames
    p0 = {n: sim.task_obs()["objects"][n][0].copy() for n in OBJECTS}
    hold = np.tile(home_action(), (args.worlds, 1))
    t = time.time()
    for _ in range(300):
        state = sim.step(hold)
    step_s = (time.time() - t) / 300
    p1 = sim.task_obs()["objects"]
    drift = {n: np.linalg.norm(p1[n][0] - p0[n], axis=-1) * 1000 for n in OBJECTS}
    res["drift_mm"] = {n: [round(float(v), 3) for v in drift[n]] for n in OBJECTS}
    res["max_drift_mm"] = round(float(max(d.max() for d in drift.values())), 3)
    res["nan"] = bool(np.isnan(state).any() or np.isnan(sim.states[0].body_q.numpy()).any())
    res["frame_ms"] = round(1000 * step_s, 2)
    res["world_frames_per_s"] = round(args.worlds / step_s, 1)

    # 3. right gripper: close over 0.5 s, hold 0.5 s, open, hold 0.5 s
    trace = []
    for k in range(60):
        a = hold.copy()
        a[:, 15] = max(0.0, 1 - k / 15)
        trace.append(float(sim.step(a)[0, 15]))
    closed = trace[-1]
    for k in range(45):
        a = hold.copy()
        a[:, 15] = min(1.0, k / 10)
        trace.append(float(sim.step(a)[0, 15]))
    res["gripper_state_closed"] = round(closed, 3)
    res["gripper_state_reopened"] = round(trace[-1], 3)
    res["gripper_ok"] = closed < 0.1 and trace[-1] > 0.9

    # CPU reference for the same gripper motion
    for k in range(60):
        a = home_action()
        a[15] = max(0.0, 1 - k / 15)
        cpu.step(a)
    res["cpu_gripper_state_closed"] = round(float(cpu.state()[15]), 3)

    res["pass"] = bool(res["max_drift_mm"] <= DRIFT_MAX_MM and not res["nan"] and res["gripper_ok"]
                       and res["fk_err_mm"]["pinch"] < 1.0)
    (out / "newton_n1_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "configs"}, indent=1))


if __name__ == "__main__":
    main()
