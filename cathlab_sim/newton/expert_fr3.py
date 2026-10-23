"""Scripted expert (either arm) for the Newton tasks, planned on CPU MuJoCo.

Same kinematic plan as v1 (cathlab_sim.expert): above -> descend -> close -> lift -> over the tray
-> lower -> open -> retreat, IK at every 30 Hz frame from ground-truth poses. Differences:
- right arm; tray from the VariationSpec;
- per-object grasp height: vial and syringe as v1 (pads at the object centre, fingertips clear of
  the table); the 77 mm bottle (85 mm opening) is grasped from above across its body, pads ~3 cm
  below its top;
- upright cylinders have no long axis: try 8 yaws and keep the IK solution with least joint motion;
  the syringe closes across its barrel (TCP yaw = long-axis yaw), at the barrel centre.

    python -m cathlab_sim.newton.expert_fr3 --probe       # N2 reach probe
"""
import argparse
import math

import mujoco
import numpy as np

from cathlab_sim.expert import FAST, PREGRASP_H, RETREAT_H, SLOW, \
    ScriptedExpert, _yaw_quat
from cathlab_sim.ik import solve_ik
from cathlab_sim.newton.task_scene import TABLE_TOP
from cathlab_sim.scene import HOME_Q

LIFT_H, CARRY_H = 0.15, 0.12
PLACE_DROP = 0.012  # release this far above the resting height


class TaskExpert(ScriptedExpert):
    """Scripted pick-and-place for any task (task_scene.V1Task, instruments.V2Task) with either arm."""

    def __init__(self, model: mujoco.MjModel, task, side: str = "right"):
        super().__init__(model, side=side)
        self.task = task

    def plan(self, data: mujoco.MjData, obs: dict, target: str) -> np.ndarray:
        """obs = CpuTaskSim.task_obs() for this arm (batch of 1). Returns (T, 16) float32 actions at 30 Hz."""
        gp = self.task.grasp_plan(target, obs)
        g, grasp_z, tcp0 = gp["grasp"], gp["grasp_z"], obs["tcp_pos"][0]
        centre_z = obs["target_pos"][0][2]
        above = np.array([g[0], g[1], grasp_z + PREGRASP_H])
        grasp = np.array([g[0], g[1], grasp_z])
        lift = grasp + [0, 0, LIFT_H]
        tx, ty = gp["place_xy"]
        place_z = TABLE_TOP + self.task.tray_floor_top + gp["rest_z"] + PLACE_DROP + (grasp_z - centre_z)
        over = np.array([tx, ty, place_z + CARRY_H])
        place = np.array([tx, ty, place_z])

        # start from wherever this arm is (HOME_Q, or a near-start pose)
        q0 = np.array([data.joint(f"{self.side}_fr3v2_joint{i}").qpos[0] for i in range(1, 8)])
        pick_quat, q = self._best_quat(data, above, [_yaw_quat(y) for y in gp["yaws"]], q0.copy())
        place_quat, _ = self._best_quat(data, over, [_yaw_quat(y) for y in gp["place_yaws"]], q)
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, data.site(f"{self.side}_tcp").xmat)
        self.actions, self.q, self.g = [], q0, 1.0
        self._move(data, tcp0, above, cur_quat, pick_quat, FAST)
        self._move(data, above, grasp, pick_quat, pick_quat, SLOW)
        self._gripper(data, 0.0, ramp_s=0.5, hold_s=0.4)
        self._move(data, grasp, lift, pick_quat, pick_quat, SLOW)
        self._move(data, lift, over, pick_quat, place_quat, FAST)
        self._move(data, over, place, place_quat, place_quat, SLOW)
        self._gripper(data, 1.0, ramp_s=0.3, hold_s=0.3)
        self._move(data, place, place + [0, 0, RETREAT_H], place_quat, place_quat, FAST)
        self._hold(0.7)  # object settles on the tray; also covers the 15-frame success hold
        return np.array(self.actions, dtype=np.float32)

    def _best_quat(self, data, pos, quats, q_init):
        """v1's selection, over explicit quaternions."""
        best = None
        for quat in quats:
            q, pe, re = solve_ik(self.model, data, self.side, pos, quat, q_init=q_init)
            score = (pe > 0.005 or re > 0.03, np.abs(q - q_init).sum())
            if best is None or score < best[0]:
                best = (score, quat, q)
        return best[1], best[2]


def reach_probe(variation=None) -> dict:
    """Right-arm IK to the pick-region corners and the tray centre, 5 cm above the table, gripper down."""
    from cathlab_sim.newton.cpu_sim import CpuTaskSim
    sim = CpuTaskSim(variation)
    v = sim.variation.scene
    (cx, cy), (sx, sy) = v.pick_region_center, v.pick_region_size
    points = {f"corner_{i}": (cx + dx * sx / 2, cy + dy * sy / 2)
              for i, (dx, dy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)])}
    points["tray_centre"] = tuple(v.tray_center)
    out = {}
    for name, (x, y) in points.items():
        target = np.array([x, y, TABLE_TOP + 0.05])
        best = min((solve_ik(sim.model, sim.data, "right", target, _yaw_quat(k * math.pi / 4), q_init=np.array(HOME_Q))
                    for k in range(8)), key=lambda r: r[1] + 0.1 * r[2])
        out[name] = {"xy": [round(x, 3), round(y, 3)], "pos_err_mm": round(1000 * best[1], 2),
                     "rot_err_deg": round(math.degrees(best[2]), 2), "ok": bool(best[1] < 0.01)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.probe:
        import json
        print(json.dumps(reach_probe(), indent=1))


if __name__ == "__main__":
    main()
