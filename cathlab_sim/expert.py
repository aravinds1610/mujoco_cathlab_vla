"""Scripted pick-and-place expert (v1): ground-truth object pose + IK, one arm.

Plan: above object -> descend -> close -> lift -> over handover tray -> lower -> open -> retreat.
Fingers close across the object's narrow side. Each move is a straight line in Cartesian
space, solved with IK at every 30 Hz frame; the other arm holds the ready pose.

Usage: python -m cathlab_sim.expert --n 10 [--target syringe]   # success rate over seeds
"""
import argparse
import math

import mujoco
import numpy as np

from cathlab_sim.ik import solve_ik
from cathlab_sim.robot_io import CONTROL_HZ
from cathlab_sim.scene import HOME_Q, TABLE_TOP, TRAYS
from cathlab_sim.spec_io import ObjectDef

# Menagerie 2F-85 pad tips sit 46 mm (open) / 59 mm (closed) beyond the TCP marker,
# and the lower pad box is centred ~50 mm beyond it when closed.
PAD_CENTRE_CLOSED = 0.050
TIP_CLOSED = 0.059
TABLE_CLEARANCE = 0.002
PREGRASP_H, LIFT_H, CARRY_H, RETREAT_H = 0.12, 0.15, 0.15, 0.12
FAST, SLOW = 0.20, 0.05  # m/s
DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # TCP z pointing down, yaw 0


def _yaw_quat(yaw):
    """TCP pointing down, rotated by yaw about world z."""
    q, out = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), np.zeros(4)
    mujoco.mju_mulQuat(out, q, DOWN)
    return out


def _slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def long_axis_yaw(obj: ObjectDef, quat: np.ndarray) -> float:
    """Yaw (rad) of the object's long horizontal axis in the world."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    if obj.type == "capsule":
        axis = R[:, 2]                                   # capsule axis is local z
    else:
        axis = R[:, 0] if obj.size[0] >= obj.size[1] else R[:, 1]
    return math.atan2(axis[1], axis[0])


class ScriptedExpert:
    def __init__(self, model: mujoco.MjModel, side: str = "left"):
        self.model, self.side = model, side
        self.arm = slice(0, 7) if side == "left" else slice(8, 15)
        self.grip = 7 if side == "left" else 15

    def _best_quat(self, data, pos, yaws, q_init):
        """Among candidate yaws, pick the IK solution with small error and least joint motion."""
        best = None
        for yaw in yaws:
            quat = _yaw_quat(yaw)
            q, pe, re = solve_ik(self.model, data, self.side, pos, quat, q_init=q_init)
            score = (pe > 0.005 or re > 0.03, np.abs(q - q_init).sum())
            if best is None or score < best[0]:
                best = (score, quat, q)
        return best[1], best[2]

    def plan(self, data: mujoco.MjData, obs: dict, obj: ObjectDef) -> np.ndarray:
        """Return a (T, 16) array of 30 Hz actions for the whole pick-and-place."""
        p_obj, q_obj = obs["privileged.target_pos"], obs["privileged.target_quat"]
        tcp0 = obs[f"privileged.{self.side}_tcp_pos"]
        # fingers (TCP y) close across the long axis: TCP yaw = long-axis yaw (+180 deg is equivalent)
        a = long_axis_yaw(obj, q_obj)
        grasp_z = max(p_obj[2] + PAD_CENTRE_CLOSED, TABLE_TOP + TIP_CLOSED + TABLE_CLEARANCE)
        above = np.array([p_obj[0], p_obj[1], p_obj[2] + PREGRASP_H])
        grasp = np.array([p_obj[0], p_obj[1], grasp_z])
        lift = np.array([p_obj[0], p_obj[1], grasp_z + LIFT_H])
        tx, ty = TRAYS["handover_tray"]
        place_z = TABLE_TOP + 0.010 + obj.half_height + 0.015 + (grasp_z - p_obj[2])
        over = np.array([tx, ty, place_z + CARRY_H])
        place = np.array([tx, ty, place_z])

        q = np.array(HOME_Q)
        pick_quat, q = self._best_quat(data, above, [a, a + math.pi], q)
        place_quat, _ = self._best_quat(data, over, [a + k * math.pi / 2 for k in range(4)], q)

        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, data.site(f"{self.side}_tcp").xmat)
        self.actions, self.q, self.g = [], np.array(HOME_Q), 1.0
        self._move(data, tcp0, above, cur_quat, pick_quat, FAST)
        self._move(data, above, grasp, pick_quat, pick_quat, SLOW)
        self._gripper(data, 0.0, ramp_s=0.5, hold_s=0.4)
        self._move(data, grasp, lift, pick_quat, pick_quat, SLOW)
        self._move(data, lift, over, pick_quat, place_quat, FAST)
        self._move(data, over, place, place_quat, place_quat, SLOW)
        self._gripper(data, 1.0, ramp_s=0.3, hold_s=0.3)
        self._move(data, place, place + [0, 0, RETREAT_H], place_quat, place_quat, FAST)
        self._hold(0.7)  # let the object settle on the tray
        return np.array(self.actions, dtype=np.float32)

    # -- trajectory building -------------------------------------------
    def _emit(self):
        a = np.concatenate([HOME_Q, [1.0], HOME_Q, [1.0]])
        a[self.arm], a[self.grip] = self.q, self.g
        self.actions.append(a)

    def _move(self, data, p0, p1, q0, q1, speed):
        n = max(10, int(np.linalg.norm(p1 - p0) / speed * CONTROL_HZ))
        for k in range(1, n + 1):
            t = k / n
            self.q, _, _ = solve_ik(self.model, data, self.side, p0 + t * (p1 - p0), _slerp(q0, q1, t),
                                    q_init=self.q, iters=60)
            self._emit()

    def _gripper(self, data, target, ramp_s, hold_s):
        g0, n = self.g, int(ramp_s * CONTROL_HZ)
        for k in range(1, n + 1):
            self.g = g0 + (target - g0) * k / n
            self._emit()
        self._hold(hold_s)

    def _hold(self, seconds):
        for _ in range(int(seconds * CONTROL_HZ)):
            self._emit()


def run_episode(backend, scene, seed, target, side="left"):
    """Plan once from the reset observation and execute open-loop. Returns the final StepInfo."""
    obs = backend.reset(scene, seed, target)
    actions = ScriptedExpert(backend.model, side).plan(backend.data, obs, backend.objects[target])
    info = None
    for a in actions:
        obs, info = backend.step(a)
    return info, len(actions)


def main():
    from cathlab_sim.backend import MujocoBackend
    from cathlab_sim.contracts import STAGES
    from cathlab_sim.spec_io import load_scene_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="seeds per object")
    ap.add_argument("--target", default=None, help="one object (default: all)")
    args = ap.parse_args()
    scene = load_scene_spec()
    be = MujocoBackend(render_images=False)
    targets = [args.target] if args.target else list(scene.objects)
    total = 0
    for t in targets:
        fails = {}
        wins = 0
        for seed in range(args.n):
            info, n = run_episode(be, scene, seed, t)
            wins += info.success
            if not info.success:
                stage = next(s for s in STAGES if not info.stage_flags[s])
                fails[stage] = fails.get(stage, 0) + 1
        total += wins
        print(f"{t:10s} {wins}/{args.n} success  failures by stage: {fails or '-'}")
    print(f"overall {total}/{args.n * len(targets)}")


if __name__ == "__main__":
    main()
