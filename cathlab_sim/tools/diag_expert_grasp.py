"""Diagnostic: why does the scripted expert fail at lift? Replays failing and passing seeds and prints
object push before closing, gripper-object contacts and grip force at lift start, and object vs TCP rise.

    PYTHONPATH=. python cathlab_sim/tools/diag_expert_grasp.py
"""
import math, sys
import mujoco, numpy as np
from cathlab_sim.backend import MujocoBackend
from cathlab_sim.expert import ScriptedExpert
from cathlab_sim.spec_io import load_scene_spec

def yaw(q): return math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

be = MujocoBackend(render_images=False); scene = load_scene_spec()
cases = {"gauze": ([14, 24, 49, 69], [4, 9]), "scissors": ([18, 48, 53, 68], [3, 8])}
print(f"{'obj':8s} {'seed':>4s} {'res':4s} | {'push_mm':>7s} {'dyaw':>5s} | {'contacts (geom: z_on_obj mm)':44s} {'F_N':>5s} {'drv':>4s} | {'rise obj/tcp mm':>15s}")
for obj, (bad, good) in cases.items():
    for seed in bad + good:
        obs = be.reset(scene, seed, obj); m = be.model
        acts = ScriptedExpert(m).plan(be.data, obs, be.objects[obj])
        ic = int(np.argmax(acts[:, 7] < 1.0)); il = ic + 15 + 12   # close start, lift start (0.5 s ramp + 0.4 s hold)
        bid = m.body(obj).id; gid = m.geom(f"{obj}_geom").id
        p0, y0 = be.data.xpos[bid].copy(), yaw(be.data.xquat[bid])
        for k, a in enumerate(acts):
            if k == ic: push = 1000 * np.linalg.norm(be.data.xpos[bid][:2] - p0[:2]); dy = yaw(be.data.xquat[bid]) - y0
            if k == il:
                pz, tz = be.data.xpos[bid][2], be.data.site("left_tcp").xpos[2]
                cs, F = [], 0.0
                for i in range(be.data.ncon):
                    c = be.data.contact[i]
                    if gid in (c.geom1, c.geom2):
                        other = c.geom2 if c.geom1 == gid else c.geom1
                        if m.geom_bodyid[other] and m.body(m.geom_bodyid[other]).name.startswith("left_gripper_"):
                            f = np.zeros(6); mujoco.mj_contactForce(m, be.data, i, f); F += abs(f[0])
                            cs.append(f"{m.body(m.geom_bodyid[other]).name.replace('left_gripper_', '')[:10]}:{1000 * (c.pos[2] - pz):+.0f}")
                drv = be.data.joint("left_gripper_right_driver_joint").qpos[0]
            if k == il + 30:
                rise = (1000 * (be.data.xpos[bid][2] - pz), 1000 * (be.data.site("left_tcp").xpos[2] - tz))
            obs, info = be.step(a)
        res = "OK" if info.success else "FAIL"
        print(f"{obj:8s} {seed:4d} {res:4s} | {push:7.1f} {dy:+5.0f} | {' '.join(cs)[:44]:44s} {F:5.1f} {drv:4.2f} | {rise[0]:6.0f} / {rise[1]:4.0f}")
