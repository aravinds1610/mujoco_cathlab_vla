"""CPU MuJoCo twin of the right-arm task scene: plans episodes and is the CPU side of sim2sim.

Same model as Newton (task_scene.build_task_spec), same episode parameters (episode_params),
same stage code (task_stages). One world per instance.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from cathlab_sim.newton.task_scene import (OBJECTS, TABLE_TOP, build_task_spec, episode_params, grasp_point,
                                           load_variation)
from cathlab_sim.newton.task_stages import instant_flags
from cathlab_sim.robot_io import RobotIO
from cathlab_sim.scene import CAMERAS, IMAGE_SIZE, set_home

SETTLE_FRAMES = 15  # 0.5 s at 30 Hz, robot holding HOME_Q, before the episode starts
CONTROL_HZ = 30


def home_action() -> np.ndarray:
    from cathlab_sim.scene import HOME_Q
    return np.concatenate([HOME_Q, [1.0], HOME_Q, [1.0]]).astype(np.float32)


class CpuTaskSim:
    def __init__(self, variation=None, render: bool = False, image_size: int = IMAGE_SIZE):
        self.variation = variation or load_variation()
        self.model = build_task_spec(self.variation).compile()
        self.data = mujoco.MjData(self.model)
        self.io = RobotIO(self.model)
        self.renderer = mujoco.Renderer(self.model, image_size, image_size) if render else None
        m = self.model
        self.obj_body = {n: m.body(n).id for n in OBJECTS}
        self.obj_geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj_body[n]] for n in OBJECTS}
        self.base_mass = m.body_mass.copy()
        self.base_inertia = m.body_inertia.copy()
        self.pinch = m.site("right_gripper_pinch").id
        self.driver_q = m.joint("right_gripper_right_driver_joint").qposadr[0]
        self.light = m.light("key_light").id
        self.base_light = m.light_diffuse[self.light].copy()

    def reset(self, cfg, jitter=(0.0, 0.0), right_q=None) -> None:
        """right_q: optional start pose (7 joints) for the right arm instead of HOME_Q (near-start episodes)."""
        m, d = self.model, self.data
        p = episode_params(self.variation, cfg, jitter)
        self.cfg, self.target = cfg, cfg.object
        self.center_z = self.variation.objects[cfg.object].center_z()
        mujoco.mj_resetData(m, d)
        m.body_mass[:] = self.base_mass
        m.body_inertia[:] = self.base_inertia
        for n in OBJECTS:
            b = self.obj_body[n]
            m.body_mass[b] *= p["mass_scale"]
            m.body_inertia[b] *= p["mass_scale"]
            for g in self.obj_geoms[n]:
                m.geom_friction[g, 0] = p["friction"]
                m.geom_rgba[g] = p["rgba"][n]
            adr = m.jnt_qposadr[m.body_jntadr[b]]
            d.qpos[adr:adr + 3], d.qpos[adr + 3:adr + 7] = p["poses"][n]
        m.geom_rgba[m.geom("table").id] = p["rgba"]["table"]
        m.light_diffuse[self.light] = np.clip(self.base_light * p["light_gain"], 0, 1)
        set_home(m, d)
        self.last_action = home_action()
        if right_q is not None:
            for i, v in enumerate(right_q, start=1):
                d.qpos[m.joint(f"right_fr3v2_joint{i}").qposadr[0]] = v
                d.ctrl[m.actuator(f"right_fr3v2_joint{i}").id] = v
            self.last_action[8:15] = right_q
            mujoco.mj_forward(m, d)
        for _ in range(SETTLE_FRAMES):
            self.step(self.last_action)

    def step(self, action: np.ndarray) -> np.ndarray:
        self.last_action = np.asarray(action, np.float32)
        return self.io.step(self.data, self.last_action)

    def state(self) -> np.ndarray:
        return self.io.get_state(self.data)

    def task_obs(self) -> dict:
        """Ground truth for the expert and the stage predicates (batched with N = 1)."""
        d, b = self.data, self.obj_body[self.target]
        pos, quat = d.xpos[b].copy(), d.xquat[b].copy()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        return {"target_pos": pos[None], "target_quat": quat[None], "grasp_pos": grasp_point(self.target, pos, quat)[None],
                "target_speed": np.array([np.linalg.norm(vel[3:])]), "pinch_pos": d.site_xpos[self.pinch][None].copy(),
                "driver_q": np.array([d.qpos[self.driver_q]]), "grip_cmd": np.array([self.last_action[15]]),
                "tcp_pos": d.site("right_tcp").xpos[None].copy(),
                "objects": {n: (d.xpos[self.obj_body[n]].copy(), d.xquat[self.obj_body[n]].copy()) for n in OBJECTS}}

    def flags(self, obs=None) -> dict:
        o = obs or self.task_obs()
        return instant_flags(o["target_pos"], o["grasp_pos"], o["target_speed"], o["pinch_pos"], o["driver_q"],
                             o["grip_cmd"], self.center_z, TABLE_TOP, self.variation.scene.tray_center)

    def images(self) -> dict[str, np.ndarray]:
        out = {}
        for cam in CAMERAS:
            self.renderer.update_scene(self.data, camera=cam)
            out[cam] = self.renderer.render().copy()
        return out
