"""Stage predicates (handoff §2) computed from MuJoCo state.

| Stage     | Predicate                                                |
|-----------|----------------------------------------------------------|
| reach     | a TCP within 5 cm of the target                          |
| grasp     | a gripper closed and touching the target                 |
| lift      | target's lowest point > table + 5 cm                     |
| transport | target xy within the handover tray                       |
| place     | target resting on the handover tray, grippers open, free |

`instant_flags` evaluates them on the current state; `latch` turns that into the
episode-level flags: a stage counts once achieved, and only after the previous one.
"""
import mujoco
import numpy as np

from cathlab_sim.contracts import STAGES
from cathlab_sim.robot_io import GRIPPER_CLOSED_RAD
from cathlab_sim.scene import SIDES, TABLE_TOP, TRAY_HALF, TRAYS

REACH_DIST = 0.05
LIFT_HEIGHT = 0.05
GRIPPER_CLOSED_CMD = 0.5   # dataset gripper action below this = "closing"
GRIPPER_OPEN_STATE = 0.75  # dataset gripper state above this = "open"
REST_SPEED = 0.02          # m/s


class StageEvaluator:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.tcp = {s: model.site(f"{s}_tcp").id for s in SIDES}
        self.driver_q = {s: model.joint(f"{s}_gripper_right_driver_joint").qposadr[0] for s in SIDES}
        self.grip_act = {s: model.actuator(f"{s}_gripper_fingers_actuator").id for s in SIDES}
        bodies_of = lambda prefix: {b for b in range(model.nbody) if model.body(b).name.startswith(prefix)}
        geoms_in = lambda bodies: {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}
        self.gripper_geoms = {s: geoms_in(bodies_of(f"{s}_gripper_")) for s in SIDES}
        self.arm_geoms = {s: geoms_in(bodies_of(f"{s}_fr3v2_")) - self.gripper_geoms[s] for s in SIDES}
        self.robot_geoms = {s: self.arm_geoms[s] | self.gripper_geoms[s] for s in SIDES}
        names = ["table", "pedestal", "floor"] + [f"handover_tray_{p}" for p in ["floor", "wall0", "wall1", "wall2", "wall3"]]
        self.env_geoms = {model.geom(n).id for n in names}
        self.tray_floor = model.geom("handover_tray_floor").id
        self.tray_xy, self.tray_half = np.array(TRAYS["handover_tray"]), np.array(TRAY_HALF)

    def set_target(self, name: str, half_height: float):
        self.target_body = self.model.body(name).id
        self.target_geom = self.model.geom(f"{name}_geom").id
        self.target_half_h = half_height

    def _contacts(self, data):
        return [(c.geom1, c.geom2) for c in data.contact[:data.ncon]]

    def instant_flags(self, data: mujoco.MjData) -> dict[str, bool]:
        p = data.xpos[self.target_body]
        contacts = self._contacts(data)
        touching = lambda geoms: any((a == self.target_geom and b in geoms) or (b == self.target_geom and a in geoms)
                                     for a, b in contacts)
        closing = {s: data.ctrl[self.grip_act[s]] / 255.0 > 1 - GRIPPER_CLOSED_CMD for s in SIDES}
        is_open = {s: 1 - data.qpos[self.driver_q[s]] / GRIPPER_CLOSED_RAD > GRIPPER_OPEN_STATE for s in SIDES}
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.target_body, vel, 0)
        in_tray = bool(np.all(np.abs(p[:2] - self.tray_xy) <= self.tray_half))
        flags = {
            "reach": min(np.linalg.norm(data.site_xpos[self.tcp[s]] - p) for s in SIDES) < REACH_DIST,
            "grasp": any(closing[s] and touching(self.gripper_geoms[s]) for s in SIDES),
            "lift": p[2] - self.target_half_h > TABLE_TOP + LIFT_HEIGHT,
            "transport": in_tray,
            "place": in_tray and touching({self.tray_floor}) and np.linalg.norm(vel[3:]) < REST_SPEED
                     and all(is_open.values()) and not any(touching(self.gripper_geoms[s]) for s in SIDES),
        }
        return {k: bool(v) for k, v in flags.items()}  # plain bools for pydantic

    def collision(self, data: mujoco.MjData) -> bool:
        for a, b in self._contacts(data):
            for s in SIDES:
                if (a in self.arm_geoms[s] and b in self.env_geoms) or (b in self.arm_geoms[s] and a in self.env_geoms):
                    return True
            l, r = self.robot_geoms["left"], self.robot_geoms["right"]
            if (a in l and b in r) or (a in r and b in l):
                return True
        return False


def latch(prev: dict[str, bool], instant: dict[str, bool]) -> dict[str, bool]:
    """A stage latches True once achieved, but only after every earlier stage has."""
    out, ok_so_far = {}, True
    for s in STAGES:
        out[s] = ok_so_far and (prev[s] or instant[s])
        ok_so_far = out[s]
    return out
