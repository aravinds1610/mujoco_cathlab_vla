"""16-D action/state mapping between the FR3 Duo dataset schema and MuJoCo.

Schema (matches the Siemens LeRobot data, robot_type fr3_duo, 30 Hz):
  [left_joint1..7, left_gripper, right_joint1..7, right_gripper]
  arm joints: absolute radians (action = target, state = measured)
  gripper:    normalized, 1 = open, 0 = closed

MuJoCo side: arm joints are position actuators (ctrl = target rad);
the Robotiq 2F-85 actuator takes 0..255 (0 = open), driving its driver joint 0..0.8 rad.
"""
import mujoco
import numpy as np

from cathlab_sim.scene import SIDES

NAMES = [f"{side}_{part}" for side in SIDES for part in [f"joint{i}" for i in range(1, 8)] + ["gripper"]]
CONTROL_HZ = 30
GRIPPER_CLOSED_RAD = 0.8   # Siemens gripper_contract.closed_joint_rad; Menagerie driver range is 0..0.8
GRIPPER_CTRL_MAX = 255.0   # Menagerie 2F-85 ctrlrange


def _arm_joints(side):
    return [f"{side}_fr3v2_joint{i}" for i in range(1, 8)]


class RobotIO:
    """Caches MuJoCo indices once so per-step calls are cheap."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.act_ids, self.qpos_ids = [], []
        for side in SIDES:
            for j in _arm_joints(side):
                self.act_ids.append(model.actuator(j).id)
                self.qpos_ids.append(model.joint(j).qposadr[0])
            self.act_ids.append(model.actuator(f"{side}_gripper_fingers_actuator").id)
            self.qpos_ids.append(model.joint(f"{side}_gripper_right_driver_joint").qposadr[0])
        self.act_ids, self.qpos_ids = np.array(self.act_ids), np.array(self.qpos_ids)
        self.is_gripper = np.array([n.endswith("gripper") for n in NAMES])
        self.n_substeps = round(1.0 / CONTROL_HZ / model.opt.timestep)
        assert abs(self.n_substeps * model.opt.timestep - 1.0 / CONTROL_HZ) < 1e-9, \
            "sim timestep must divide the 30 Hz control period"

    def apply_action(self, data: mujoco.MjData, action: np.ndarray) -> None:
        """Write a 16-D dataset action into data.ctrl (does not step)."""
        a = np.asarray(action, dtype=np.float64)
        assert a.shape == (16,), a.shape
        ctrl = a.copy()
        g = np.clip(a[self.is_gripper], 0.0, 1.0)
        ctrl[self.is_gripper] = (1.0 - g) * GRIPPER_CTRL_MAX
        data.ctrl[self.act_ids] = ctrl

    def get_state(self, data: mujoco.MjData) -> np.ndarray:
        """Read the 16-D measured state (float32, dataset convention)."""
        s = data.qpos[self.qpos_ids].copy()
        s[self.is_gripper] = 1.0 - s[self.is_gripper] / GRIPPER_CLOSED_RAD
        return s.astype(np.float32)

    def step(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
        """Apply one 30 Hz action, advance the sim one control period, return the new state."""
        self.apply_action(data, action)
        mujoco.mj_step(self.model, data, nstep=self.n_substeps)
        return self.get_state(data)
