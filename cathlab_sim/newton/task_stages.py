"""Stage predicates for the right-arm task, from poses and gripper joints only (numpy, batched).

Both engines (CPU MuJoCo and Newton/MuJoCo-Warp) report the same few quantities, so success is
judged by identical code; that is what makes the sim2sim comparison fair. v1's stages.py used
MuJoCo contact lists; here contact is approximated geometrically (see the table).

| Stage     | Predicate                                                                  |
|-----------|----------------------------------------------------------------------------|
| reach     | right pinch point within 5 cm of the grasp point                           |
| grasp     | gripper commanded closed, stopped short of fully closed, grasp point < 3 cm |
| lift      | target lowest point > table + 5 cm                                         |
| transport | target xy within the tray                                                  |
| place     | in the tray, resting on its floor, slow, gripper open, pinch > 4 cm away   |

Success (handoff A6): every stage latched in order AND the place condition has held for
SUCCESS_HOLD consecutive frames.
"""
import numpy as np

from cathlab_sim.contracts import STAGES
from cathlab_sim.robot_io import GRIPPER_CLOSED_RAD
from cathlab_sim.scene import TRAY_HALF

REACH_DIST = 0.05
GRASP_DIST = 0.03
GRASP_BLOCKED = 0.05     # rad short of fully closed = something between the fingers
LIFT_HEIGHT = 0.05
REST_SPEED = 0.02        # m/s
RELEASE_DIST = 0.04
GRIPPER_OPEN_STATE = 0.75
PLACE_Z_TOL = 0.015
SUCCESS_HOLD = 15        # frames (0.5 s at 30 Hz)
TRAY_FLOOR_TOP = 0.010   # above the table top (scene._add_tray: floor box 2 x 5 mm)


def instant_flags(target_pos, grasp_pos, target_speed, pinch_pos, driver_q, grip_cmd,
                  center_z, table_top, tray_xy, tray_half=TRAY_HALF, floor_top=TRAY_FLOOR_TOP) -> dict[str, np.ndarray]:
    """All inputs batched over worlds (leading axis N). center_z = resting height of the object centre
    above its support; grip_cmd = dataset gripper action (1 open, 0 closed). Returns {stage: bool[N]}."""
    gap = np.linalg.norm(pinch_pos - grasp_pos, axis=-1)
    in_tray = np.all(np.abs(target_pos[:, :2] - np.asarray(tray_xy)) <= np.asarray(tray_half), axis=-1)
    grip_state = 1 - driver_q / GRIPPER_CLOSED_RAD
    on_floor = np.abs(target_pos[:, 2] - (table_top + floor_top + center_z)) < PLACE_Z_TOL
    return {
        "reach": gap < REACH_DIST,
        "grasp": (grip_cmd < 0.5) & (driver_q < GRIPPER_CLOSED_RAD - GRASP_BLOCKED) & (gap < GRASP_DIST),
        "lift": target_pos[:, 2] - center_z > table_top + LIFT_HEIGHT,
        "transport": in_tray,
        "place": in_tray & on_floor & (target_speed < REST_SPEED) & (grip_state > GRIPPER_OPEN_STATE)
                 & (gap > RELEASE_DIST),
    }


class StageTracker:
    """Latches stages in order per world and applies the success hold."""

    def __init__(self, n: int):
        self.flags = {s: np.zeros(n, bool) for s in STAGES}
        self.place_run = np.zeros(n, int)
        self.success = np.zeros(n, bool)

    def update(self, instant: dict[str, np.ndarray]) -> np.ndarray:
        """Advance one frame; returns the per-world success flag (latched once the hold is met)."""
        ok = np.ones_like(self.flags["reach"])
        for s in STAGES:
            self.flags[s] = ok & (self.flags[s] | instant[s])
            ok = self.flags[s]
        self.place_run = np.where(instant["place"] & ok, self.place_run + 1, 0)
        self.success |= ok & (self.place_run >= SUCCESS_HOLD)
        return self.success.copy()

    def failure_stage(self, i: int) -> str | None:
        return next((s for s in STAGES if not self.flags[s][i]), None)
