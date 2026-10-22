"""Seeded random object layouts on the instrument table.

The model always contains every object; objects not used in an episode are parked
on the floor behind the robot (out of camera view), so the compiled model never changes.
"""
import math

import mujoco
import numpy as np

from cathlab_sim.scene import TABLE_TOP
from cathlab_sim.spec_io import ObjectDef

PARK_X = -1.5  # behind the robot


def _corners(x, y, yaw, half):
    c, s = math.cos(yaw), math.sin(yaw)
    ax, ay = np.array([c, s]) * half[0], np.array([-s, c]) * half[1]
    centre = np.array([x, y])
    return np.array([centre + ax + ay, centre - ax + ay, centre - ax - ay, centre + ax - ay])


def _overlap(a, b, margin):
    """Separating-axis test for two rectangles (corner arrays), with a safety margin."""
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
            pa, pb = a @ axis, b @ axis
            if pa.max() + margin < pb.min() or pb.max() + margin < pa.min():
                return False
    return True


def sample_layout(rng: np.random.Generator, randomization: dict, objects: dict[str, ObjectDef],
                  target: str, max_tries: int = 200) -> dict[str, tuple[float, float, float]]:
    """Return {name: (x, y, yaw)} for the target plus n random distractors, non-overlapping."""
    if target not in objects:
        raise ValueError(f"unknown target {target!r}; scene has {sorted(objects)}")
    lo_n, hi_n = (int(v) for v in randomization["n_distractors"])
    others = [n for n in objects if n != target]
    n = int(rng.integers(lo_n, min(hi_n, len(others)) + 1))
    chosen = [target] + list(rng.choice(others, size=n, replace=False))
    (x0, x1), (y0, y1), (w0, w1) = randomization["area_x"], randomization["area_y"], randomization["yaw"]
    margin = randomization.get("margin", [0.0])[0]

    placed, rects = {}, []
    for name in chosen:
        half = objects[name].footprint
        for _ in range(max_tries):
            x, y, yaw = rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(w0, w1)
            r = _corners(x, y, yaw, half)
            inside = r[:, 0].min() >= x0 - 0.05 and r[:, 0].max() <= x1 + 0.05 and \
                r[:, 1].min() >= y0 - 0.05 and r[:, 1].max() <= y1 + 0.05
            if inside and not any(_overlap(r, o, margin) for o in rects):
                placed[str(name)] = (x, y, yaw)
                rects.append(r)
                break
        else:
            if name == target:
                raise RuntimeError(f"could not place target {target!r} in {max_tries} tries")
    return placed


def _flat_quat(obj: ObjectDef, yaw: float) -> np.ndarray:
    q_yaw = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    if obj.type == "box":
        return q_yaw
    # capsule axis is local z: lay it along local x first (rotate +90 deg about y), then yaw
    q_lay = np.array([math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q_yaw, q_lay)
    return out


def apply_layout(model, data, layout: dict, objects: dict[str, ObjectDef]) -> None:
    """Write object poses into data.qpos; unused objects are parked off-table."""
    for i, (name, obj) in enumerate(objects.items()):
        adr = model.joint(f"{name}_joint").qposadr[0]
        if name in layout:
            x, y, yaw = layout[name]
            pos = [x, y, TABLE_TOP + obj.half_height + 0.002]
        else:
            yaw = 0.0
            pos = [PARK_X, -1.0 + 0.35 * i, obj.half_height + 0.002]
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = _flat_quat(obj, yaw)
        data.qvel[model.joint(f"{name}_joint").dofadr[0]:][:6] = 0
    mujoco.mj_forward(model, data)
