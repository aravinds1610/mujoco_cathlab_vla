"""Minimal FR3 Duo cathlab scene built with MjSpec.

Robot embodiment copied from the Siemens FR3 Duo (see isaac_reference.json):
V-mount arm bases, gripper finger orientation, head + wrist camera mounts.
Everything else (table, trays, objects) is our own and will be randomized.
World frame: Z-up, robot faces +X, robot's left = +Y.
"""
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from cathlab_sim.contracts import SceneSpec
from cathlab_sim.spec_io import load_object_defs, load_scene_spec

PKG = Path(__file__).resolve().parent


def _find_menagerie() -> Path:
    """MuJoCo Menagerie checkout: $CATHLAB_MENAGERIE, external/mujoco_menagerie, or ContactPilot's copy."""
    env = os.environ.get("CATHLAB_MENAGERIE")
    candidates = ([Path(env)] if env else []) + [PKG.parent / "external/mujoco_menagerie",
                                                 PKG.parent / "external/ContactPilot/mujoco_menagerie"]
    for c in candidates:
        if (c / "franka_fr3_v2/fr3v2.xml").exists():
            return c
    raise FileNotFoundError("MuJoCo Menagerie not found; run: git clone --depth 1 "
                            "https://github.com/google-deepmind/mujoco_menagerie external/mujoco_menagerie")


MENAGERIE = _find_menagerie()
ARM_XML = MENAGERIE / "franka_fr3_v2/fr3v2.xml"
GRIPPER_XML = MENAGERIE / "robotiq_2f85/2f85.xml"
REF = json.loads((PKG / "isaac_reference.json").read_text())

SIDES = ("left", "right")
MOUNT_Z = REF["mount_height_above_floor"]  # Duo mount_link height (~1.02 m)
TABLE_TOP = MOUNT_Z - 0.18                 # Isaac work surfaces sit ~0.18 m below the mount
# Ready pose from the Siemens Isaac doc: [0, -45, 0, -135, 0, 90, 45] deg
HOME_Q = np.deg2rad([0, -45, 0, -135, 0, 90, 45])
TCP_IN_LINK8 = REF["gripper"]["tcp_in_link8"]
# Menagerie 2F-85 fingers open along flange x; the Duo's open along flange y -> +90 deg about z
GRIPPER_QUAT = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
# Dataset images are square 224x224: the recorder overrides apertures to 4.8 mm
# (fr3_conversion.json), so use those FOVs rather than the USD's authored ones.
CAM_FOVY = {"external_camera": math.degrees(2 * math.atan(2.4 / 2.4)),
            "wrist": math.degrees(2 * math.atan(2.4 / 2.5290722846984863))}
IMAGE_SIZE = 224

# Task objects come from the SceneSpec YAML (configs/). Use capsules/boxes, not cylinders:
# cylinder-box contacts sank ~1 cm into the tray floor.
GEOM_TYPES = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "box": mujoco.mjtGeom.mjGEOM_BOX}
TRAYS = {"handover_tray": (0.58, -0.26)}  # within the left arm's reach (single-arm expert v1)
TRAY_HALF = (0.15, 0.115)  # ~300 x 230 mm interior, like the Isaac serving tray
TRAY_WALL_H = 0.03


def _xyaxes_to_quat(xyaxes):
    x, y = np.array(xyaxes[:3]), np.array(xyaxes[3:])
    mat = np.column_stack([x, y, np.cross(x, y)])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def _arm_with_gripper() -> mujoco.MjSpec:
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    for key in list(arm.keys):  # child keyframes clash once two arms are merged
        arm.delete(key)
    flange = arm.body("fr3v2_link8")
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    flange.add_frame(quat=GRIPPER_QUAT).attach_body(gripper.body("base_mount"), "gripper_", "")
    flange.add_site(name="tcp", pos=TCP_IN_LINK8, size=[0.005] * 3, rgba=[1, 0, 0, 0.5], group=4)
    cam = REF["cameras"]["left_wrist_camera"]  # same mount on both arms (in flange frame)
    flange.add_camera(name="wrist_camera", pos=cam["pos"], xyaxes=cam["xyaxes"], fovy=CAM_FOVY["wrist"])
    # Gravity compensation, like the real Franka controller: without it the position
    # servos sag ~0.01 rad under load (~1 cm at the gripper)
    for body in arm.bodies:
        body.gravcomp = 1.0
    return arm


def _add_tray(world, name, xy):
    body = world.add_body(name=name, pos=[xy[0], xy[1], TABLE_TOP])
    hx, hy, t, h = TRAY_HALF[0], TRAY_HALF[1], 0.005, TRAY_WALL_H
    grey = [0.7, 0.72, 0.75, 1]
    body.add_geom(name=f"{name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[hx, hy, t], pos=[0, 0, t], rgba=grey)
    for i, (px, py, sx, sy) in enumerate([(hx, 0, t, hy), (-hx, 0, t, hy), (0, hy, hx, t), (0, -hy, hx, t)]):
        body.add_geom(name=f"{name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[sx, sy, h / 2], pos=[px, py, h / 2], rgba=grey)


def build_spec(scene: SceneSpec | None = None) -> mujoco.MjSpec:
    """Build the scene. Objects start parked; layout.apply_layout places them per episode."""
    scene = scene or load_scene_spec()
    spec = mujoco.MjSpec()
    spec.modelname = "fr3_duo_cathlab_minimal"
    spec.option.timestep = 1 / 600  # 20 physics steps per 30 Hz control frame
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    world = spec.worldbody

    world.add_light(pos=[0.5, 0, 2.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.55, 0.55, 0.6, 1])

    # Pedestal whose top is the Duo mount_link
    world.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.25, MOUNT_Z / 2],
                   pos=[-0.15, 0, MOUNT_Z / 2], rgba=[0.9, 0.9, 0.92, 1])
    # Table in front of the robot
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.65, TABLE_TOP / 2],
                   pos=[0.60, 0, TABLE_TOP / 2], rgba=[0.6, 0.62, 0.65, 1], friction=[0.9, 0.005, 0.0001])

    # V-mount: arm bases 10 cm apart, each tilted ~52 deg outward (from the Isaac Duo)
    for side in SIDES:
        base = REF["arm_bases"][side]
        pos = np.array(base["pos"]) + [0, 0, MOUNT_Z]
        mount = world.add_frame(pos=pos, quat=_xyaxes_to_quat(base["xyaxes"]))
        mount.attach_body(_arm_with_gripper().body("fr3v2_link0"), f"{side}_", "")

    head = REF["cameras"]["external_camera"]
    world.add_camera(name="external_camera", pos=np.array(head["pos"]) + [0, 0, MOUNT_Z],
                     xyaxes=head["xyaxes"], fovy=CAM_FOVY["external_camera"])

    for name, xy in TRAYS.items():
        _add_tray(world, name, xy)

    for i, (name, obj) in enumerate(load_object_defs(scene).items()):
        size = obj.size + [0] * (3 - len(obj.size))
        body = world.add_body(name=name, pos=[-1.5, -1.0 + 0.35 * i, 0.05])  # parked behind the robot
        body.add_freejoint(name=f"{name}_joint")
        body.add_geom(name=f"{name}_geom", type=GEOM_TYPES[obj.type], size=size, rgba=obj.rgba,
                      mass=obj.mass, friction=obj.friction)
    return spec


CAMERAS = ("external_camera", "left_wrist_camera", "right_wrist_camera")


def set_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put both arms in the ready pose, grippers open, and hold it."""
    for side in SIDES:
        for i, q in enumerate(HOME_Q, start=1):
            jname = f"{side}_fr3v2_joint{i}"
            data.qpos[model.joint(jname).qposadr[0]] = q
            data.ctrl[model.actuator(jname).id] = q
        data.ctrl[model.actuator(f"{side}_gripper_fingers_actuator").id] = 0.0  # 0 = open
    mujoco.mj_forward(model, data)
