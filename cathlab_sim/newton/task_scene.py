"""Right-arm FR3 Duo task scene: the team objects (bottle, vial, syringe) and a tray within the
right arm's reach. Pure MjSpec, so the CPU MuJoCo planner and Newton load the same model.

Robot, cameras, table and tray come from cathlab_sim.scene (v1); only the task layout differs:
the right arm picks from the robot's right (-y) and places on a tray at +y (v1 mirrored).
Object geometry follows the VariationSpec (configs/variation/cathlab_fr3_newton.yaml).
"""
import math
import re
from pathlib import Path

import mujoco
import numpy as np

from cathlab_sim import scene as v1
from cathlab_sim.newton.variation import ObjectSpec, VariationSpec
from cathlab_sim.spec_io import REPO

DEFAULT_VARIATION = REPO / "configs/variation/cathlab_fr3_newton.yaml"
TABLE_TOP = v1.TABLE_TOP
OBJECTS = ("bottle", "vial", "syringe")
# Syringe composite (handoff A2): barrel 15 x 100 mm + plunger 8 x 40 mm + flange 30 x 10 x 3 mm,
# 140 mm long, lying along body x with the origin at the composite's centre. The barrel is a capsule
# (capsule-box contacts are analytic; a lying cylinder balances on one GJK contact point).
SYRINGE_BARREL_X = -0.02          # barrel centre in the body frame (the grasp point)
SYRINGE_PLUNGER_X = 0.05
SYRINGE_FLANGE_X = 0.03
SOL_DEFAULTS = {"solref": [0.02, 1.0], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0]}


def load_variation(path: str | Path = DEFAULT_VARIATION) -> VariationSpec:
    return VariationSpec.load(path)


def _add_object(world, name: str, obj: ObjectSpec, rgba, park_xy):
    body = world.add_body(name=name, pos=[park_xy[0], park_xy[1], obj.center_z() + 0.002])
    body.add_freejoint(name=f"{name}_joint")
    friction = [0.9, 0.005, 0.0001]
    if obj.shape == "cylinder":
        # MuJoCo-Warp gives cylinder-box pairs one contact point, so an upright cylinder rocks on the
        # table. A thin box "foot" inscribed in the base (box-box: multi-point contact) carries it;
        # the cylinder, raised 0.5 mm, still provides the side contacts for the gripper.
        half_l, foot_h, foot_w = obj.length / 2, 0.001, obj.radius / math.sqrt(2)
        body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.00025],
                      size=[obj.radius, half_l - 0.00025, 0], rgba=rgba, mass=0.95 * obj.mass, friction=friction)
        body.add_geom(name=f"{name}_foot", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -half_l + foot_h],
                      size=[foot_w, foot_w, foot_h], rgba=rgba, mass=0.05 * obj.mass, friction=friction)
        return
    along_x = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]  # geom z axis -> body x
    barrel_m, plunger_m, flange_m = 0.8 * obj.mass, 0.15 * obj.mass, 0.05 * obj.mass
    body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE, pos=[SYRINGE_BARREL_X, 0, 0],
                  quat=along_x, size=[obj.radius, 0.05 - obj.radius, 0], rgba=rgba, mass=barrel_m, friction=friction)
    body.add_geom(name=f"{name}_plunger", type=mujoco.mjtGeom.mjGEOM_CAPSULE, pos=[SYRINGE_PLUNGER_X, 0, 0],
                  quat=along_x, size=[0.004, 0.016, 0], rgba=rgba, mass=plunger_m, friction=friction)
    body.add_geom(name=f"{name}_flange", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[SYRINGE_FLANGE_X, 0, 0],
                  size=[0.0015, 0.015, 0.005], rgba=rgba, mass=flange_m, friction=friction)


def build_task_spec(variation: VariationSpec | None = None) -> mujoco.MjSpec:
    """v1 robot + table + cameras, a right-arm tray, and the team objects (parked behind the robot)."""
    variation = variation or load_variation()
    spec = build_base_spec(variation.table_colors[0])
    world = spec.worldbody
    v1._add_tray(world, "handover_tray", variation.scene.tray_center)
    for i, name in enumerate(OBJECTS):
        _add_object(world, name, variation.objects[name], variation.obj_colors[0][name], (-1.5, -0.5 + 0.35 * i))
    return spec


def build_base_spec(table_rgba) -> mujoco.MjSpec:
    """Robot (FR3 Duo, both arms), head camera, light, floor, pedestal and table: shared by every task."""
    spec = mujoco.MjSpec()
    spec.modelname = "fr3_duo_cathlab_fr3_newton"
    spec.option.timestep = 1 / 600  # 20 physics steps per 30 Hz control frame
    # Same integrator as Newton's SolverMuJoCo default, so CPU and GPU runs differ only in the engine
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Multi-point contacts for flat faces (an upright cylinder on a box otherwise rocks on one point).
    # Default-on since MuJoCo 3.4 (it became a disable flag); opt-in before that.
    if hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
        spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    world = spec.worldbody

    world.add_light(name="key_light", pos=[0.5, 0, 2.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.55, 0.55, 0.6, 1])
    world.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.25, v1.MOUNT_Z / 2],
                   pos=[-0.15, 0, v1.MOUNT_Z / 2], rgba=[0.9, 0.9, 0.92, 1])
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.65, TABLE_TOP / 2],
                   pos=[0.60, 0, TABLE_TOP / 2], rgba=list(table_rgba), friction=[0.9, 0.005, 0.0001])

    for side in v1.SIDES:
        base = v1.REF["arm_bases"][side]
        pos = np.array(base["pos"]) + [0, 0, v1.MOUNT_Z]
        mount = world.add_frame(pos=pos, quat=v1._xyaxes_to_quat(base["xyaxes"]))
        mount.attach_body(v1._arm_with_gripper().body("fr3v2_link0"), f"{side}_", "")

    head = v1.REF["cameras"]["external_camera"]
    world.add_camera(name="external_camera", pos=np.array(head["pos"]) + [0, 0, v1.MOUNT_Z],
                     xyaxes=head["xyaxes"], fovy=v1.CAM_FOVY["external_camera"])
    return spec


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    """Approximate blackbody colour (Tanner Helland's fit), normalised so 6500 K is ~white."""
    def raw(k):
        t = k / 100.0
        r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
        g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
        return np.clip([r, g, b], 0, 255) / 255.0
    return raw(kelvin) / raw(6500.0)


def episode_params(variation: VariationSpec, cfg, jitter=(0.0, 0.0)) -> dict:
    """World-frame physical values for one EpisodeConfig (shared by both engines).

    poses: {object: (pos[3], quat_wxyz[4])} with z on the table top; rgba: {object|'table': rgba};
    light_gain: linear RGB multiplier (intensity x colour temperature). `jitter` shifts the whole layout
    in xy (the "more episodes" seeds), so objects keep their separation.
    """
    from cathlab_sim.newton.variation import object_poses
    poses = {}
    for name, p in object_poses(variation, cfg).items():
        poses[name] = (np.array([p[0] + jitter[0], p[1] + jitter[1], p[2] + TABLE_TOP + 0.0005]), np.array(p[3:7]))
    rgba = dict(variation.obj_colors[cfg.obj_color_id])
    rgba["table"] = variation.table_colors[cfg.table_color_id]
    return {"poses": poses, "rgba": rgba, "mass_scale": cfg.mass_scale, "friction": cfg.friction,
            "light_gain": cfg.light_intensity * kelvin_to_rgb(cfg.light_temp_k)}


BOTTLE_GRASP_Z = 0.152 / 2 - 0.03  # above the bottle centre: pads ~3 cm below its top


def grasp_point(name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """World position the gripper closes on: the vial centre, the syringe barrel centre, or the
    bottle's upper body (the 77 mm bottle is grasped from above, near its top)."""
    if name == "vial":
        return np.asarray(pos, float)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, float))
    mat = mat.reshape(3, 3)
    if name == "syringe":
        return np.asarray(pos, float) + mat[:, 0] * SYRINGE_BARREL_X
    return np.asarray(pos, float) + mat[:, 2] * BOTTLE_GRASP_Z


def _pad_sol(match: re.Match) -> str:
    attr, values = match.group(1), [float(v) for v in match.group(2).split()]
    full = SOL_DEFAULTS["solref" if attr.startswith("solref") else "solimp"]
    values += full[len(values):]
    return f'{attr}="{" ".join(repr(v) for v in values)}"'


def to_newton_xml(spec: mujoco.MjSpec) -> str:
    """MJCF string that Newton's importer reads the same way MuJoCo does.

    Two fixes over spec.to_xml(): mesh paths are made absolute (the arm and the gripper have
    different meshdirs, which to_xml drops), and solref/solimp attributes are padded to full
    length. to_xml omits trailing values equal to MuJoCo's defaults ("0.005 1" -> "0.005");
    MuJoCo refills them, Newton reads them as 0 (broken joint limits on the gripper).
    """
    dirs = [v1.ARM_XML.parent / "assets", v1.GRIPPER_XML.parent / "assets"]
    for mesh in spec.meshes:
        if not Path(mesh.file).is_absolute():
            mesh.file = str(next(d / mesh.file for d in dirs if (d / mesh.file).exists()))
    xml = spec.to_xml()
    # solreffriction/solimpfriction are left alone: their default "0 0" means "use solref"
    return re.sub(r'\b(sol(?:ref|imp)(?:limit)?)="([^"]*)"', _pad_sol, xml)


INSTRUCTIONS = ("Hand me the {object}", "Please pass the {object}", "Can you give me the {object}")


class V1Task:
    """v1 task (right arm, bottle / vial / syringe primitives, tray at +y) behind the interface the simulators,
    expert and generator use. V2Task (instruments.py) implements the same members."""
    name = "v1"
    objects = OBJECTS
    tray_floor_top = 0.010  # v1 scene._add_tray floor top above the table

    def __init__(self, variation: VariationSpec | None = None):
        self.variation = variation or load_variation()
        self.tray_center = tuple(self.variation.scene.tray_center)
        self.tray_half = tuple(v1.TRAY_HALF)

    def build_spec(self) -> mujoco.MjSpec:
        return build_task_spec(self.variation)

    def episode_params(self, cfg, jitter=(0.0, 0.0)) -> dict:
        return episode_params(self.variation, cfg, jitter)

    def grasp_point(self, name, pos, quat_wxyz) -> np.ndarray:
        return grasp_point(name, pos, quat_wxyz)

    def rest_z(self, name: str) -> float:
        return self.variation.objects[name].center_z()

    def side(self, cfg) -> str:
        return "right"

    def instruction(self, cfg) -> str:
        return INSTRUCTIONS[cfg.seed % 3].format(object=cfg.object)

    def configs(self, seed: int = 0) -> list:
        from cathlab_sim.newton.variation import generation_order, train_configs
        return generation_order(self.variation, train_configs(self.variation, seed=seed), seed=0)

    def eval_configs(self, tier: str, seeds: list[int]) -> list:
        from cathlab_sim.newton.variation import eval_configs
        return eval_configs(self.variation, tier, seeds)

    def grasp_plan(self, target: str, obs: dict) -> dict:
        """Grasp point, TCP height and yaws, and the place pose, for the scripted expert."""
        from cathlab_sim.expert import PAD_CENTRE_CLOSED, TABLE_CLEARANCE, TIP_CLOSED
        spec = self.variation.objects[target]
        g, q_obj = obs["grasp_pos"][0], obs["target_quat"][0]
        grasp_z = max(g[2] + PAD_CENTRE_CLOSED, TABLE_TOP + TIP_CLOSED + TABLE_CLEARANCE)
        if spec.upright:
            yaws = [k * math.pi / 4 for k in range(8)]
            place_yaws = yaws
        else:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q_obj)
            axis = R.reshape(3, 3)[:, 0]
            a = math.atan2(axis[1], axis[0])
            yaws = [a, a + math.pi]
            place_yaws = [a + k * math.pi / 2 for k in range(4)]
        return {"grasp": g, "grasp_z": grasp_z, "yaws": yaws, "place_xy": self.tray_center, "place_yaws": place_yaws,
                "rest_z": spec.center_z()}
