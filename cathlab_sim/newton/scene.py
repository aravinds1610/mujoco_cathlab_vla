"""Newton (MuJoCo-Warp solver) version of the right-arm task scene: N parallel worlds on the GPU
with batched cameras.

    sim = NewtonTaskSim(n_worlds=8)
    sim.reset(cfgs)                 # one EpisodeConfig per world
    state = sim.step(actions)       # (N, 16) dataset actions -> (N, 16) states
    imgs = sim.images()             # {camera: (N, 224, 224, 3) uint8}

The worlds share one coordinate frame (spacing 0); SolverMuJoCo(separate_worlds=True) keeps their
physics apart and the tiled camera renders each world's own shapes.

Newton workarounds (see docs/NEWTON_REPORT.md):
- task_scene.to_newton_xml fixes mesh paths and short solref/solimp attributes;
- add_mjcf does not import <camera> tags: cameras are rebuilt here from isaac_reference.json;
- the tiled renderer has no light intensity or colour: each world's linear HDR image is scaled by its
  light gain (intensity x colour temperature) before sRGB encoding, which equals scaling the lights.
"""
import math

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorTiledCamera

from cathlab_sim import scene as v1
from cathlab_sim.newton.task_scene import TABLE_TOP, V1Task, to_newton_xml
from cathlab_sim.newton.task_stages import instant_flags
from cathlab_sim.robot_io import GRIPPER_CLOSED_RAD, GRIPPER_CTRL_MAX
from cathlab_sim.scene import CAMERAS, HOME_Q, IMAGE_SIZE

PHYSICS_DT = 1 / 600
SUBSTEPS = 20        # per 30 Hz control frame
SETTLE_FRAMES = 15   # same as cpu_sim
ARM = {"left": slice(0, 7), "right": slice(8, 15)}
GRIP = {"left": 7, "right": 15}
NJMAX, NCONMAX = 600, 150  # constraint rows / contacts per world


def _mat_from_xyaxes(xyaxes) -> np.ndarray:
    x, y = np.array(xyaxes[:3], float), np.array(xyaxes[3:], float)
    x /= np.linalg.norm(x)
    y -= x * (x @ y)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw -> (..., 3, 3)."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _mat_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat().reshape(*m.shape[:-2], 4)


@wp.func
def _srgb(c: float) -> wp.uint8:
    c = wp.clamp(c, 0.0, 1.0)
    if c <= 0.0031308:
        c = 12.92 * c
    else:
        c = 1.055 * wp.pow(c, 1.0 / 2.4) - 0.055
    return wp.uint8(c * 255.0 + 0.5)


@wp.kernel
def _light_and_encode(hdr: wp.array4d(dtype=wp.vec3f), gain: wp.array(dtype=wp.vec3f),
                      out: wp.array4d(dtype=wp.uint8)):
    """Per-world light gain on the linear image, then sRGB to uint8. out: (N * C, H, W, 3) (warp's 4-D limit)."""
    w, c, y, x = wp.tid()
    v = wp.cw_mul(hdr[w, c, y, x], gain[w])
    i = w * hdr.shape[1] + c
    out[i, y, x, 0] = _srgb(v[0])
    out[i, y, x, 1] = _srgb(v[1])
    out[i, y, x, 2] = _srgb(v[2])


def build_newton(spec_xml: str, n_worlds: int):
    """One scene builder from the MJCF string, replicated n_worlds times -> (model, solver)."""
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.add_mjcf(spec_xml, ctrl_direct=True)
    # add_mjcf hides every collision geom (only <visual> classes render). The robot has separate visual
    # meshes, but the table, tray, floor and objects are collision-only: make those visible.
    for i, label in enumerate(scene.shape_label):
        if "_fr3v2_" not in label and "_gripper_" not in label and "_hull" not in label:
            scene.shape_flags[i] |= int(newton.ShapeFlags.VISIBLE)
    main = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(main)
    main.replicate(scene, n_worlds, spacing=(0.0, 0.0, 0.0))
    model = main.finalize()
    # The automatic njmax estimate (~64 constraint rows per world) overflows once the gripper and the
    # multi-point object contacts are active ("nefc overflow"), which silently drops contacts.
    solver = newton.solvers.SolverMuJoCo(model, separate_worlds=True, enable_multiccd=True,
                                         njmax=NJMAX, nconmax=NCONMAX)
    return model, solver


class NewtonTaskSim:
    def __init__(self, n_worlds: int, variation=None, render: bool = True, image_size: int = IMAGE_SIZE, task=None,
                 use_graph: bool = True):
        self.n = n_worlds
        self.task = task or V1Task(variation)
        self.variation = getattr(self.task, "variation", None)
        spec = self.task.build_spec()
        self.mj_model = spec.compile()  # reference for local frames (sites, cameras) and actuator order
        self.model, self.solver = build_newton(to_newton_xml(spec), n_worlds)
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()
        self._index()
        self._copy_robot_colors()
        self._apply_task_colors()
        self.base_mass = self.model.body_mass.numpy().copy()
        self.base_inertia = self.model.body_inertia.numpy().copy()
        self.base_color = self.model.shape_color.numpy().copy()
        self.light_gain = np.ones((n_worlds, 3), np.float32)
        self.graph = None
        self.use_graph = use_graph and wp.get_device().is_cuda
        self.image_size = image_size
        if render:
            self._setup_cameras()

    # -- indices --------------------------------------------------------
    def _per_world(self, labels, name) -> np.ndarray:
        """Index of the entity whose label ends with /name, for each world (N,)."""
        idx = [i for i, l in enumerate(labels) if l.endswith("/" + name)]
        assert len(idx) == self.n, (name, len(idx))
        return np.array(idx)

    def _index(self):
        m = self.model
        jl, bl, sl = m.joint_label, m.body_label, m.shape_label
        q_start = m.joint_q_start.numpy()
        qd_start = m.joint_qd_start.numpy()
        names16 = [f"{s}_fr3v2_joint{i}" for s in v1.SIDES for i in range(1, 8)]
        arm = {n: q_start[self._per_world(jl, n)] for n in names16}
        driver = {s: q_start[self._per_world(jl, f"{s}_gripper_right_driver_joint")] for s in v1.SIDES}
        cols = [arm[f"left_fr3v2_joint{i}"] for i in range(1, 8)] + [driver["left"]] + \
               [arm[f"right_fr3v2_joint{i}"] for i in range(1, 8)] + [driver["right"]]
        self.q16 = np.stack(cols, axis=1)                       # (N, 16) joint_q indices
        self.is_gripper = np.array([i in (7, 15) for i in range(16)])
        objs = self.task.objects
        self.obj_body = {n: self._per_world(bl, n) for n in objs}
        self.obj_q = {n: q_start[self._per_world(jl, f"{n}_joint")] for n in objs}
        self.obj_qd = {n: qd_start[self._per_world(jl, f"{n}_joint")] for n in objs}
        self.obj_shapes = {n: np.array([i for i, l in enumerate(sl) if f"/{n}/" in l]).reshape(self.n, -1)
                           for n in objs}
        self.table_shape = self._per_world(sl, "table")
        self.link8 = {s: self._per_world(bl, f"{s}_fr3v2_link8") for s in v1.SIDES}
        # pinch site: fixed offset in its parent body (same XML as the CPU model)
        mm = self.mj_model
        self.pinch_body, self.pinch_local = {}, {}
        for s in v1.SIDES:
            site = mm.site(f"{s}_gripper_pinch")
            self.pinch_body[s] = self._per_world(bl, mm.body(site.bodyid[0]).name)
            self.pinch_local[s] = site.pos.copy()
        self.tcp_local = mm.site("right_tcp").pos.copy()     # in link8 (same mount on both arms)
        self.nu = mm.nu
        self.world_shape_count = self.model.shape_count // self.n

    def _copy_robot_colors(self):
        """add_mjcf ignores <material> colours (robot meshes come out flat grey): copy each MuJoCo
        visual geom's material colour onto the matching Newton shape (same body, same file order)."""
        mm, m = self.mj_model, self.model
        color, body, flags = m.shape_color.numpy(), m.shape_body.numpy(), m.shape_flags.numpy()
        visible = flags & int(newton.ShapeFlags.VISIBLE)
        for b in range(1, mm.nbody):
            name = mm.body(b).name
            if "_fr3v2_" not in name and "_gripper_" not in name:
                continue
            geoms = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == b and mm.geom_group[g] == 2]
            rgba = [mm.mat_rgba[mm.geom_matid[g]] if mm.geom_matid[g] >= 0 else mm.geom_rgba[g] for g in geoms]
            for nb in self._per_world(m.body_label, name):
                shapes = [s for s in np.flatnonzero(body == nb) if visible[s]]
                if rgba and len(shapes) == len(rgba):
                    color[shapes] = np.array(rgba)[:, :3]
        m.shape_color.assign(color)

    def _apply_task_colors(self):
        """Task-provided flat colours for shapes whose texture Newton cannot import (V2Task.newton_shape_colors)."""
        colors = getattr(self.task, "newton_shape_colors", lambda: {})()
        if not colors:
            return
        c = self.model.shape_color.numpy()
        for i, label in enumerate(self.model.shape_label):
            for suffix, rgb in colors.items():
                if label.endswith(suffix):
                    c[i] = rgb
        self.model.shape_color.assign(c)

    # -- episode setup --------------------------------------------------
    def reset(self, cfgs, jitters=None, right_qs=None, arm_qs=None) -> None:
        """Apply one EpisodeConfig (and optional xy layout jitter) per world, put the robot at home (or the acting
        arm at arm_qs[w], for near-start episodes; right_qs is the older name), settle."""
        assert len(cfgs) == self.n
        jitters = jitters if jitters is not None else [(0.0, 0.0)] * self.n
        arm_qs = arm_qs if arm_qs is not None else (right_qs if right_qs is not None else [None] * self.n)
        self.cfgs = list(cfgs)
        self.targets = [c.object for c in cfgs]
        self.sides = [self.task.side(c) for c in cfgs]
        self.center_z = np.array([self.task.rest_z(c.object) for c in cfgs])
        q = self.model.joint_q.numpy().copy()
        qd = np.zeros(self.model.joint_dof_count, np.float32)
        mass, inertia = self.base_mass.copy(), self.base_inertia.copy()
        color = self.base_color.copy()
        mu = self.model.shape_material_mu.numpy().copy()
        home16 = np.concatenate([HOME_Q, [0.0], HOME_Q, [0.0]])
        for w, cfg in enumerate(cfgs):
            p = self.task.episode_params(cfg, jitters[w])
            q[self.q16[w]] = home16
            if arm_qs[w] is not None:
                q[self.q16[w][ARM[self.sides[w]]]] = arm_qs[w]
            for n in self.task.objects:
                pos, wxyz = p["poses"][n]
                q[self.obj_q[n][w]:self.obj_q[n][w] + 7] = [*pos, wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
                b = self.obj_body[n][w]
                mass[b] *= p["mass_scale"]
                inertia[b] *= p["mass_scale"]
                if n in p["rgba"]:
                    color[self.obj_shapes[n][w]] = p["rgba"][n][:3]
                mu[self.obj_shapes[n][w]] = p["friction"]
            color[self.table_shape[w]] = p["rgba"]["table"][:3]
            self.light_gain[w] = p["light_gain"]
        # the gripper's passive joints (couplers, followers) start at 0, like MuJoCo's qpos0
        self.model.body_mass.assign(mass)
        self.model.body_inertia.assign(inertia)
        self.model.body_inv_mass.assign(np.where(mass > 0, 1 / np.maximum(mass, 1e-12), 0))
        self.model.body_inv_inertia.assign(np.linalg.pinv(inertia).astype(np.float32))
        self.model.shape_material_mu.assign(mu)
        self.model.shape_color.assign(color)
        self.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.SHAPE_PROPERTIES)
        for s in self.states:
            s.joint_q.assign(q)
            s.joint_qd.assign(qd)
            newton.eval_fk(self.model, s.joint_q, s.joint_qd, s)
        self.last_action = np.tile(np.concatenate([HOME_Q, [1.0], HOME_Q, [1.0]]).astype(np.float32), (self.n, 1))
        for w, aq in enumerate(arm_qs):
            if aq is not None:
                self.last_action[w, ARM[self.sides[w]]] = aq
        for _ in range(SETTLE_FRAMES):
            self.step(self.last_action)

    # -- stepping -------------------------------------------------------
    def _substeps(self):
        for _ in range(SUBSTEPS):
            self.solver.step(self.states[0], self.states[1], self.control, None, PHYSICS_DT)
            self.states[0], self.states[1] = self.states[1], self.states[0]

    def step(self, actions: np.ndarray) -> np.ndarray:
        """Apply one 30 Hz frame of (N, 16) dataset actions; returns the (N, 16) state."""
        a = np.asarray(actions, np.float32).reshape(self.n, 16)
        self.last_action = a
        ctrl = a.copy()
        ctrl[:, self.is_gripper] = (1.0 - np.clip(a[:, self.is_gripper], 0, 1)) * GRIPPER_CTRL_MAX
        self.control.mujoco.ctrl.assign(ctrl.reshape(-1))
        if not self.use_graph:
            self._substeps()
        else:
            if self.graph is None:
                with wp.ScopedCapture() as cap:
                    self._substeps()
                self.graph = cap.graph
            wp.capture_launch(self.graph)
        return self.state()

    def state(self) -> np.ndarray:
        q = self.states[0].joint_q.numpy()
        s = q[self.q16].astype(np.float32)
        s[:, self.is_gripper] = 1.0 - s[:, self.is_gripper] / GRIPPER_CLOSED_RAD
        return s

    # -- observations ---------------------------------------------------
    def task_obs(self) -> dict:
        """Ground truth per world, for that world's acting arm."""
        st = self.states[0]
        body_q = st.body_q.numpy()           # (B, 7) p, q_xyzw
        body_qd = st.body_qd.numpy()         # (B, 6)
        q = st.joint_q.numpy()
        w_all = np.arange(self.n)
        tgt = np.array([self.obj_body[t][w] for w, t in enumerate(self.targets)])
        pos = body_q[tgt, :3]
        quat_wxyz = body_q[tgt][:, [6, 3, 4, 5]]
        grasp = np.stack([self.task.grasp_point(t, pos[w], quat_wxyz[w]) for w, t in enumerate(self.targets)])
        pb = np.array([self.pinch_body[s][w] for w, s in enumerate(self.sides)])
        pl = np.stack([self.pinch_local[s] for s in self.sides])
        pinch = body_q[pb, :3] + np.einsum("nij,nj->ni", _quat_xyzw_to_mat(body_q[pb, 3:7]), pl)
        l8 = np.array([self.link8[s][w] for w, s in enumerate(self.sides)])
        tcp = body_q[l8, :3] + _quat_xyzw_to_mat(body_q[l8, 3:7]) @ self.tcp_local
        gcol = np.array([GRIP[s] for s in self.sides])
        lin = body_qd[tgt, :3]               # Newton body_qd = (linear at the COM, angular), world frame
        objects = {n: (body_q[self.obj_body[n], :3], body_q[self.obj_body[n]][:, [6, 3, 4, 5]]) for n in self.task.objects}
        return {"target_pos": pos, "target_quat": quat_wxyz, "grasp_pos": grasp,
                "target_speed": np.linalg.norm(lin, axis=-1), "pinch_pos": pinch,
                "driver_q": q[self.q16[w_all, gcol]], "grip_cmd": self.last_action[w_all, gcol], "tcp_pos": tcp,
                "objects": objects}

    def flags(self, obs=None) -> dict:
        o = obs or self.task_obs()
        return instant_flags(o["target_pos"], o["grasp_pos"], o["target_speed"], o["pinch_pos"], o["driver_q"],
                             o["grip_cmd"], self.center_z, TABLE_TOP, self.task.tray_center,
                             tray_half=self.task.tray_half, floor_top=self.task.tray_floor_top)

    # -- cameras --------------------------------------------------------
    def _setup_cameras(self):
        size = self.image_size
        self.camera = SensorTiledCamera(self.model)
        self.camera.default_render_config.enable_shadows = True
        self.camera.utils.create_default_light(enable_shadows=True,
                                               direction=wp.vec3f(*(np.array([0.3, 0.2, -1.0]) / np.linalg.norm([0.3, 0.2, -1.0]))))
        fovs = [math.radians(v1.CAM_FOVY["external_camera"])] + [math.radians(v1.CAM_FOVY["wrist"])] * 2
        self.rays = self.camera.utils.compute_camera_rays_pinhole(size, size, camera_fovs=fovs)
        self.hdr = self.camera.utils.create_hdr_color_image_output(size, size, len(CAMERAS))
        self.rgb = None
        head = v1.REF["cameras"]["external_camera"]
        self.head_pos = np.array(head["pos"]) + [0, 0, v1.MOUNT_Z]
        self.head_rot = _mat_from_xyaxes(head["xyaxes"])
        wrist = {s: v1.REF["cameras"][f"{s}_wrist_camera"] for s in v1.SIDES}
        # wrist cameras are defined in the flange frame; scene.py mounts both with the left entry
        self.wrist_pos = np.array(wrist["left"]["pos"])
        self.wrist_rot = _mat_from_xyaxes(wrist["left"]["xyaxes"])

    def camera_transforms(self) -> wp.array:
        body_q = self.states[0].body_q.numpy()
        pos = np.zeros((len(CAMERAS), self.n, 3))
        rot = np.zeros((len(CAMERAS), self.n, 3, 3))
        pos[0], rot[0] = self.head_pos, self.head_rot
        for c, side in ((1, "left"), (2, "right")):
            b = self.link8[side]
            R = _quat_xyzw_to_mat(body_q[b, 3:7])
            pos[c] = body_q[b, :3] + R @ self.wrist_pos
            rot[c] = R @ self.wrist_rot
        xf = np.concatenate([pos, _mat_to_quat_xyzw(rot)], axis=-1).astype(np.float32)
        return wp.array(xf, dtype=wp.transformf)

    def images(self) -> dict[str, np.ndarray]:
        """{camera: (N, H, W, 3) uint8} for every world, lit per world."""
        st = self.states[0]
        self.model.bvh_refit_shapes(st)
        self.camera.update(st, self.camera_transforms(), self.rays, hdr_color_image=self.hdr)
        if self.rgb is None:
            self.rgb = wp.empty((self.n * len(CAMERAS), self.image_size, self.image_size, 3), dtype=wp.uint8)
        gain = wp.array(self.light_gain, dtype=wp.vec3f)
        wp.launch(_light_and_encode, dim=self.hdr.shape, inputs=[self.hdr, gain, self.rgb])
        rgb = self.rgb.numpy().reshape(self.n, len(CAMERAS), self.image_size, self.image_size, 3)
        return {cam: rgb[:, c] for c, cam in enumerate(CAMERAS)}
