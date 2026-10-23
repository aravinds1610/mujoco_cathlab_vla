"""v2 task objects: realistic surgical instruments (NVIDIA Isaac for Healthcare / Lightwheel, CC BY-NC 4.0).

`prepare()` bakes each converted asset (usd_assets.py + collision_hulls.py output) lying flat: long axis along body
x, thinnest dimension vertical, xy-centred, bottom at z = 0, and picks a grasp point from the geometry. The result is
external/i4h_mujoco/<name>/laid/{part*.obj, hull*.obj, spec.json}, used by both CPU MuJoCo and Newton.

    python -m cathlab_sim.newton.instruments          # prepare all, print the grasp specs
"""
import json
from pathlib import Path

import numpy as np

from cathlab_sim.spec_io import REPO

ASSETS = REPO / "external/i4h_mujoco"
# name -> mass (kg), estimated for real instruments (not measured)
INSTRUMENTS = {"scissors": 0.060, "tweezers": 0.025, "trocar": 0.090, "puncture_device": 0.050}
TRAY = "plate"
GRIPPER_OPEN = 0.085  # Robotiq 2F-85 stroke


def _lay_rotation(verts: np.ndarray) -> np.ndarray:
    """3x3 rotation: thinnest extent -> z, longest -> x."""
    ext = verts.max(0) - verts.min(0)
    order = np.argsort(ext)[::-1]  # longest, middle, thinnest
    R = np.zeros((3, 3))
    for new, old in enumerate(order):
        R[new, old] = 1.0
    if np.linalg.det(R) < 0:
        R[1] *= -1
    return R


def _grasp_from_geometry(verts: np.ndarray) -> dict:
    """Grasp slice across the long (x) axis: narrow enough for the gripper, as central as possible."""
    xs = verts[:, 0]
    lo, hi = xs.min(), xs.max()
    best = None
    for t in np.linspace(0.25, 0.75, 21):
        x0 = lo + t * (hi - lo)
        sl = verts[np.abs(xs - x0) < 0.004]
        if len(sl) < 8:
            continue
        width = sl[:, 1].max() - sl[:, 1].min()
        if not 0.006 <= width <= 0.045:
            continue
        score = abs(t - 0.5) + max(0.0, width - 0.030) * 10  # central first, then narrower than 30 mm
        if best is None or score < best[0]:
            best = (score, x0, (sl[:, 1].max() + sl[:, 1].min()) / 2, (sl[:, 2].max() + sl[:, 2].min()) / 2,
                    width, sl[:, 2].max())
    if best is None:
        raise RuntimeError("no graspable slice")
    _, x0, y0, z0, width, top = best
    return {"point": [float(x0), float(y0), float(z0)], "width": float(width), "slice_top": float(top)}


def prepare(name: str) -> dict:
    """Bake one asset lying flat; return its spec (also written to laid/spec.json)."""
    import trimesh
    src = ASSETS / name
    man = json.loads((src / "manifest.json").read_text())
    vis = [trimesh.load(src / p["obj"], force="mesh", process=False) for p in man["parts"]]
    hulls = [trimesh.load(src / h, force="mesh", process=False) for h in man.get("collision", [])]
    allv = np.concatenate([m.vertices for m in vis])
    R = _lay_rotation(allv)
    rv = allv @ R.T
    shift = -np.array([(rv[:, 0].max() + rv[:, 0].min()) / 2, (rv[:, 1].max() + rv[:, 1].min()) / 2, rv[:, 2].min()])
    out = src / "laid"
    out.mkdir(exist_ok=True)
    parts, hull_files = [], []
    for i, (m, p) in enumerate(zip(vis, man["parts"])):
        m = m.copy()
        m.vertices = m.vertices @ R.T + shift
        m.export(out / p["obj"], include_texture=False)
        if p["texture"]:
            (out / p["texture"]).write_bytes((src / p["texture"]).read_bytes())
        parts.append({**p})
    for i, h in enumerate(hulls):
        h = h.copy()
        h.vertices = h.vertices @ R.T + shift
        f = f"hull{i:02d}.obj"
        h.export(out / f)
        hull_files.append({"obj": f, "volume": float(abs(h.volume))})
    v = allv @ R.T + shift
    spec = {"name": name, "size_m": (v.max(0) - v.min(0)).round(4).tolist(), "parts": parts, "hulls": hull_files,
            "mass": INSTRUMENTS.get(name, 0.5), "grasp": _grasp_from_geometry(v) if name in INSTRUMENTS else None}
    (out / "spec.json").write_text(json.dumps(spec, indent=2))
    return spec


def load_spec(name: str) -> dict:
    return json.loads((ASSETS / name / "laid/spec.json").read_text())


if __name__ == "__main__":
    for n in [*INSTRUMENTS, TRAY]:
        s = prepare(n)
        g = s["grasp"]
        print(f"{n:16s} size {np.round(np.array(s['size_m']) * 1000).astype(int).tolist()} mm, {len(s['hulls'])} hulls"
              + (f", grasp at x {g['point'][0]*1000:+.0f} mm, width {g['width']*1000:.0f} mm, "
                 f"height {g['point'][2]*1000:.0f} mm" if g else ""))


# ---------------------------------------------------------------------------------------------------------------
# v2 task: both arms, instruments on either side of a central tray
# ---------------------------------------------------------------------------------------------------------------
import math  # noqa: E402

import mujoco  # noqa: E402

from cathlab_sim.newton.task_scene import TABLE_TOP, build_base_spec, kelvin_to_rgb  # noqa: E402

TRAY_CENTER = (0.615, 0.0)        # tray (Lightwheel plate, 427 x 303 mm) long side along x; both arms reach it
TRAY_INNER_HALF = (0.189, 0.1335)  # inner floor, measured from the mesh
TRAY_FLOOR = 0.0026                # inner floor top above the table (m)
TRAY_RIM_TOP = 0.022
PLACE_Y = 0.065                    # each arm places on its own half of the tray
REGION_X = (0.38, 0.70)            # pick area per side (|y| range), reachable by that side's arm
REGION_Y = (0.24, 0.56)
CLEARANCE = 0.04                   # gap between object footprints (open gripper fingers)
N_LAYOUTS, LAYOUT_SEED = 50, 2024
TRAIN_LAYOUTS, HELD_OUT_LAYOUTS = list(range(45)), list(range(45, 50))
TABLE_COLORS = [[0.80, 0.80, 0.80, 1.0], [0.55, 0.40, 0.25, 1.0], [0.30, 0.45, 0.60, 1.0], [0.90, 0.90, 0.85, 1.0],
                [0.25, 0.30, 0.25, 1.0]]  # same five as the v1 grid
TRAIN_TABLES, HELD_OUT_TABLES = [0, 1, 2, 3], [4]
RANGES = {"mass_scale": (0.7, 1.3), "friction": (0.5, 1.0), "light_intensity": (0.7, 1.3), "light_temp_k": (4000.0, 7500.0)}
V2_TIERS = ("in_distribution", "held_out_layout", "held_out_table_color", "all_held_out")


def _add_textured_parts(spec, body, name: str, d: Path, parts: list):
    for i, part in enumerate(parts):
        mesh = f"{name}_vis{i}"
        spec.add_mesh(name=mesh, file=str(d / part["obj"]), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        material = ""
        if part["texture"]:
            spec.add_texture(name=f"tex_{mesh}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(d / part["texture"]))
            m = spec.add_material(name=f"mat_{mesh}")
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tex_{mesh}"
            material = f"mat_{mesh}"
        body.add_geom(name=f"{name}_visual{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, material=material,
                      contype=0, conaffinity=0, group=2, mass=0, rgba=[1, 1, 1, 1] if material else [*part["rgb"], 1])


def _add_tray(spec):
    s = load_spec(TRAY)
    d = ASSETS / TRAY / "laid"
    body = spec.worldbody.add_body(name="handover_tray", pos=[*TRAY_CENTER, TABLE_TOP])
    _add_textured_parts(spec, body, "handover_tray", d, s["parts"])
    hx, hy = s["size_m"][0] / 2, s["size_m"][1] / 2
    ix, iy = TRAY_INNER_HALF
    hide = dict(rgba=[0, 0, 0, 0], group=3, friction=[0.9, 0.005, 0.0001])
    body.add_geom(name="handover_tray_hull_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, TRAY_FLOOR / 2],
                  pos=[0, 0, TRAY_FLOOR / 2], **hide)
    for k, sx in enumerate((1, -1)):
        body.add_geom(name=f"handover_tray_hull_wx{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[(hx - ix) / 2, hy, TRAY_RIM_TOP / 2], pos=[sx * (hx + ix) / 2, 0, TRAY_RIM_TOP / 2], **hide)
        body.add_geom(name=f"handover_tray_hull_wy{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[ix, (hy - iy) / 2, TRAY_RIM_TOP / 2], pos=[0, sx * (hy + iy) / 2, TRAY_RIM_TOP / 2], **hide)


def _add_instrument(spec, name: str, park_xy):
    s = load_spec(name)
    d = ASSETS / name / "laid"
    body = spec.worldbody.add_body(name=name, pos=[park_xy[0], park_xy[1], 0.002])
    body.add_freejoint(name=f"{name}_joint")
    _add_textured_parts(spec, body, name, d, s["parts"])
    vol = sum(h["volume"] for h in s["hulls"])
    for i, h in enumerate(s["hulls"]):
        mesh = f"{name}_hullmesh{i}"
        spec.add_mesh(name=mesh, file=str(d / h["obj"]))
        body.add_geom(name=f"{name}_hull{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, rgba=[0, 0, 0, 0],
                      group=3, mass=s["mass"] * h["volume"] / vol, friction=[0.9, 0.005, 0.0001])


def _footprint(name: str):
    sx, sy, _ = load_spec(name)["size_m"]
    return sx / 2 + CLEARANCE / 2, sy / 2 + CLEARANCE / 2


def v2_layouts() -> list[dict[str, tuple[float, float, float]]]:
    """Seeded layouts: {object: (x, y, yaw)}, two instruments per side, footprints >= CLEARANCE apart."""
    from cathlab_sim.layout import _corners, _overlap
    out = []
    for i in range(N_LAYOUTS):
        rng = np.random.default_rng(LAYOUT_SEED + i)
        while True:
            order = list(rng.permutation(list(INSTRUMENTS)))
            placed, rects, ok = {}, [], True
            for k, name in enumerate(order):
                side = 1 if k < 2 else -1
                half = _footprint(name)
                for _ in range(300):
                    x, y = rng.uniform(*REGION_X), side * rng.uniform(*REGION_Y)
                    yaw = rng.uniform(-math.pi, math.pi)
                    r = _corners(x, y, yaw, half)
                    inside = r[:, 0].min() >= REGION_X[0] - 0.05 and r[:, 0].max() <= REGION_X[1] + 0.08 and \
                        np.abs(r[:, 1]).min() >= REGION_Y[0] - 0.06 and np.abs(r[:, 1]).max() <= REGION_Y[1] + 0.06
                    if inside and not any(_overlap(r, o, 0.0) for o in rects):
                        placed[str(name)] = (float(x), float(y), float(yaw))
                        rects.append(r)
                        break
                else:
                    ok = False
                    break
            if ok:
                out.append(placed)
                break
    return out


class V2Task:
    """v2: surgical instruments (meshes), the arm on the object's side picks, central tray."""
    name = "v2"
    objects = tuple(INSTRUMENTS)
    tray_center = TRAY_CENTER
    tray_half = (TRAY_INNER_HALF[0] - 0.01, TRAY_INNER_HALF[1] - 0.01)
    tray_floor_top = TRAY_FLOOR

    def __init__(self):
        self.layouts = v2_layouts()
        self.grasp_local = {n: np.array(load_spec(n)["grasp"]["point"]) for n in INSTRUMENTS}

    def build_spec(self) -> mujoco.MjSpec:
        spec = build_base_spec(TABLE_COLORS[0])
        spec.modelname = "fr3_duo_cathlab_instruments"
        # The Robotiq 2F-85 model's own contact settings (MjSpec.attach keeps the parent's defaults instead):
        # elliptic friction cone + impratio 10. Without them light instruments creep ~0.14 mm/s on CPU MuJoCo.
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        _add_tray(spec)
        for i, name in enumerate(INSTRUMENTS):
            _add_instrument(spec, name, (-1.5, -0.6 + 0.4 * i))
        return spec

    def episode_params(self, cfg, jitter=(0.0, 0.0)) -> dict:
        poses = {}
        for name, (x, y, yaw) in self.layouts[cfg.layout_id].items():
            poses[name] = (np.array([x + jitter[0], y + jitter[1], TABLE_TOP + 0.001]),
                           np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
        return {"poses": poses, "rgba": {"table": TABLE_COLORS[cfg.table_color_id]}, "mass_scale": cfg.mass_scale,
                "friction": cfg.friction, "light_gain": cfg.light_intensity * kelvin_to_rgb(cfg.light_temp_k)}

    def newton_shape_colors(self) -> dict[str, list[float]]:
        """Newton's MJCF import drops materials/textures: give textured visual parts their texture's mean colour.
        Returns {shape-label suffix: rgb}."""
        from PIL import Image
        out = {}
        for name in [*INSTRUMENTS, TRAY]:
            d = ASSETS / name / "laid"
            prefix = "handover_tray" if name == TRAY else name
            for i, part in enumerate(load_spec(name)["parts"]):
                if part["texture"]:
                    img = np.asarray(Image.open(d / part["texture"]).convert("RGB"), float) / 255.0
                    out[f"/{prefix}_visual{i}"] = img.reshape(-1, 3).mean(0).tolist()
        return out

    def grasp_point(self, name, pos, quat_wxyz) -> np.ndarray:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat_wxyz, float))
        return np.asarray(pos, float) + R.reshape(3, 3) @ self.grasp_local[name]

    def rest_z(self, name: str) -> float:
        return 0.0  # body origin = bottom of the laid mesh

    def side(self, cfg) -> str:
        return "left" if self.layouts[cfg.layout_id][cfg.object][1] > 0 else "right"

    def instruction(self, cfg) -> str:
        from cathlab_sim.newton.task_scene import INSTRUCTIONS
        return INSTRUCTIONS[cfg.seed % 3].format(object=cfg.object.replace("_", " "))

    def _cfg(self, obj, layout, table, seed, tier):
        from cathlab_sim.newton.variation import EpisodeConfig
        rng = np.random.default_rng(seed)
        return EpisodeConfig(object=obj, layout_id=layout, obj_color_id=0, table_color_id=table, seed=seed, tier=tier,
                             **{k: float(rng.uniform(*r)) for k, r in RANGES.items()})

    def configs(self, seed: int = 0) -> list:
        """Train grid: 45 layouts x 4 instruments x 2 table colours = 360, objects interleaved."""
        out = []
        for li in TRAIN_LAYOUTS:
            for oi, obj in enumerate(INSTRUMENTS):
                for k in range(2):
                    t = TRAIN_TABLES[(li + oi + 2 * k + seed) % len(TRAIN_TABLES)]
                    out.append(self._cfg(obj, li, t, seed * 1_000_003 + len(out), "train"))
        rng = np.random.default_rng(seed)
        per = [[c for c in out if c.object == o] for o in INSTRUMENTS]
        per = [[g[i] for i in rng.permutation(len(g))] for g in per]
        return [g[i] for i in range(max(map(len, per))) for g in per if i < len(g)]

    def eval_configs(self, tier: str, seeds: list[int]) -> list:
        lay = HELD_OUT_LAYOUTS if tier in ("held_out_layout", "all_held_out") else TRAIN_LAYOUTS
        tab = HELD_OUT_TABLES if tier in ("held_out_table_color", "all_held_out") else TRAIN_TABLES
        return [self._cfg(obj, lay[s % len(lay)], tab[s % len(tab)], s, tier) for obj in INSTRUMENTS for s in seeds]

    def grasp_plan(self, target: str, obs: dict) -> dict:
        from cathlab_sim.expert import PAD_CENTRE_CLOSED, TABLE_CLEARANCE, TIP_CLOSED
        g, q_obj = obs["grasp_pos"][0], obs["target_quat"][0]
        grasp_z = max(g[2] + PAD_CENTRE_CLOSED, TABLE_TOP + TIP_CLOSED + TABLE_CLEARANCE)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q_obj)
        axis = R.reshape(3, 3)[:, 0]
        a = math.atan2(axis[1], axis[0])
        place_y = PLACE_Y if g[1] > 0 else -PLACE_Y
        return {"grasp": g, "grasp_z": grasp_z, "yaws": [a, a + math.pi],
                "place_xy": (TRAY_CENTER[0], place_y), "place_yaws": [0.0, math.pi], "rest_z": 0.0}
