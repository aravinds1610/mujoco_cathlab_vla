# Copied from the team repo (Rakeshdharan/vla-cathlab-companion@9deeb21, dataengine/variation.py); team code, unchanged.
import itertools
from pathlib import Path
from typing import Literal
import numpy as np
import yaml
from pydantic import BaseModel

TIERS: tuple[str, ...] = (
    "in_distribution",
    "held_out_layout",
    "held_out_obj_color",
    "held_out_table_color",
    "all_held_out",
)


class SceneGeometry(BaseModel):
    table_size: tuple[float, float]
    tray_center: tuple[float, float]
    tray_inner: tuple[float, float]
    tray_floor_z: float
    pick_region_center: tuple[float, float]
    pick_region_size: tuple[float, float]
    min_separation: float
    # A tall upright object may not stand within this distance of the line from the robot base to another object,
    # closer to the base than that object (the forearm would hit it). 0 disables the rule.
    reach_corridor_width: float = 0.0
    tall_object_height: float = 0.10


class ObjectSpec(BaseModel):
    shape: Literal["cylinder", "syringe"]
    radius: float
    length: float
    upright: bool
    mass: float
    grasp: Literal["side", "top"]

    def center_z(self) -> float:
        return self.length / 2 if self.upright else self.radius

    def half_extents(self) -> np.ndarray:
        """Half sizes in the object frame (long axis = z if upright, x if lying)."""
        if self.upright:
            return np.array([self.radius, self.radius, self.length / 2])
        return np.array([self.length / 2, self.radius, self.radius])


class AxisSplit(BaseModel):
    train: list[int]
    held_out: list[int]


class VariationSpec(BaseModel):
    variation_id: str
    scene: SceneGeometry
    objects: dict[str, ObjectSpec]
    n_layouts: int
    layout_seed: int
    obj_colors: list[dict[str, list[float]]]
    table_colors: list[list[float]]
    layout_split: AxisSplit
    obj_color_split: AxisSplit
    table_color_split: AxisSplit
    mass_scale: tuple[float, float]
    friction: tuple[float, float]
    light_intensity: tuple[float, float]
    light_temp_k: tuple[float, float]

    @classmethod
    def load(cls, path: str | Path) -> "VariationSpec":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class EpisodeConfig(BaseModel):
    object: str
    layout_id: int
    obj_color_id: int
    table_color_id: int
    mass_scale: float
    friction: float
    light_intensity: float
    light_temp_k: float
    seed: int
    tier: str


def layouts(spec: VariationSpec) -> list[dict[str, tuple[float, float, float]]]:
    """Deterministic layouts: object -> (x, y, yaw), non-overlapping inside the pick region."""
    cx, cy = spec.scene.pick_region_center
    sx, sy = spec.scene.pick_region_size
    out = []
    for i in range(spec.n_layouts):
        rng = np.random.default_rng(spec.layout_seed + i)
        while True:
            placed: dict[str, tuple[float, float, float]] = {}
            for name, obj in spec.objects.items():
                margin = obj.length / 2 if not obj.upright else obj.radius
                x = rng.uniform(cx - sx / 2 + margin, cx + sx / 2 - margin)
                y = rng.uniform(cy - sy / 2 + margin, cy + sy / 2 - margin)
                placed[name] = (float(x), float(y), float(rng.uniform(-np.pi, np.pi)))
            pts = np.array([p[:2] for p in placed.values()])
            dists = np.linalg.norm(pts[:, None] - pts[None], axis=-1)[np.triu_indices(len(pts), 1)]
            if dists.min() >= spec.scene.min_separation and not _blocks_reach(spec, placed):
                out.append(placed)
                break
    return out


def _blocks_reach(spec: VariationSpec, placed: dict[str, tuple[float, float, float]]) -> bool:
    width = spec.scene.reach_corridor_width
    if width <= 0:
        return False
    for tall, obj in spec.objects.items():
        if not (obj.upright and obj.length >= spec.scene.tall_object_height):
            continue
        b = np.array(placed[tall][:2])
        for other, pose in placed.items():
            if other == tall:
                continue
            p = np.array(pose[:2])
            t = b @ p / (p @ p)
            lateral = abs(p[0] * b[1] - p[1] * b[0]) / np.linalg.norm(p)
            if 0 < t < 1 and lateral < width:
                return True
    return False


def object_poses(spec: VariationSpec, cfg: EpisodeConfig) -> dict[str, np.ndarray]:
    """Initial object centre poses [x, y, z, qw, qx, qy, qz] (yaw about z) for a config."""
    layout = layouts(spec)[cfg.layout_id]
    poses = {}
    for name, (x, y, yaw) in layout.items():
        z = spec.objects[name].center_z()
        poses[name] = np.array([x, y, z, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return poses


def _continuous(spec: VariationSpec, rng: np.random.Generator) -> dict[str, float]:
    return {
        "mass_scale": float(rng.uniform(*spec.mass_scale)),
        "friction": float(rng.uniform(*spec.friction)),
        "light_intensity": float(rng.uniform(*spec.light_intensity)),
        "light_temp_k": float(rng.uniform(*spec.light_temp_k)),
    }


def train_configs(spec: VariationSpec, seed: int = 0) -> list[EpisodeConfig]:
    """Full training grid: objects × train layouts × train object colours × train table colours."""
    grid = itertools.product(
        spec.objects, spec.layout_split.train, spec.obj_color_split.train, spec.table_color_split.train
    )
    out = []
    for i, (obj, layout, oc, tc) in enumerate(grid):
        rng = np.random.default_rng(seed * 1_000_003 + i)
        out.append(EpisodeConfig(object=obj, layout_id=layout, obj_color_id=oc, table_color_id=tc,
                                 seed=seed * 1_000_003 + i, tier="train", **_continuous(spec, rng)))
    return out


def eval_configs(spec: VariationSpec, tier: str, seeds: list[int]) -> list[EpisodeConfig]:
    """Eval episodes for one tier: every object × every seed. Seeds should not overlap training seeds."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    held_layout = tier in ("held_out_layout", "all_held_out")
    held_oc = tier in ("held_out_obj_color", "all_held_out")
    held_tc = tier in ("held_out_table_color", "all_held_out")
    lay = spec.layout_split.held_out if held_layout else spec.layout_split.train
    oc = spec.obj_color_split.held_out if held_oc else spec.obj_color_split.train
    tc = spec.table_color_split.held_out if held_tc else spec.table_color_split.train
    out = []
    for obj in spec.objects:
        for s in seeds:
            rng = np.random.default_rng(s)
            out.append(EpisodeConfig(object=obj, layout_id=lay[s % len(lay)], obj_color_id=oc[s % len(oc)],
                                     table_color_id=tc[s % len(tc)], seed=s, tier=tier, **_continuous(spec, rng)))
    return out


def generation_order(spec: VariationSpec, cfgs: list[EpisodeConfig], seed: int = 0) -> list[EpisodeConfig]:
    """Order configs for generation: objects interleaved (bottle, vial, syringe, ...), each object's configs in a fixed
    seeded permutation. Any window or shard of this list is object-balanced and mixes layouts/colours, and every
    simulator that uses it generates the same (paired) episodes."""
    rng = np.random.default_rng(seed)
    per_object = [[c for c in cfgs if c.object == o] for o in spec.objects]
    per_object = [[group[i] for i in rng.permutation(len(group))] for group in per_object]
    return [group[i] for i in range(max(map(len, per_object))) for group in per_object if i < len(group)]
