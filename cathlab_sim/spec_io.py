"""Load SceneSpec YAML and per-object definitions (paths relative to the repo root)."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from cathlab_sim.contracts import SceneSpec

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO / "configs/scenes/cathlab_minimal.yaml"


class ObjectDef(BaseModel):
    """A primitive task object. AR4 will add mesh objects (MJCF files) alongside."""
    type: Literal["capsule", "box"]
    size: list[float]                 # capsule: [radius, half-length]; box: half-extents
    mass: float
    friction: list[float]
    rgba: list[float]

    @property
    def half_height(self) -> float:
        """Resting half-height when lying flat (capsules lie on their side)."""
        return self.size[0] if self.type == "capsule" else self.size[2]

    @property
    def footprint(self) -> tuple[float, float]:
        """Half-extents (along, across) of the flat-lying footprint in the object's yaw frame."""
        if self.type == "capsule":
            return self.size[1] + self.size[0], self.size[0]
        return self.size[0], self.size[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def load_scene_spec(path: str | Path = DEFAULT_SCENE) -> SceneSpec:
    return SceneSpec.model_validate(yaml.safe_load(resolve(path).read_text()))


def load_object_defs(scene: SceneSpec) -> dict[str, ObjectDef]:
    return {name: ObjectDef.model_validate(yaml.safe_load(resolve(p).read_text()))
            for name, p in scene.objects.items()}
