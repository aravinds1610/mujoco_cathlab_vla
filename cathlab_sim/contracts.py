"""Team interfaces, copied verbatim from the handoff (aravind-handoff.md §4).

Do not change field names or signatures: the team pipeline imports these shapes.
Only GapSpec is promoted from a field list to a pydantic model so it can be loaded from JSON.
"""
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

Stage = Literal["reach", "grasp", "lift", "transport", "place"]
STAGES = ("reach", "grasp", "lift", "transport", "place")
DataSource = Literal["provided", "domain_rand", "mimicgen", "scripted", "teleop", "cosmos"]


class TaskSpec(BaseModel):
    task_id: str
    instruction_templates: list[str]
    objects: list[str]
    success_predicate: str
    stage_predicates: dict[Stage, str]

    def instructions(self, obj: str) -> list[str]:
        return [t.format(object=obj) for t in self.instruction_templates]


class SceneSpec(BaseModel):
    scene_id: str
    backend: str                      # we will add "mujoco" for you
    asset: str                        # path to scene file (MJCF for you)
    objects: dict[str, str]           # object name -> asset path
    cameras: list[str]
    randomization: dict[str, list[float]] = Field(default_factory=dict)


class DatasetManifest(BaseModel):     # saved as manifest.yaml next to every dataset
    dataset_id: str                   # e.g. "cathlab/handover_mujoco_scripted"
    version: str                      # "v1", "v2", ... never overwrite a version
    source: DataSource                # "scripted" for your expert
    task_id: str
    scene_id: str
    n_episodes: int
    created_by: str
    parent_versions: list[str] = Field(default_factory=list)
    quality_score: float | None = None
    notes: str = ""


class StepInfo(BaseModel):
    stage_flags: dict[str, bool]      # one bool per STAGES entry, from sim state
    success: bool                     # all stages True
    collision: bool = False


class SimBackend(ABC):
    @abstractmethod
    def reset(self, scene: SceneSpec, seed: int, target_object: str) -> dict[str, np.ndarray]: ...
    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], StepInfo]: ...
    @abstractmethod
    def render(self) -> np.ndarray: ...  # (H, W, 3) uint8
    def close(self) -> None: pass


class PolicyMeta(BaseModel):
    name: str; params_m: float; chunk_size: int; control_hz: float; action_dim: int


class PolicyAdapter(ABC):              # wrap your scripted expert in this so we can evaluate it
    @abstractmethod
    def load(self, ckpt: str) -> None: ...
    @abstractmethod
    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray: ...  # (chunk, 16)
    @abstractmethod
    def meta(self) -> PolicyMeta: ...


class GapSpec(BaseModel):              # what the loop sends to a data generator
    failed_stage: str                  # e.g. "grasp"
    objects: list[str]                 # e.g. ["sheath"]
    scene_variants: list[str]          # e.g. ["heavy_clutter"]
    n_failures: int
    suggested_generator: str           # e.g. "scripted"
