"""MujocoBackend: the team's SimBackend contract on top of the cathlab scene.

Observation keys follow the LeRobot dataset schema so episodes can be written as-is:
  observation.state                      float32[16]
  observation.images.<camera>            uint8[224, 224, 3]   (external, left_wrist, right_wrist)
plus privileged.* keys (ground truth) for the scripted expert only, never for policies.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must be set before mujoco loads
import mujoco
import numpy as np

from cathlab_sim.contracts import STAGES, SceneSpec, SimBackend, StepInfo
from cathlab_sim.layout import apply_layout, sample_layout
from cathlab_sim.robot_io import RobotIO
from cathlab_sim.scene import CAMERAS, IMAGE_SIZE, SIDES, build_spec, set_home
from cathlab_sim.spec_io import load_object_defs
from cathlab_sim.stages import StageEvaluator, latch

SETTLE_S = 0.5  # let objects settle after placement before the episode starts


class MujocoBackend(SimBackend):
    def __init__(self, render_images: bool = True, image_size: int = IMAGE_SIZE):
        self.render_images = render_images
        self.image_size = image_size
        self.scene: SceneSpec | None = None
        self.renderer = None

    # -- contract -------------------------------------------------------
    def reset(self, scene: SceneSpec, seed: int, target_object: str) -> dict[str, np.ndarray]:
        if self.scene is None or scene != self.scene:
            self._load(scene)
        self.seed, self.target = seed, target_object
        mujoco.mj_resetData(self.model, self.data)
        set_home(self.model, self.data)
        self.layout = sample_layout(np.random.default_rng(seed), scene.randomization, self.objects, target_object)
        apply_layout(self.model, self.data, self.layout, self.objects)
        mujoco.mj_step(self.model, self.data, nstep=round(SETTLE_S / self.model.opt.timestep))
        self.stages.set_target(target_object, self.objects[target_object].half_height)
        self.flags = {s: False for s in STAGES}
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], StepInfo]:
        self.io.step(self.data, action)
        self.flags = latch(self.flags, self.stages.instant_flags(self.data))
        info = StepInfo(stage_flags=dict(self.flags), success=all(self.flags.values()),
                        collision=self.stages.collision(self.data))
        return self._obs(), info

    def render(self) -> np.ndarray:
        return self._image("external_camera")

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # -- helpers --------------------------------------------------------
    def _load(self, scene: SceneSpec):
        self.close()
        self.scene = scene
        self.objects = load_object_defs(scene)
        self.model = build_spec(scene).compile()
        self.data = mujoco.MjData(self.model)
        self.io = RobotIO(self.model)
        self.stages = StageEvaluator(self.model)
        if self.render_images:
            self.renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)

    def _image(self, camera: str) -> np.ndarray:
        if self.renderer is None:
            return np.zeros((self.image_size, self.image_size, 3), np.uint8)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def _obs(self) -> dict[str, np.ndarray]:
        obs = {"observation.state": self.io.get_state(self.data)}
        for cam in CAMERAS:
            obs[f"observation.images.{cam}"] = self._image(cam)
        t = self.model.body(self.target).id
        obs["privileged.target_pos"] = self.data.xpos[t].copy()
        obs["privileged.target_quat"] = self.data.xquat[t].copy()
        for s in SIDES:
            obs[f"privileged.{s}_tcp_pos"] = self.data.site(f"{s}_tcp").xpos.copy()
        return obs
