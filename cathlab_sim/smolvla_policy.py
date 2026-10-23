"""A fine-tuned SmolVLA checkpoint wrapped in the team's PolicyAdapter contract.

    policy = SmolVLAPolicyAdapter("outputs/smolvla_v1_3k/checkpoints/last/pretrained_model")
    chunk = policy.predict(obs, "Hand me the syringe")   # (n_exec, 16) joint targets + grippers

The checkpoint carries its own processors (camera renaming, tokenizer, normalization stats),
so this only converts MujocoBackend observations into the dataset's format: images HWC uint8
-> CHW float in [0, 1], state float32. SmolVLA predicts 50 actions; we execute the first
n_exec and then re-plan from the new observation.
"""
import time
from pathlib import Path

import numpy as np
import torch

from cathlab_sim.contracts import PolicyAdapter, PolicyMeta
from cathlab_sim.robot_io import CONTROL_HZ
from cathlab_sim.scene import CAMERAS


class SmolVLAPolicyAdapter(PolicyAdapter):
    """blend > 0 cross-fades the first `blend` actions of each new chunk from the previous chunk's
    unexecuted continuation (removes the jump at chunk switches). noise_seed fixes the flow-matching
    start noise, so consecutive chunks are sampled consistently. Both off by default (v1 behaviour)."""

    def __init__(self, ckpt: str | Path | None = None, n_exec: int = 25, device: str = "cuda",
                 blend: int = 0, noise_seed: int | None = None):
        self.n_exec, self.device = n_exec, device
        self.blend, self.noise_seed = blend, noise_seed
        self.policy, self._prev, self._noise = None, None, None
        if ckpt is not None:
            self.load(str(ckpt))

    def load(self, ckpt: str) -> None:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.ckpt = ckpt
        self.policy = SmolVLAPolicy.from_pretrained(ckpt).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": self.device}})
        self.params_m = sum(p.numel() for p in self.policy.parameters()) / 1e6
        if self.noise_seed is not None:
            cfg = self.policy.config
            g = torch.Generator().manual_seed(self.noise_seed)
            self._noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=g).to(self.device)

    def reset(self) -> None:
        """Call at the start of every episode (after backend.reset)."""
        self.policy.reset()
        self._prev = None

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        frame = {"observation.state": torch.from_numpy(np.asarray(obs["observation.state"], np.float32)),
                 "task": instruction}
        for cam in CAMERAS:
            img = obs[f"observation.images.{cam}"]
            frame[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(frame), noise=self._noise)  # (1, 50, 16), normalized
            chunk = self.post(chunk)                                   # back to radians / gripper 0..1
        new = chunk[0].cpu().numpy().astype(np.float32)
        out = new[:self.n_exec].copy()
        if self.blend and self._prev is not None:
            b = min(self.blend, len(self._prev), len(out))
            w = (np.arange(1, b + 1) / (b + 1))[:, None]
            out[:b] = (1 - w) * self._prev[:b] + w * new[:b]
        self._prev = new[self.n_exec:]
        return out

    def meta(self) -> PolicyMeta:
        return PolicyMeta(name=f"smolvla:{Path(self.ckpt).parent.parent.parent.name}/{Path(self.ckpt).parent.name}",
                          params_m=round(self.params_m, 1), chunk_size=self.n_exec,
                          control_hz=float(CONTROL_HZ), action_dim=16)
