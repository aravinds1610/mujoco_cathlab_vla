"""The scripted expert wrapped in the team's PolicyAdapter contract, so it can be evaluated
like any learned policy (it is our "own expert policy" for the definition of done).

It uses privileged ground truth (object pose, via the backend) and finds the target by
matching an object name in the instruction (ground-truth ID; the handoff allows this).
"""
import numpy as np

from cathlab_sim.contracts import PolicyAdapter, PolicyMeta
from cathlab_sim.expert import ScriptedExpert
from cathlab_sim.robot_io import CONTROL_HZ


class ScriptedExpertPolicy(PolicyAdapter):
    def __init__(self, backend, chunk_size: int = 10):
        self.backend, self.chunk_size = backend, chunk_size
        self.reset()

    def reset(self) -> None:
        """Call at the start of every episode (after backend.reset)."""
        self.plan, self.cursor = None, 0

    def load(self, ckpt: str) -> None:
        pass  # nothing to load: the expert is a program, not a checkpoint

    def _target_from(self, instruction: str) -> str:
        hits = [o for o in self.backend.objects if o in instruction.lower()]
        if len(hits) != 1:
            raise ValueError(f"instruction {instruction!r} must name exactly one of {sorted(self.backend.objects)}")
        return hits[0]

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        if self.plan is None:
            target = self._target_from(instruction)
            if target != self.backend.target:
                raise ValueError(f"instruction names {target!r} but the episode target is {self.backend.target!r}")
            self.plan = ScriptedExpert(self.backend.model).plan(self.backend.data, obs, self.backend.objects[target])
        chunk = self.plan[self.cursor:self.cursor + self.chunk_size]
        if len(chunk) < self.chunk_size:  # past the end: hold the last action
            chunk = np.concatenate([chunk, np.repeat(self.plan[-1:], self.chunk_size - len(chunk), axis=0)])
        self.cursor += self.chunk_size
        return chunk

    @property
    def done(self) -> bool:
        return self.plan is not None and self.cursor >= len(self.plan)

    def meta(self) -> PolicyMeta:
        return PolicyMeta(name="scripted_ik_expert_v1", params_m=0.0, chunk_size=self.chunk_size,
                          control_hz=float(CONTROL_HZ), action_dim=16)
