"""Watch a trained policy (or the expert) do the right-arm task live.

Physics and the policy's camera images come from Newton (what the model was trained on); every frame
is mirrored into a MuJoCo viewer window so you can watch. Run in .venv-eval (Newton + LeRobot):

    python -m cathlab_sim.newton.watch --ckpt outputs/smolvla_newton_v1_16k/checkpoints/last/pretrained_model
    python -m cathlab_sim.newton.watch --expert                     # scripted expert, for comparison
    python -m cathlab_sim.newton.watch --ckpt ... --say vial        # swap test: ask for another object

Scenes come from the team's held-out tier (unseen layouts, palette and table colour). Options:
--episodes 3 --tier all_held_out --speed 1.0 --no-viewer. Close the viewer window to stop.
"""
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import numpy as np

from cathlab_sim.newton.runner import INSTRUCTIONS
from cathlab_sim.newton.task_stages import StageTracker
from cathlab_sim.robot_io import CONTROL_HZ
from cathlab_sim.scene import CAMERAS

MAX_FRAMES = 600  # 20 s; the expert needs ~455


class Mirror:
    """Copies Newton world 0's joint coordinates into a MuJoCo MjData for display."""

    def __init__(self, sim):
        m = sim.mj_model
        self.model, self.data = m, mujoco.MjData(m)
        q_start = sim.model.joint_q_start.numpy()
        self.pairs = []  # (mujoco qpos adr, newton q index, is_free)
        for j in range(m.njnt):
            name = m.joint(j).name
            idx = [i for i, l in enumerate(sim.model.joint_label) if l.endswith("/" + name)]
            if idx:
                self.pairs.append((m.jnt_qposadr[j], q_start[idx[0]], m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE))
        self.geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(n).id]
                      for n in ("bottle", "vial", "syringe")}

    def colours(self, params):
        for n, gs in self.geoms.items():
            for g in gs:
                self.model.geom_rgba[g] = params["rgba"][n]
        self.model.geom_rgba[self.model.geom("table").id] = params["rgba"]["table"]

    def sync(self, sim):
        q = sim.states[0].joint_q.numpy()
        for adr, i, free in self.pairs:
            if free:
                self.data.qpos[adr:adr + 3] = q[i:i + 3]
                self.data.qpos[adr + 3:adr + 7] = q[[i + 6, i + 3, i + 4, i + 5]]  # xyzw -> wxyz
            else:
                self.data.qpos[adr] = q[i]
        mujoco.mj_forward(self.model, self.data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="SmolVLA pretrained_model folder")
    ap.add_argument("--expert", action="store_true", help="run the scripted expert instead")
    ap.add_argument("--tier", default="all_held_out")
    ap.add_argument("--seed0", type=int, default=100_000)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--say", default=None, help="name this object in the instruction (swap test)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--blend", type=int, default=10, help="cross-fade frames at chunk switches (0 = off)")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()
    assert args.expert or args.ckpt, "give --ckpt or --expert"

    from cathlab_sim.newton.scene import NewtonTaskSim
    from cathlab_sim.newton.task_scene import episode_params
    from cathlab_sim.newton.variation import eval_configs
    sim = NewtonTaskSim(1)
    # eval_configs lists object-major (all bottle seeds, then vial, then syringe): cycle the objects,
    # each episode on its own seed
    n = args.episodes
    every = eval_configs(sim.variation, args.tier, list(range(args.seed0, args.seed0 + n)))
    cfgs = [every[(k % 3) * n + k] for k in range(n)]
    policy = None
    if not args.expert:
        from cathlab_sim.smolvla_policy import SmolVLAPolicyAdapter
        policy = SmolVLAPolicyAdapter(args.ckpt, blend=args.blend)
    mirror = Mirror(sim)
    viewer = None
    try:
        for ep, cfg in enumerate(cfgs):
            sim.reset([cfg])
            said = args.say or cfg.object
            instruction = INSTRUCTIONS[cfg.seed % 3].format(object=said)
            mirror.colours(episode_params(sim.variation, cfg))
            mirror.sync(sim)
            if viewer is None and not args.no_viewer:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(mirror.model, mirror.data)
                viewer.cam.lookat[:] = [0.50, 0.0, 0.95]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 180, -35
            print(f"episode {ep + 1}: layout {cfg.layout_id}, target {cfg.object}, \"{instruction}\"", flush=True)
            if args.expert:
                plan = _expert_plan(cfg)
            tracker, frames, success, t_policy = StageTracker(1), 0, False, []
            while frames < MAX_FRAMES and not success:
                if args.expert:
                    chunk = plan[frames:frames + 25] if frames < len(plan) else plan[-1:].repeat(25, 0)
                else:
                    imgs = sim.images()
                    obs = {"observation.state": sim.state()[0],
                           **{f"observation.images.{c}": imgs[c][0] for c in CAMERAS}}
                    if frames == 0:
                        policy.reset()
                    t0 = time.perf_counter()
                    chunk = policy.predict(obs, instruction)
                    t_policy.append(time.perf_counter() - t0)
                for a in chunk:
                    t0 = time.perf_counter()
                    sim.step(a[None])
                    frames += 1
                    success = bool(tracker.update(sim.flags())[0])
                    if viewer is not None:
                        if not viewer.is_running():
                            return
                        mirror.sync(sim)
                        viewer.sync()
                        time.sleep(max(0.0, 1.0 / (CONTROL_HZ * args.speed) - (time.perf_counter() - t0)))
                    if success or frames >= MAX_FRAMES:
                        break
            stage = None if success else tracker.failure_stage(0)
            extra = f", policy {1000 * np.mean(t_policy):.0f} ms/call" if t_policy else ""
            print(f"  -> {'SUCCESS' if success else 'failed at ' + str(stage)} after {frames / CONTROL_HZ:.1f} s{extra}",
                  flush=True)
            if viewer is not None:
                time.sleep(1.5)
    finally:
        if viewer is not None:
            viewer.close()


def _expert_plan(cfg):
    from cathlab_sim.newton.runner import _init_worker, plan_one
    _init_worker()
    return plan_one({"cfg": cfg.model_dump()})["actions"]


if __name__ == "__main__":
    main()
