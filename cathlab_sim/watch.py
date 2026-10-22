"""Watch a policy do the handover live in the MuJoCo viewer (real time).

    python -m cathlab_sim.watch                                   # scripted expert
    python -m cathlab_sim.watch --policy smolvla --ckpt outputs/smolvla_v1_3k/checkpoints/last/pretrained_model

Options: --seed 10000 --target syringe --episodes 5 --speed 1.0. Each episode uses the next seed;
use seeds >= 10000 to see layouts the model never trained on. Close the viewer window to stop.
"""
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")  # on-screen window; the backend would default to headless EGL
import mujoco
import mujoco.viewer
import numpy as np

from cathlab_sim.backend import MujocoBackend
from cathlab_sim.contracts import STAGES
from cathlab_sim.evaluate import MAX_EPISODE_S, make_policy
from cathlab_sim.generate import DEFAULT_TASK, load_task
from cathlab_sim.robot_io import CONTROL_HZ
from cathlab_sim.spec_io import load_scene_spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["expert", "smolvla"], default="expert")
    ap.add_argument("--ckpt", default=None, help="SmolVLA checkpoint folder (pretrained_model)")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--target", default=None, help="object to pick (default: cycle through all)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (2.0 = twice real time)")
    ap.add_argument("--no-viewer", action="store_true", help="run headless (for testing)")
    args = ap.parse_args()

    scene, task = load_scene_spec(), load_task(DEFAULT_TASK)
    be = MujocoBackend(render_images=args.policy != "expert")  # learned policies see camera images
    policy = make_policy(args.policy, be, args.ckpt)
    targets = [args.target] if args.target else [o for o in task.objects if o in scene.objects and o != "catheter"]
    rng = np.random.default_rng(args.seed)
    viewer = None
    try:
        for ep in range(args.episodes):
            seed, target = args.seed + ep, targets[ep % len(targets)]
            instruction = str(rng.choice(task.instructions(target)))
            obs = be.reset(scene, seed, target)
            policy.reset()
            if viewer is None and not args.no_viewer:
                viewer = mujoco.viewer.launch_passive(be.model, be.data)
                viewer.cam.lookat[:] = [0.45, 0.05, 0.95]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 200, -30
            print(f"episode {ep + 1}: seed {seed}, \"{instruction}\"", flush=True)
            info, steps = None, 0
            while steps < MAX_EPISODE_S * CONTROL_HZ and not getattr(policy, "done", False):
                for a in policy.predict(obs, instruction):
                    t0 = time.perf_counter()
                    obs, info = be.step(a)
                    steps += 1
                    if viewer is not None:
                        if not viewer.is_running():
                            return
                        viewer.sync()
                        time.sleep(max(0.0, 1.0 / (CONTROL_HZ * args.speed) - (time.perf_counter() - t0)))
                    if info.success:
                        break
                if info.success:
                    break
            failed = next((s for s in STAGES if not info.stage_flags[s]), None)
            print(f"  -> {'SUCCESS' if info.success else 'failed at ' + failed} after {steps / CONTROL_HZ:.1f} s",
                  flush=True)
            if viewer is not None:
                time.sleep(1.0)
    finally:
        if viewer is not None:
            viewer.close()
        be.close()


if __name__ == "__main__":
    main()
