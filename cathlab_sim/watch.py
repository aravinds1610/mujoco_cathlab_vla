"""Watch the scripted expert do pick-and-place live in the MuJoCo viewer (real time).

Usage: python -m cathlab_sim.watch [--seed 0] [--target syringe] [--episodes 5] [--speed 1.0]
Each episode uses the next seed. Close the viewer window to stop.
"""
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")  # on-screen window; the backend would default to headless EGL
import mujoco
import mujoco.viewer

from cathlab_sim.backend import MujocoBackend
from cathlab_sim.contracts import STAGES
from cathlab_sim.expert import ScriptedExpert
from cathlab_sim.robot_io import CONTROL_HZ
from cathlab_sim.spec_io import load_scene_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default=None, help="object to pick (default: cycle through all)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (2.0 = twice real time)")
    ap.add_argument("--no-viewer", action="store_true", help="run headless (for testing)")
    args = ap.parse_args()

    scene = load_scene_spec()
    be = MujocoBackend(render_images=False)
    targets = [args.target] if args.target else [o for o in scene.objects if o != "catheter"]
    viewer = None
    try:
        for ep in range(args.episodes):
            seed, target = args.seed + ep, targets[ep % len(targets)]
            obs = be.reset(scene, seed, target)
            if viewer is None and not args.no_viewer:
                viewer = mujoco.viewer.launch_passive(be.model, be.data)
                viewer.cam.lookat[:] = [0.45, 0.05, 0.95]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 200, -30
            print(f"episode {ep + 1}: seed {seed}, '{target}'", flush=True)
            actions = ScriptedExpert(be.model).plan(be.data, obs, be.objects[target])
            info = None
            for a in actions:
                t0 = time.perf_counter()
                obs, info = be.step(a)
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    viewer.sync()
                    time.sleep(max(0.0, 1.0 / (CONTROL_HZ * args.speed) - (time.perf_counter() - t0)))
            failed = next((s for s in STAGES if not info.stage_flags[s]), None)
            print(f"  -> {'SUCCESS' if info.success else 'failed at ' + failed}", flush=True)
            if viewer is not None:
                time.sleep(1.0)
    finally:
        if viewer is not None:
            viewer.close()
        be.close()


if __name__ == "__main__":
    main()
