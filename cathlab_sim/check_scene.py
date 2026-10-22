"""Pass/fail check: scene compiles, matches the 16-D schema, is stable, renders
3 cameras at 224x224, and both arms can reach their work areas pointing down.

Usage: python -m cathlab_sim.check_scene [--png out.png]
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering
import mujoco
import numpy as np

from cathlab_sim.ik import solve_ik
from cathlab_sim.layout import apply_layout, sample_layout
from cathlab_sim.scene import CAMERAS, HOME_Q, IMAGE_SIZE, SIDES, TABLE_TOP, TRAYS, build_spec, set_home
from cathlab_sim.spec_io import load_object_defs, load_scene_spec

SCENE = load_scene_spec()
RND = SCENE.randomization
AREA_CORNERS = {f"source_corner_{i}": (x, y, TABLE_TOP + 0.10)
                for i, (x, y) in enumerate([(RND["area_x"][a], RND["area_y"][b]) for a in (0, 1) for b in (0, 1)])}

# Gripper pointing straight down (TCP z = world -z), fingers opening along world y
DOWN_QUATS = [np.array(q) for q in ([0, 1, 0, 0], [0, 0, 1, 0])]  # two yaw options, 90 deg apart
REACH = {  # side -> named targets 10 cm above the surface
    "left":  {**AREA_CORNERS, "handover_point": (0.55, 0.0, TABLE_TOP + 0.25),
              "handover_tray": (*TRAYS["handover_tray"], TABLE_TOP + 0.10)},
    "right": {"handover_tray": (*TRAYS["handover_tray"], TABLE_TOP + 0.10), "handover_point": (0.55, 0.0, TABLE_TOP + 0.25)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None, help="save a 3-camera mosaic")
    args = ap.parse_args()

    model = build_spec(SCENE).compile()
    objects = load_object_defs(SCENE)
    data = mujoco.MjData(model)
    acts = [model.actuator(i).name for i in range(model.nu)]
    print(f"actuators ({model.nu}): {acts}")

    set_home(model, data)
    layout = sample_layout(np.random.default_rng(0), RND, objects, target="syringe")
    apply_layout(model, data, layout, objects)
    print(f"layout (seed 0): {sorted(layout)}")
    mujoco.mj_step(model, data, nstep=round(2.0 / model.opt.timestep))  # settle 2 s

    ok = True
    ok &= check(model.nu == 16, f"16 actuators (got {model.nu})")
    ok &= check(np.all(np.isfinite(data.qpos)), "state finite after 2 s")
    for side in SIDES:
        q = [data.joint(f"{side}_fr3v2_joint{i}").qpos[0] for i in range(1, 8)]
        err = np.max(np.abs(np.array(q) - HOME_Q))
        ok &= check(err < 0.02, f"{side} arm holds home pose (max err {err:.4f} rad)")
    for n in layout:
        z = data.body(n).xpos[2]
        expected = TABLE_TOP + objects[n].half_height
        moved = np.linalg.norm(data.body(n).xpos[:2] - layout[n][:2])
        ok &= check(abs(z - expected) < 0.004 and moved < 0.01,
                    f"{n} resting on table (expected z {expected:.3f}, got {z:.3f}; drifted {moved*1000:.1f} mm)")

    # Gripper fingers must open along the flange y axis, like the Duo
    for side in SIDES:
        R8 = data.body(f"{side}_fr3v2_link8").xmat.reshape(3, 3)
        d = R8.T @ (data.body(f"{side}_gripper_right_pad").xpos - data.body(f"{side}_gripper_left_pad").xpos)
        ok &= check(abs(d[1]) > 0.05 and abs(d[0]) < 0.005, f"{side} fingers open along flange y (sep {d.round(3)})")

    images = []
    with mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE) as r:
        for cam in CAMERAS:
            r.update_scene(data, camera=cam)
            img = r.render()
            images.append(img)
            ok &= check(img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and img.std() > 10,
                        f"{cam} renders {img.shape} (pixel std {img.std():.1f})")

    for side, targets in REACH.items():
        for name, pos in targets.items():
            best = min((solve_ik(model, data, side, np.array(pos), q, q_init=HOME_Q) for q in DOWN_QUATS),
                       key=lambda r: r[1] + 0.1 * r[2])
            _, pe, re = best
            ok &= check(pe < 0.005 and re < 0.03,
                        f"{side} reaches {name} pointing down (pos err {pe*1000:.1f} mm, rot err {np.degrees(re):.1f} deg)")

    if args.png:
        import PIL.Image
        PIL.Image.fromarray(np.hstack(images)).save(args.png)
        print(f"saved {args.png}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


def check(cond, msg):
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    return bool(cond)


if __name__ == "__main__":
    main()
