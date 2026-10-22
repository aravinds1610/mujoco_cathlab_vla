"""Open the scene in MuJoCo's interactive viewer with a seeded random layout.

Usage: python -m cathlab_sim.view [--seed 0] [--target syringe]
"""
import argparse

import mujoco
import mujoco.viewer
import numpy as np

from cathlab_sim.layout import apply_layout, sample_layout
from cathlab_sim.scene import build_spec, set_home
from cathlab_sim.spec_io import load_object_defs, load_scene_spec

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--target", default="syringe")
args = ap.parse_args()

scene = load_scene_spec()
objects = load_object_defs(scene)
model = build_spec(scene).compile()
data = mujoco.MjData(model)
set_home(model, data)
layout = sample_layout(np.random.default_rng(args.seed), scene.randomization, objects, args.target)
apply_layout(model, data, layout, objects)
print(f"seed {args.seed}, target {args.target}, objects on table: {sorted(layout)}")
mujoco.viewer.launch(model, data)
