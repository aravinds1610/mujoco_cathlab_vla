"""Seeded layouts: reproducible, varied, non-overlapping, physically stable."""
import mujoco
import numpy as np
import pytest

from cathlab_sim.layout import PARK_X, _corners, _overlap, apply_layout, sample_layout
from cathlab_sim.scene import TABLE_TOP, build_spec, set_home
from cathlab_sim.spec_io import load_object_defs, load_scene_spec

SCENE = load_scene_spec()
OBJS = load_object_defs(SCENE)
RND = SCENE.randomization


def layout(seed, target="sheath"):
    return sample_layout(np.random.default_rng(seed), RND, OBJS, target)


def test_same_seed_same_layout():
    assert layout(42) == layout(42)


def test_different_seeds_differ():
    assert layout(1) != layout(2)


@pytest.mark.parametrize("target", sorted(OBJS))
def test_target_present_and_count_in_range(target):
    for seed in range(20):
        lay = layout(seed, target)
        assert target in lay
        assert 1 + RND["n_distractors"][0] <= len(lay) <= 1 + RND["n_distractors"][1]


def test_no_overlaps_over_many_seeds():
    for seed in range(100):
        rects = [_corners(x, y, yaw, OBJS[n].footprint) for n, (x, y, yaw) in layout(seed).items()]
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                assert not _overlap(rects[i], rects[j], 0.0), f"seed {seed}: overlap"


def test_unknown_target_rejected():
    with pytest.raises(ValueError):
        layout(0, target="scalpel")


def test_layout_settles_in_sim():
    model = build_spec(SCENE).compile()
    for seed in range(5):
        data = mujoco.MjData(model)
        set_home(model, data)
        lay = layout(seed)
        apply_layout(model, data, lay, OBJS)
        mujoco.mj_step(model, data, nstep=600)  # 1 s
        for name, obj in OBJS.items():
            p = data.body(name).xpos
            if name in lay:
                assert abs(p[2] - (TABLE_TOP + obj.half_height)) < 0.004, f"seed {seed} {name} z {p[2]:.3f}"
                assert np.linalg.norm(p[:2] - lay[name][:2]) < 0.01, f"seed {seed} {name} drifted"
            else:
                assert p[0] == pytest.approx(PARK_X, abs=0.01), f"{name} should stay parked"
