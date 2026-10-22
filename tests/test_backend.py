"""MujocoBackend contract, determinism, stage predicates and latching."""
import mujoco
import numpy as np
import pytest

from cathlab_sim.backend import MujocoBackend
from cathlab_sim.contracts import STAGES, SimBackend
from cathlab_sim.scene import HOME_Q, TABLE_TOP, TRAYS
from cathlab_sim.spec_io import load_scene_spec
from cathlab_sim.stages import latch

SCENE = load_scene_spec()
HOLD = np.concatenate([HOME_Q, [1.0], HOME_Q, [1.0]]).astype(np.float32)


@pytest.fixture(scope="module")
def be():
    b = MujocoBackend(render_images=True)
    yield b
    b.close()


def test_is_a_simbackend(be):
    assert isinstance(be, SimBackend)


def test_obs_schema(be):
    obs = be.reset(SCENE, seed=0, target_object="syringe")
    assert obs["observation.state"].shape == (16,) and obs["observation.state"].dtype == np.float32
    for cam in ("external_camera", "left_wrist_camera", "right_wrist_camera"):
        img = obs[f"observation.images.{cam}"]
        assert img.shape == (224, 224, 3) and img.dtype == np.uint8 and img.std() > 5
    assert be.render().shape == (224, 224, 3)


def test_same_seed_is_bit_identical(be):
    a = be.reset(SCENE, seed=7, target_object="sheath")
    _, ia = be.step(HOLD)
    a2, _ = be.step(HOLD)
    b = be.reset(SCENE, seed=7, target_object="sheath")
    be.step(HOLD)
    b2, _ = be.step(HOLD)
    np.testing.assert_array_equal(a["privileged.target_pos"], b["privileged.target_pos"])
    np.testing.assert_array_equal(a2["observation.state"], b2["observation.state"])
    # Physics is bit-exact; GPU rasterization is not guaranteed to be, so allow
    # a handful of pixels to differ by one intensity level.
    ia_, ib_ = a2["observation.images.external_camera"].astype(int), b2["observation.images.external_camera"].astype(int)
    diff = np.abs(ia_ - ib_)
    assert diff.max() <= 1 and (diff > 0).mean() < 1e-4


def test_different_seeds_move_target(be):
    p1 = be.reset(SCENE, seed=1, target_object="gauze")["privileged.target_pos"]
    p2 = be.reset(SCENE, seed=2, target_object="gauze")["privileged.target_pos"]
    assert np.linalg.norm(p1 - p2) > 0.01


def test_idle_episode_has_no_stages_and_no_collision(be):
    be.reset(SCENE, seed=3, target_object="catheter")
    for _ in range(10):
        _, info = be.step(HOLD)
    assert set(info.stage_flags) == set(STAGES)
    assert not any(info.stage_flags.values()) and not info.success and not info.collision


def _teleport(be, pos, quat=(1, 0, 0, 0)):
    j = be.model.joint(f"{be.target}_joint")
    be.data.qpos[j.qposadr[0]:j.qposadr[0] + 7] = [*pos, *quat]
    be.data.qvel[j.dofadr[0]:j.dofadr[0] + 6] = 0
    mujoco.mj_forward(be.model, be.data)


def test_reach_and_lift_predicates(be):
    obs = be.reset(SCENE, seed=0, target_object="gauze")
    _teleport(be, obs["privileged.left_tcp_pos"] + [0, 0, -0.03])   # 3 cm under the left TCP
    f = be.stages.instant_flags(be.data)
    assert f["reach"] and f["lift"] and not f["grasp"]


def test_transport_and_place_predicates(be):
    be.reset(SCENE, seed=0, target_object="gauze")
    x, y = TRAYS["handover_tray"]
    _teleport(be, (x, y, TABLE_TOP + 0.10))                         # above the tray: transport only
    f = be.stages.instant_flags(be.data)
    assert f["transport"] and not f["place"]
    _teleport(be, (x, y, TABLE_TOP + 0.010 + 0.008 + 0.001))       # on the tray floor (5 mm floor x2 + half-height)
    mujoco.mj_step(be.model, be.data, nstep=300)
    f = be.stages.instant_flags(be.data)
    assert f["transport"] and f["place"]


def test_latch_is_ordered_and_sticky():
    none = {s: False for s in STAGES}
    only_lift = {**none, "lift": True}
    assert latch(none, only_lift) == none                    # lift without reach/grasp does not count
    reached = latch(none, {**none, "reach": True})
    assert reached["reach"] and not reached["grasp"]
    assert latch(reached, none)["reach"]                      # sticky once achieved
    all_true = {s: True for s in STAGES}
    assert all(latch(none, all_true).values())


def test_arms_meeting_flags_collision(be):
    """IK both TCPs to the same point: the grippers must collide."""
    from cathlab_sim.ik import solve_ik
    be.reset(SCENE, seed=4, target_object="syringe")
    meet = np.array([0.55, 0.0, TABLE_TOP + 0.25])
    down = np.array([0, 1, 0, 0])
    a = HOLD.copy()
    for side, sl in (("left", slice(0, 7)), ("right", slice(8, 15))):
        q, pe, _ = solve_ik(be.model, be.data, side, meet, down, q_init=HOME_Q)
        assert pe < 0.005
        a[sl] = q
    hit = False
    for _ in range(60):
        _, info = be.step(a)
        hit |= info.collision
    assert hit
