"""CPU-only tests for the 16-D action/state mapping. Run: pytest -q tests"""
import mujoco
import numpy as np
import pytest

from cathlab_sim.robot_io import GRIPPER_CTRL_MAX, NAMES, RobotIO
from cathlab_sim.scene import HOME_Q, build_spec, set_home


@pytest.fixture(scope="module")
def sim():
    model = build_spec().compile()
    return model, RobotIO(model)


def fresh(model):
    data = mujoco.MjData(model)
    set_home(model, data)
    return data


def test_names_match_dataset_schema():
    expected = ([f"left_joint{i}" for i in range(1, 8)] + ["left_gripper"]
                + [f"right_joint{i}" for i in range(1, 8)] + ["right_gripper"])
    assert NAMES == expected


def test_control_period_is_whole_substeps(sim):
    model, io = sim
    assert io.n_substeps * model.opt.timestep == pytest.approx(1 / 30)


def test_gripper_command_mapping(sim):
    model, io = sim
    data = fresh(model)
    a = np.concatenate([HOME_Q, [1.0], HOME_Q, [0.0]])  # left open, right closed
    io.apply_action(data, a)
    assert data.ctrl[io.act_ids[7]] == pytest.approx(0.0)
    assert data.ctrl[io.act_ids[15]] == pytest.approx(GRIPPER_CTRL_MAX)


def test_home_state_reads_back(sim):
    model, io = sim
    data = fresh(model)
    s = io.get_state(data)
    assert s.dtype == np.float32 and s.shape == (16,)
    np.testing.assert_allclose(s[:7], HOME_Q, atol=1e-6)
    np.testing.assert_allclose(s[[7, 15]], [1.0, 1.0], atol=1e-3)  # open at start


def test_action_tracks_to_state(sim):
    """Hold a 16-D target for 1.5 s: measured state must reach it."""
    model, io = sim
    data = fresh(model)
    rng = np.random.default_rng(0)
    target = np.concatenate([HOME_Q + rng.uniform(-0.2, 0.2, 7), [0.0],
                             HOME_Q + rng.uniform(-0.2, 0.2, 7), [1.0]])  # left closes, right stays open
    for _ in range(45):  # 45 frames at 30 Hz = 1.5 s
        s = io.step(data, target)
    arm = ~io.is_gripper
    np.testing.assert_allclose(s[arm], target[arm], atol=0.01)
    assert s[7] < 0.05, f"left gripper should be closed, got {s[7]:.3f}"
    assert s[15] > 0.95, f"right gripper should be open, got {s[15]:.3f}"
