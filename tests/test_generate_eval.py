"""Dataset writer acceptance check (§6) and evaluation harness."""
import json

import numpy as np
import pytest
import yaml

from cathlab_sim.contracts import DatasetManifest, PolicyAdapter
from cathlab_sim.evaluate import evaluate, wilson_ci
from cathlab_sim.generate import DEFAULT_TASK, load_task, run
from cathlab_sim.robot_io import NAMES
from cathlab_sim.spec_io import load_scene_spec


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds") / "handover_v1"
    manifest = run(n=2, out=out, objects=["guidewire", "scissors"], quiet=True)
    return out, manifest


def test_dataset_loads_with_lerobot_and_matches_schema(dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    out, _ = dataset
    ds = LeRobotDataset("cathlab/handover_mujoco_scripted", root=out, video_backend="pyav")
    assert ds.num_episodes == 2 and ds.fps == 30 and ds.meta.robot_type == "fr3_duo"
    feats = ds.meta.features
    for key in ("observation.state", "action"):
        assert feats[key]["dtype"] == "float32" and tuple(feats[key]["shape"]) == (16,) and feats[key]["names"] == NAMES
    for cam in ("external_camera", "left_wrist_camera", "right_wrist_camera"):
        f = feats[f"observation.images.{cam}"]
        assert f["dtype"] == "video" and tuple(f["shape"]) == (224, 224, 3)
    assert feats["success"]["dtype"] == "bool"
    s = ds[0]
    assert s["observation.images.external_camera"].shape == (3, 224, 224)
    assert s["task"] in ("Hand me the guidewire", "Help me with the guidewire", "Please pass the guidewire")
    last = ds[ds.num_frames - 1]
    assert bool(last["success"])  # saved episodes are successful by the end


def test_manifest_and_episode_metadata(dataset):
    out, manifest = dataset
    m = DatasetManifest.model_validate(yaml.safe_load((out / "manifest.yaml").read_text()))
    assert m == manifest and m.source == "scripted" and m.n_episodes == 2 and m.version == "v1"
    rows = [json.loads(l) for l in (out / "meta/cathlab_episodes_extra.jsonl").read_text().splitlines()]
    assert all({"seed", "target", "instruction", "success", "failure_stage"} <= set(r) for r in rows)


def test_existing_dataset_is_never_overwritten(dataset):
    out, _ = dataset
    with pytest.raises(FileExistsError):
        run(n=1, out=out, quiet=True)


def test_wilson_ci():
    lo, hi = wilson_ci(48, 60)
    assert 0.67 < lo < 0.70 and 0.87 < hi < 0.90
    assert wilson_ci(0, 10)[0] == 0.0 and wilson_ci(10, 10)[1] == 1.0


def test_expert_policy_adapter_evaluates():
    from cathlab_sim.backend import MujocoBackend
    from cathlab_sim.expert_policy import ScriptedExpertPolicy
    be = MujocoBackend(render_images=False)
    policy = ScriptedExpertPolicy(be)
    assert isinstance(policy, PolicyAdapter) and policy.meta().action_dim == 16
    report = evaluate(policy, be, load_scene_spec(), load_task(DEFAULT_TASK), [0, 1], ["guidewire"], quiet=True)
    assert report["n"] == 2 and report["successes"] == 2 and report["ci95"][1] == 1.0
    be.close()


def test_constant_columns_have_unit_std(dataset):
    out, _ = dataset
    stats = json.loads((out / "meta/stats.json").read_text())
    right_arm = [NAMES.index(f"right_joint{i}") for i in range(1, 8)]
    for key in ("action", "observation.state"):
        assert all(stats[key]["std"][j] == 1.0 for j in right_arm)  # the expert never moves the right arm
        assert all(v >= 1e-3 for v in stats[key]["std"])
