"""The contracts must accept the handoff's examples and reject wrong types."""
import pydantic
import pytest

from cathlab_sim.contracts import STAGES, DatasetManifest, GapSpec, SceneSpec, StepInfo, TaskSpec


def test_taskspec_instructions():
    t = TaskSpec(task_id="handover", instruction_templates=["Hand me the {object}", "Please pass the {object}"],
                 objects=["syringe"], success_predicate="place",
                 stage_predicates={s: s for s in STAGES})
    assert t.instructions("syringe") == ["Hand me the syringe", "Please pass the syringe"]


def test_stage_keys_are_validated():
    with pytest.raises(pydantic.ValidationError):
        TaskSpec(task_id="x", instruction_templates=[], objects=[], success_predicate="",
                 stage_predicates={"fly": "nope"})


def test_manifest_rejects_unknown_source():
    base = dict(dataset_id="cathlab/handover_mujoco_scripted", version="v1", task_id="handover",
                scene_id="cathlab_minimal", n_episodes=1, created_by="aravind")
    DatasetManifest(source="scripted", **base)
    with pytest.raises(pydantic.ValidationError):
        DatasetManifest(source="magic", **base)


def test_gapspec_from_json():
    g = GapSpec.model_validate_json('{"failed_stage": "grasp", "objects": ["sheath"], '
                                    '"scene_variants": ["heavy_clutter"], "n_failures": 7, '
                                    '"suggested_generator": "scripted"}')
    assert g.objects == ["sheath"] and g.n_failures == 7


def test_scenespec_and_stepinfo_defaults():
    s = SceneSpec(scene_id="cathlab_minimal", backend="mujoco", asset="x.yaml", objects={}, cameras=[])
    assert s.randomization == {}
    assert StepInfo(stage_flags={k: False for k in STAGES}, success=False).collision is False
