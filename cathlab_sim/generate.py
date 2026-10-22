"""Generate a LeRobot v3 dataset of scripted-expert demonstrations.

    python -m cathlab_sim.generate --n 200 --out <dir> [--gap gap.json] [--objects syringe sheath]

Writes, under <dir>:
  - a LeRobot v3.0 dataset (robot_type fr3_duo, 30 fps, 16-D state/action, 3 videos, `success`,
    episode `task` = the natural-language instruction), via lerobot's own writer;
  - meta/cathlab_episodes_extra.jsonl: per-attempt seed, target, instruction, outcome, failure stage;
  - manifest.yaml (contracts.DatasetManifest, source "scripted").
Only successful episodes are saved unless --keep-failures is given. Seeds are recorded per episode;
re-running with the same arguments reproduces the same episodes.
"""
import argparse
import getpass
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cathlab_sim.backend import MujocoBackend
from cathlab_sim.contracts import STAGES, DatasetManifest, GapSpec, TaskSpec
from cathlab_sim.expert import ScriptedExpert
from cathlab_sim.robot_io import CONTROL_HZ, NAMES
from cathlab_sim.scene import CAMERAS, IMAGE_SIZE
from cathlab_sim.spec_io import REPO, load_scene_spec, resolve

DEFAULT_TASK = REPO / "configs/tasks/handover.yaml"
GAP_OBJECT_PROB = 0.7  # with a GapSpec, this share of episodes targets the gap's objects
MIN_STD = 1e-3  # below this a column is constant (e.g. the idle right arm); its std is set to 1


def floor_constant_stats(root: Path) -> list[str]:
    """Set std = 1 for constant columns in meta/stats.json and return their names.

    Policies normalize with (x - mean) / (std + 1e-8). For a joint that never moves, std is
    float rounding noise (~1e-8), so normalizing turns that noise into values of +-50 and the
    training loss is dominated by it. With std = 1 those columns normalize to ~0.
    """
    path = root / "meta/stats.json"
    stats = json.loads(path.read_text())
    fixed = []
    for key in ("action", "observation.state"):
        std = stats[key]["std"]
        for j, v in enumerate(std):
            if v < MIN_STD:
                std[j] = 1.0
                fixed.append(f"{key}[{NAMES[j]}]")
    path.write_text(json.dumps(stats, indent=4))
    return fixed


def features() -> dict:
    vec = {"dtype": "float32", "shape": (16,), "names": NAMES}
    feats = {"observation.state": dict(vec), "action": dict(vec)}
    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                                              "names": ["height", "width", "channel"]}
    feats["success"] = {"dtype": "bool", "shape": (1,), "names": None}
    return feats


def load_task(path) -> TaskSpec:
    return TaskSpec.model_validate(yaml.safe_load(resolve(path).read_text()))


def episode_plan(i: int, rng: np.random.Generator, objects: list[str], gap: GapSpec | None):
    """Target object for attempt i: round-robin, or biased toward the gap's objects."""
    if gap is not None and gap.objects and rng.random() < GAP_OBJECT_PROB:
        return str(rng.choice([o for o in gap.objects if o in objects]))
    return objects[i % len(objects)]


def apply_gap_to_scene(scene, gap: GapSpec | None):
    """Bias scene randomization for known gap variants (unknown variants are recorded only)."""
    if gap is None:
        return scene
    rnd = dict(scene.randomization)
    if "heavy_clutter" in gap.scene_variants:
        hi = rnd["n_distractors"][1]
        rnd["n_distractors"] = [hi, hi]
    return scene.model_copy(update={"randomization": rnd})


def run(n: int, out: Path, scene_path=None, task_path=DEFAULT_TASK, gap: GapSpec | None = None,
        objects: list[str] | None = None, seed0: int = 0, keep_failures: bool = False,
        dataset_id: str = "cathlab/handover_mujoco_scripted", version: str = "v1",
        max_attempts_factor: int = 3, quiet: bool = False) -> DatasetManifest:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; datasets are versioned, never overwritten")
    scene = apply_gap_to_scene(load_scene_spec(scene_path) if scene_path else load_scene_spec(), gap)
    task = load_task(task_path)
    objects = objects or [o for o in task.objects if o in scene.objects]
    rng = np.random.default_rng(seed0)

    ds = LeRobotDataset.create(repo_id=dataset_id, fps=CONTROL_HZ, features=features(), root=out,
                               robot_type="fr3_duo", use_videos=True)
    be = MujocoBackend(render_images=True)
    extra, saved, attempts, t0 = [], 0, 0, time.time()
    try:
        while saved < n and attempts < n * max_attempts_factor:
            seed = seed0 + attempts
            target = episode_plan(attempts, rng, objects, gap)
            instruction = str(rng.choice(task.instructions(target)))
            attempts += 1

            obs = be.reset(scene, seed, target)
            actions = ScriptedExpert(be.model).plan(be.data, obs, be.objects[target])
            info = None
            for a in actions:
                frame_obs = obs
                obs, info = be.step(a)
                frame = {k: v for k, v in frame_obs.items() if not k.startswith("privileged.")}
                frame.update({"action": a.astype(np.float32), "success": np.array([info.success]),
                              "task": instruction})
                ds.add_frame(frame)

            failed = next((s for s in STAGES if not info.stage_flags[s]), None)
            keep = info.success or keep_failures
            if keep:
                ds.save_episode()
                saved += 1
            else:
                ds.clear_episode_buffer()
            extra.append({"episode_index": saved - 1 if keep else None, "seed": seed, "target": target,
                          "instruction": instruction, "success": info.success, "failure_stage": failed,
                          "collision_any": None, "n_frames": len(actions), "saved": keep})
            if not quiet:
                print(f"[{attempts:4d}] seed {seed:5d} {target:10s} {'OK  ' if info.success else 'FAIL'}"
                      f"{'' if info.success else ' at ' + failed:14s} saved {saved}/{n}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        ds.finalize()
        be.close()

    floor_constant_stats(out)
    wins = sum(e["success"] for e in extra)
    (out / "meta").mkdir(exist_ok=True)
    with open(out / "meta/cathlab_episodes_extra.jsonl", "w") as f:
        for e in extra:
            f.write(json.dumps(e) + "\n")
    per_obj = {o: f"{sum(e['success'] for e in extra if e['target'] == o)}/"
                  f"{sum(e['target'] == o for e in extra)}" for o in objects}
    manifest = DatasetManifest(
        dataset_id=dataset_id, version=version, source="scripted", task_id=task.task_id,
        scene_id=scene.scene_id, n_episodes=saved, created_by=getpass.getuser(),
        quality_score=round(wins / max(attempts, 1), 4),
        notes=(f"scripted IK expert v1 (ground-truth grasp). attempts={attempts}, seeds {seed0}..{seed0 + attempts - 1}; "
               f"expert success per object {per_obj}; keep_failures={keep_failures}; "
               f"gap={gap.model_dump() if gap else None}"))
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest.model_dump(), sort_keys=False))
    if not quiet:
        print(f"saved {saved} episodes from {attempts} attempts -> {out}\nmanifest: {manifest.notes}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="episodes to save")
    ap.add_argument("--out", type=Path, required=True, help="new dataset directory (must not exist)")
    ap.add_argument("--gap", type=Path, default=None, help="GapSpec JSON to bias sampling toward")
    ap.add_argument("--objects", nargs="+", default=None, help="restrict targets (default: all task objects)")
    ap.add_argument("--scene", default=None, help="SceneSpec YAML (default: configs/scenes/cathlab_minimal.yaml)")
    ap.add_argument("--task", default=DEFAULT_TASK, help="TaskSpec YAML")
    ap.add_argument("--seed0", type=int, default=0, help="first episode seed")
    ap.add_argument("--keep-failures", action="store_true", help="also save failed episodes")
    ap.add_argument("--dataset-id", default="cathlab/handover_mujoco_scripted")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    gap = GapSpec.model_validate_json(args.gap.read_text()) if args.gap else None
    run(args.n, args.out, args.scene, args.task, gap, args.objects, args.seed0, args.keep_failures,
        args.dataset_id, args.version)


if __name__ == "__main__":
    main()
