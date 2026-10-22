# MuJoCo track: status

*Aravind → Rakesh, 2026-10-22*

- **Repo:** https://github.com/aravinds1610/mujoco_cathlab_vla
- **Dataset v1:** https://huggingface.co/datasets/aravindsuresh16/cathlab_handover_mujoco_v1

## Summary

The MuJoCo backend, a scripted expert, the dataset generator and the evaluation harness all work and match the shared interfaces. A 200-episode LeRobot v3 dataset is published. The expert scores **78% (95% CI 66–87%)** on 60 held-out seeds.

A first SmolVLA fine-tune runs end to end but scores **0/60**. Diagnostics point to undertraining and narrow data, not a pipeline bug.

Not started: mesh objects (AR4), GraspGen (AR1/AR5) and the Isaac bake-off (AR6). Three decisions from you would unblock the next steps (see [Decisions needed](#decisions-needed)).

## Deliverables

| # | Deliverable | Status | Notes |
|---|---|---|---|
| AR1 | GraspGen spike | ❌ Not started | Waiting on the licence and gripper decision below |
| AR2 | FR3 Duo model, 16-D mapping | ✅ Done | Menagerie `franka_fr3_v2` ×2 + Robotiq 2F-85. Base poses, mount height and camera poses taken from the Isaac USD. Doc: `docs/AR2_robot_model.md` |
| AR3 | `MujocoBackend` + cathlab scene from `SceneSpec` YAML | ✅ Done | Stage flags and collisions from sim state; seeded clutter (target + 2–5 distractors, random pose and yaw) |
| AR4 | `add_object(name, mesh_path)` mesh pipeline | ❌ Not started | 4 CC BY 4.0 meshes downloaded (syringe, scissors, catheter, gauze) in `assets/meshes/`, with attribution |
| AR5 | Generator: ≥200 episodes, all 6 objects, GraspGen, `GapSpec` | ⚠️ Partial | 200 episodes ✅, `GapSpec` ✅, one-command CLI ✅. **5 of 6 objects** (catheter excluded, see Known issues). Grasp comes from the **ground-truth pose**, not GraspGen |
| AR6 | MuJoCo vs Isaac bake-off report | ❌ Not started | MuJoCo-side numbers mostly exist (below). Needs Isaac-side numbers from the team |

### Acceptance checks

| Check | Status |
|---|---|
| Loads with `LeRobotDataset(repo_id, root=path)`, schema exact (keys, shapes, fps 30) | ✅ Checked by a test, and loaded from the HF Hub |
| Expert as `PolicyAdapter`, ≥50 seeds, 95% CI | ✅ 47/60 = 78.3% (Wilson CI 66.4–86.9%) |
| Tests run without a GPU | ⚠️ 36 tests pass, but they render with EGL (GPU). CPU rendering (OSMesa) is not installed yet, so a CPU-only run is untested |
| `python -m cathlab_sim.generate --n 200 --out <dir> [--gap gap.json]` | ✅ |
| Licences and attribution | ⚠️ Menagerie and meshes attributed. ContactPilot and franka-vla-generalization have no licence file, so no code was copied from them |

## Integration notes

- **Contracts:** `cathlab_sim/contracts.py` is the shared interface code copied verbatim. **One deviation:** `GapSpec` is a pydantic `BaseModel` (the brief only lists its fields), so it loads from `gap.json`.
- **Backend:** `MujocoBackend.reset(scene, seed, target_object)` and `.step(action16)` return `(obs, StepInfo)`.
  - Observation keys follow the dataset names: `observation.state` and `observation.images.{external,left_wrist,right_wrist}_camera` (224×224 uint8).
  - It also provides `privileged.*` keys (target pose, TCP positions) for experts only.
- **Dataset:** `robot_type: fr3_duo`, 30 fps, `action`/`observation.state` float32[16] in the agreed order, 3 AV1 videos, `success` per frame, `task` = instruction. It also writes:
  - `manifest.yaml` (`DatasetManifest`, `source: scripted`);
  - `meta/cathlab_episodes_extra.jsonl`: seed, target, instruction and failure stage for every attempt, including the failed attempts that were not saved.
- **Not included:** the Siemens recorder fields `raw_state_row` / `raw_action_row`.
- **Normalization caveat:** the expert only moves the left arm, so the right-arm columns are constant. Their std is rounding noise (~1e-8), which inflated SmolVLA's normalized loss to ~700. The generator now sets std = 1 for constant columns in `meta/stats.json`. If you compute your own stats, do the same.
- **Run it:** the README covers setup (one `pip install` + a Menagerie clone), generate, evaluate, fine-tune and live viewing (`python -m cathlab_sim.watch --policy expert|smolvla`).

## Results

Held-out seeds 10000–10059, 10 per object, all 6 objects.

| Policy | Success | 95% CI | Failure stages | Latency |
|---|---|---|---|---|
| Scripted IK expert | 47/60 = 78% | 66–87% | lift 13 | 6.5 ms per call |
| SmolVLA (`smolvla_base`, 3k steps, batch 8, v1) | 0/60 = 0% | 0–6% | reach 58, grasp 1, lift 1 | 119 ms per 25-action chunk |

Expert per object: guidewire 10/10, syringe 10/10, sheath 9/10, gauze 8/10, scissors 7/10, catheter 3/10 (unreliable, see Known issues).

**Dataset v1:** 200 successful episodes from 236 attempts, 91,375 frames, 14–17 s each.
- Episodes per object: guidewire 48, syringe 46, sheath 42, scissors 42, **gauze 22**.
- The 3 instruction templates are roughly balanced.
- All 16 table cells and 8 yaw bins are covered.
- Gripper actions are clean (exactly 2 per episode) and there are no action jumps.
- QA report: `reports/qa_v1.json`.

**What the diagnostics show** (scripts in `cathlab_sim/tools/`):

- **Expert failures are missed grasps, not slips.** On failing seeds the gripper closes almost fully on air (0–6 N grip); on successful seeds it stops on the object at about 100 N. The object isn't pushed beforehand. Still to determine: wrong finger orientation (a 7 × 7 cm gauze is 9.9 cm on the diagonal, wider than the 85 mm opening) or a sideways offset.
- **SmolVLA learned the motions but not the targeting.** On training frames its 25-step action error is 0.035 rad, against 0.087 rad for holding still. The error is largest at the start of an episode, where the model must find the target in the images. Live images and state match the dataset (within video compression), so the adapter is fine. It saw only 26% of one pass over the data, so the next step is longer training (≥11k steps, one pass) on more varied data.

**Hardware (RTX 4060 laptop, 8 GB):** SmolVLA fine-tuning at batch 8 uses about 4.5 GB and runs at about 2.3 steps/s (one pass is about 80 min). AMP fails in lerobot 0.6.1 because SmolVLA already runs partly in bfloat16.

## Known issues

- **Catheter is unstable** in MuJoCo (NaN accelerations on its joint; a thin 4 mm capsule). It is excluded from v1, and its eval numbers are unreliable.
- **Left arm only.** The 16-D action is complete, but the right arm holds its home pose.
- **Success-only demos:** no recovery behaviour in the data.
- **Narrow visuals:** one table, one tray position, fixed lighting and colours, primitive shapes (capsules and boxes).
- **Gauze under-represented** (22 episodes) because the expert succeeds on it only about 47% of the time.

## Decisions needed

1. **GraspGen:** it has an NVIDIA non-commercial research licence, and its released checkpoint is for the Panda gripper, not our Robotiq 2F-85. Is a ground-truth-pose grasp acceptable for AR5, or should I spend time on GraspGen (AR1 spike first)?
2. **Objects:** is 5 of 6 objects acceptable for now, with the catheter deferred?
3. **SmolVLA:** should I keep training on my side, or do you take the dataset into the team's training pipeline? A useful run needs about 80+ min of GPU here.

I also need from the team: **Isaac-side numbers for AR6** (scene and object authoring time, episodes per hour, expert success rate).

## Next steps (proposed order)

1. **Expert:** find the grasp-miss cause and fix it, add check-and-retry after lift, and use the arm nearer the object.
2. **AR4:** mesh pipeline (CoACD convex decomposition) for at least 2 objects.
3. **Dataset v2** (parent v1): meshes, visual randomization (colours, textures, lighting, small camera shifts), extra gauze episodes via `GapSpec`, optional retry demos.
4. **SmolVLA:** a longer run on v2, then held-out eval plus eval on unseen scene variations.
5. **AR6** bake-off report.
6. CPU-only test run (OSMesa).
