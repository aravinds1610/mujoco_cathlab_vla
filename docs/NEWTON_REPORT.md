# FR3 Duo in Newton: synthetic data report

*Aravind, 2026-10-23. Newton 1.6.0, MuJoCo-Warp 3.12.0, MuJoCo 3.12.0, Warp 1.18.0. Laptop: i9-14900HX (32 threads), RTX 4060 Laptop 8 GB.*

**Dataset:** [`aravindsuresh16/cathlab_fr3_newton_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1), tags `v1` and `v3.0` (LeRobot v3). 384 episodes (159,568 frames), one per cell of the train grid, all 3 objects balanced. Verified from the Hub: `LeRobotDataset(repo, revision="v1")` loads `action (16,)`, `observation.state (16,)`, 3 images `(3, 224, 224)`, fps 30, `robot_type fr3_duo`, `codebase_version v3.0`.

## Summary

- The right-arm FR3 Duo task runs in Newton's MuJoCo-Warp solver with parallel worlds and batched cameras. Its physics and images match CPU MuJoCo (§2).
- The scripted expert succeeds **100 %** on every object and every tier, in both engines (§4, §5).
- **Sim2sim:** CPU MuJoCo and Newton agree on **60/60** episode outcomes. The object's final position differs by a median of 0.05–1.3 mm (§5).
- **Throughput:** on this laptop, **Newton is not faster than CPU MuJoCo for this task**. Newton reaches ~1,580 episodes/h at 32 worlds, about 1.35× a single CPU process but about 0.5× a 12-process CPU pool (§3). The GPU is compute-bound at 32 worlds (100 % utilisation) and uses only ~1.2 GB of VRAM. A larger GPU is where Newton would pull ahead; that is not measured here.

## 1. What was built

All code is in `cathlab_sim/newton/`. Run it in `.venv-newton` (Newton). Conversion runs in `.venv` (LeRobot 0.6.1). Never install both into one env.

| Module | Role |
|---|---|
| `task_scene.py` | Right-arm task scene (MjSpec): v1 robot, cameras and table, plus the team objects and a tray at +y. `to_newton_xml()` exports it for Newton |
| `variation.py` | Team code, copied unchanged (credit line at the top) |
| `configs/variation/cathlab_fr3_newton.yaml` | Our geometry, with the team's grid (palettes, table colours, splits, ranges) |
| `cpu_sim.py` | CPU MuJoCo twin: plans episodes, and is the CPU side of sim2sim and the fallback engine |
| `scene.py` | `build_newton()` and `NewtonTaskSim`: N worlds, per-world randomisation, batched 3-camera rendering |
| `task_stages.py` | Stage predicates shared by both engines, computed from poses and gripper joints only |
| `expert_fr3.py` | Right-arm scripted expert; `--probe` runs the reach probe |
| `runner.py` | `check` (N2) and `generate` (N4, `--engine newton\|mujoco`) |
| `randomize.py` | N3 preview and check |
| `report_n5.py` | Sim2sim and held-out tiers |
| `convert.py` | Raw `.npz` → LeRobot v3, plus push to the Hub |
| `check_n1.py`, `preview.py` | N1 go/no-go and the preview GIF |

**Pipeline:** plan on CPU MuJoCo (process pool, from ground-truth poses) → execute the same 16-D action sequence in Newton (N worlds) → judge success from Newton's state → raw `.npz` per successful episode → convert.

## 2. N1: Newton port (go at ~15:25)

Results are in `reports/newton_n1_check.json`, `newton_n1_worlds.png` and `newton_n1_cpu_vs_newton.png`.

| Check | Result |
|---|---|
| FK vs CPU MuJoCo (right TCP, pinch point) | 0.002 / 0.003 mm |
| Objects at rest, 300 frames holding `HOME_Q` | max drift **0.24 mm** (limit 2 mm), no NaN |
| Right gripper close / open, empty hand | state 0.023 closed (CPU: 0.023), 0.997 reopened |
| Images | Same viewpoints, objects and poses as MuJoCo renders. Remaining differences: shading, shadows and floor brightness (different renderer) |

### Newton workarounds (each one was a real failure first)

1. **`trimesh` missing.** `newton[sim]` does not pull it in, and `add_mjcf` needs it for meshes. Installed `trimesh` alone, not the heavy `[importers]` extra. Without `scipy` the import took 76 s; with `scipy` it takes 3.4 s.
2. **Mesh paths.** `MjSpec.to_xml()` drops the per-model `meshdir` (the arm and gripper differ). `to_newton_xml()` writes absolute paths.
3. **Short `solref`/`solimp`.** `to_xml()` writes `solreflimit="0.005 1"` as `"0.005"`. MuJoCo refills the default; Newton reads 0, which breaks the gripper's joint limits (Newton warns "will silently misbehave"). `to_newton_xml()` pads these values to full length. `solreffriction` is left alone, because its `0 0` default means "use solref".
4. **`nefc overflow`.** The automatic `njmax` (~64 rows per world) overflowed once gripper and multi-point contacts were active, which silently dropped contacts: drift was 2.15 mm, a fail. We now set `njmax=600`. MuJoCo-Warp raises `nconmax` to 1139 by itself.
5. **Cylinder–box contact gives one point** in MuJoCo-Warp, even with multiccd, so the upright bottle and vial would rock. Each has a hidden 2 mm box "foot" inscribed in its base (box–box is multi-point). The syringe barrel is a capsule, as in v1.
6. **Collision geoms are imported invisible.** The table, tray, floor and objects are collision-only, so they did not render. They are now marked `VISIBLE` on the builder; the robot's collision meshes stay hidden.
7. **`<material>` colours are ignored.** The arms rendered flat grey. Colours are copied from the MuJoCo model per body, in file order.
8. **`add_mjcf` does not import `<camera>`.** All 3 cameras are rebuilt from `isaac_reference.json`: head fixed, wrists from `link8` per world. The ray convention matches MuJoCo's (x right, y up, looking along −z).
9. **No light intensity or colour in the tiled renderer.** It renders linear HDR; a Warp kernel multiplies each world's image by its light gain (intensity × colour-temperature RGB), then encodes to sRGB. With one light colour this equals scaling the lights. The output matches a numpy reference to ±1 level. Moving this to the GPU almost doubled generation speed (720 → 1,330 episodes/h at 16 worlds).
10. **Integrator / multiccd parity.** The CPU scene uses `implicitfast` (Newton's default). Multiccd is default-on in MuJoCo ≥ 3.4 and opt-in in 3.3.7; this is handled in code.

## 3. Throughput (task episodes of ~455 frames, three 224² cameras, compressed `.npz` written)

| Engine | Setup | Episodes/h |
|---|---|---|
| Newton (MuJoCo-Warp) | 16 worlds, numpy sRGB (first slice) | ~720 |
| Newton (MuJoCo-Warp) | 16 worlds, GPU sRGB | ~1,330 |
| Newton (MuJoCo-Warp) | **32 worlds**, GPU sRGB | **~1,580** (GPU 100 %, VRAM ~1.2 GB) |
| CPU MuJoCo + EGL | **1 process** | ~1,170 |
| CPU MuJoCo + EGL | 12 processes | ~2,990 (while a LeRobot conversion shared the CPU) |

Without rendering (physics only): Newton executes ~700 world-frames/s at 20 worlds. The CPU pool planned *and* executed the same 60 episodes in 5.7 s versus 39 s for Newton.

**Reading:** with these small scenes, the laptop GPU saturates on compute at about 32 worlds. Planning (CPU IK) is not the bottleneck: it takes ~6 s per 60 episodes on 12 processes. The case for Newton here is the second physics engine (§5) and the NVIDIA stack, not raw speed on this machine. Expect a different answer on the RTX 6000.

## 4. N2: right arm and expert

- **Reach probe:** all 4 pick-region corners (x 0.40–0.72, y −0.50 to −0.08) and the tray centre (0.58, 0.26), 5 cm above the table with the gripper pointing down, reach to within **≤ 0.05 mm**. Region and tray are kept as the mirror of v1.
- **Expert check** (`reports/newton_expert_check.json`, first 20 train configs per object): **bottle 20/20, vial 20/20, syringe 20/20** in Newton, and the same in CPU MuJoCo. Mean episode length: 458–461 frames (~15 s).
- **Grasps:**
  - vial and syringe as in v1: pads at the object centre, closing across the narrow axis (for the syringe, the barrel centre);
  - **bottle: a top grasp across its 77 mm body, pads ~3 cm below the top**, instead of the side grasp suggested in the handoff. It works 100 %, and it avoids bringing the wrist and camera down near the table. The 85 mm opening leaves 4 mm per side; IK is accurate to < 0.1 mm;
  - upright objects try 8 yaws, and the one with least joint motion is kept.
- **Grasp-miss fix:** v1's "fingers close on air" happened on wide, flat objects (gauze, scissors). The team objects are narrow, and none were seen to miss.
- **Not done:** check-and-regrasp after lift. With 100 % success there was nothing for it to fix.

## 5. N5: sim2sim and generalisation

### FR3 sim2sim
Results are in `reports/newton_n5.json`. 60 train configs (the last 60 in generation order, not used by N2). The same planned trajectories run in both engines.

| Object | CPU MuJoCo | Newton | Final object position, CPU vs Newton (median / max) |
|---|---|---|---|
| bottle | 20/20 | 20/20 | 1.31 / 2.12 mm |
| vial | 20/20 | 20/20 | 0.05 / 0.98 mm |
| syringe | 20/20 | 20/20 | 0.06 / 0.79 mm |

**Agreement: 60/60 (100 %).** The bottle diverges most, which fits it being the heaviest object, held in a single-contact cylinder grasp.

### Expert success, train vs held-out tiers
Team `eval_configs`, seeds 100000–100009, 10 per object per tier, in Newton.

| Tier | bottle | vial | syringe | CPU MuJoCo (all) |
|---|---|---|---|---|
| train (N2) | 20/20 | 20/20 | 20/20 | 100 % |
| in_distribution | 10/10 | 10/10 | 10/10 | 100 % |
| held_out_layout (8–9) | 10/10 | 10/10 | 10/10 | 100 % |
| held_out_obj_color (4) | 10/10 | 10/10 | 10/10 | 100 % |
| held_out_table_color (4) | 10/10 | 10/10 | 10/10 | 100 % |
| all_held_out | 10/10 | 10/10 | 10/10 | 100 % |

These tiers test the *expert*, which uses ground-truth poses, so colour tiers cannot affect it. They become meaningful when policies trained on this data are evaluated on the same tiers.

## 6. Dataset details

- Schema (handoff A3), unchanged from v1:
  - `action` and `observation.state` float32[16] (left j1–7, left gripper, right j1–7, right gripper; rad; gripper 1 = open);
  - three 224×224 AV1 videos;
  - `success` per frame, `task` = instruction;
  - 30 fps, `robot_type fr3_duo`;
  - std = 1 for constant columns (the idle left arm).
- Instructions: `seed % 3` → "Hand me the {object}", "Please pass the {object}", "Can you give me the {object}".
- Data quality (handoff A6):
  - no-op frames dropped: 455 → ~413 frames per episode on average. This removes the gripper-settling holds and most of the final hold;
  - success must hold for 15 frames in the sim;
  - every attempt is logged in `meta/episodes_extra.jsonl` (config, layout jitter, failure stage, engine, newton/mujoco_warp/warp/mujoco versions, git commit);
  - the last kept frame carries the episode outcome, because the frames that established the 15-frame hold are no-ops and were dropped.
- Stage predicates are geometric (`task_stages.py`), not contact lists, so both engines are judged by identical code. Grasp = commanded closed + fingers stopped short of fully closed + grasp point within 3 cm of the pinch point.
- Provenance:
  - first 96 episodes: generated from a working tree on `2370bbc` (logged as `git_dirty: true`), with code equal to `206c65e` except for the GPU image kernel (output identical to ±1 level);
  - episodes 97–384: `206c65e`.

## 7. Decisions and deviations (logged)

| What | Choice | Why |
|---|---|---|
| Arm | Right (A1). Left arm holds `HOME_Q` and is still recorded | Decision A1 |
| Pick region / tray | Mirror of v1: pick x 0.40–0.72, y −0.50 to −0.08; tray at (0.58, 0.26) | Reach probe: all points within 0.05 mm |
| Bottle size | Ø77 × 152 mm, 0.30 kg (A2). The team yaml says 250 mm tall | A2 is the decision |
| Bottle grasp | Top grasp across the body, pads 3 cm below the top | 100 % in both engines; keeps the wrist away from the table |
| Upright cylinders | Hidden 2 mm box foot | MuJoCo-Warp: one contact point for cylinder–box |
| World count | 16 for the 384 grid; 32 measured as the faster setting | 32 ≈ the GPU's compute limit; VRAM is not the limit |
| Extra objects (sheath etc.) | Not added | Optional in A2; the grid uses only the 3 team objects |
| Head-camera jitter | Not added | Optional in N3 |
| "More episodes" (N4 step 3) | Generated (seeds 1, 2 = resampled mass/friction/light and a ±1 cm layout shift; 768/768 succeeded) but **not published**: the dataset stays at the 384-episode grid | Owner's decision. `runner generate --seed 1/2` reproduces them |

## 8. Reproduce

```bash
source .venv-newton/bin/activate
python -m cathlab_sim.newton.check_n1                      # N1
python -m cathlab_sim.newton.expert_fr3 --probe            # reach probe
python -m cathlab_sim.newton.runner check                  # N2
python -m cathlab_sim.newton.randomize                     # N3
python -m cathlab_sim.newton.runner generate --out datasets/newton_raw/v1 --count 384 --worlds 32
python -m cathlab_sim.newton.report_n5                     # N5
source .venv/bin/activate
python -m cathlab_sim.newton.convert --raw datasets/newton_raw/v1 --out datasets/cathlab_fr3_newton_v1 --push
```
