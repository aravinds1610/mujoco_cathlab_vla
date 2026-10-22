# AR2: FR3 Duo robot model in MuJoCo

## Model
- Arms: 2 × MuJoCo Menagerie `franka_fr3_v2` (same arm version as the Siemens Isaac Duo).
- Grippers: 2 × Menagerie `robotiq_2f85`, rotated +90° about the flange z axis so the fingers open along flange y, as on the Duo.
- Arm base poses (V-mount, bases 10 cm apart, ~52° outward tilt) and camera mounts are copied from the Siemens USD. They are extracted reproducibly by `python -m cathlab_sim.tools.extract_isaac_reference <Hackathon_FR3Duo_flattened.usd>` into `cathlab_sim/isaac_reference.json`.
- Simulation: 600 Hz physics, 30 Hz control, so each action is held for 20 physics steps.

## 16-D action / state mapping (`cathlab_sim/robot_io.py`)

| Index | Name | Action (dataset) | MuJoCo command | State (dataset) |
|---|---|---|---|---|
| 0–6 | `left_joint1..7` | absolute target, rad | position actuator `left_fr3v2_jointN`, ctrl = target | measured joint position, rad |
| 7 | `left_gripper` | 1 = open, 0 = closed | `left_gripper_fingers_actuator`, ctrl = (1 − g) × 255 | 1 − q_driver / 0.8 |
| 8–14 | `right_joint1..7` | as left | `right_fr3v2_jointN` | as left |
| 15 | `right_gripper` | as left | `right_gripper_fingers_actuator` | as left |

- Gripper driver joint range is 0–0.8 rad in both Menagerie and the Siemens `gripper_contract.closed_joint_rad`, so the action mapping is exact.
- The state mapping is an **assumption**. It is consistent with the dataset: the minimum gripper state is 0.419, against 1 − 0.459/0.8 = 0.426 at the recorded box-hold angle. The Siemens `gripper_joint_to_norm` source was not provided.
- Ready pose (from the Siemens doc): `[0, −45, 0, −135, 0, 90, 45]°` on both arms.

## Cameras (224 × 224)

| Name | Mounted on | Field of view (vertical) |
|---|---|---|
| `external_camera` | world, at the Duo head-camera pose | 90° |
| `left_wrist_camera` / `right_wrist_camera` | each arm's `fr3v2_link8` | 87.0° |

The fields of view follow the recorder (square 4.8 mm aperture), not the USD's authored apertures (58.7° / 56.5°).

## Checks
- `python -m cathlab_sim.check_scene`: 16 actuators; scene stable; arms hold the ready pose; fingers open along flange y; 3 cameras render; IK reaches both trays and the handover point with the gripper pointing down.
- `pytest -q tests` (CPU only): schema order, 30 Hz substeps, gripper command mapping, state readback, action→state tracking.
