"""Extract the FR3 Duo layout from the Siemens Isaac USD into isaac_reference.json.

Everything is expressed relative to the robot (not the room), so the MuJoCo scene
can copy it: arm bases and head camera in the Duo `mount_link` frame (z up, x forward,
y = robot's left), wrist cameras and gripper in each arm's `fr3v2_link8` (flange) frame.

Usage: python -m cathlab_sim.tools.extract_isaac_reference <Hackathon_FR3Duo_flattened.usd>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

ROBOT = "/World/FR3Duo_withV_Mkit__1_"
HEAD_CAM = "/mount_link/cover_link/head_link/zed_mini_camera_link/zed_mini_left_camera_frame/head_cam"
OBJECTS = {  # our name -> USD prim
    "box": "/World/WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01",
    "bottle": "/World/NaturalBostonRoundBottle_A02_PR_NVD_01",
    "pill_bottle": "/World/SM_PillBottle_01v",
    "serving_tray": "/World/ServingTray",
}
OUT = Path(__file__).resolve().parents[1] / "isaac_reference.json"


def main(usd_path):
    stage = Usd.Stage.Open(usd_path)
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    def world(path):
        """(position, 3x3 rotation whose columns are the local axes) in world."""
        t = Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
        return np.array(t.GetTranslation()), np.array(Gf.Matrix3d(t.GetRotation())).T

    def rel(child, parent):
        pc, rc = world(child)
        pp, rp = world(parent)
        return rp.T @ (pc - pp), rp.T @ rc

    def cam_entry(path, parent):
        pos, rot = rel(path, parent)
        cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
        f, ap = cam.GetFocalLengthAttr().Get(), cam.GetVerticalApertureAttr().Get()
        # USD and MuJoCo cameras both look along local -Z with +Y up, so the
        # camera's x/y axes in the parent frame are exactly MuJoCo's xyaxes.
        return {"parent": parent.split("/")[-1], "pos": pos.round(5).tolist(),
                "xyaxes": np.concatenate([rot[:, 0], rot[:, 1]]).round(5).tolist(),
                "fovy_deg": round(math.degrees(2 * math.atan(ap / 2 / f)), 3)}

    mount = ROBOT + "/mount_link"
    pm, rm = world(mount)
    ref = {"source": str(usd_path), "frames": "mount_link: z up, x forward, y robot-left; link8: arm flange"}

    ref["arm_bases"] = {}
    for side in ("left", "right"):
        pos, rot = rel(f"{ROBOT}/{side}_fr3v2_link0", mount)
        ref["arm_bases"][side] = {"pos": pos.round(5).tolist(), "xyaxes": np.concatenate([rot[:, 0], rot[:, 1]]).round(5).tolist()}

    ref["cameras"] = {"external_camera": cam_entry(ROBOT + HEAD_CAM, mount)}
    for side in ("left", "right"):
        ref["cameras"][f"{side}_wrist_camera"] = cam_entry(f"{ROBOT}/{side}_d405_color_frame/{side}_wrist_cam",
                                                           f"{ROBOT}/{side}_fr3v2_link8")

    # Gripper: fingertip separation axis and TCP offset in the flange frame
    l8 = f"{ROBOT}/left_fr3v2_link8"
    tip_l, _ = rel(f"{ROBOT}/left_robotiq_85_left_finger_tip_link", l8)
    tip_r, _ = rel(f"{ROBOT}/left_robotiq_85_right_finger_tip_link", l8)
    tcp, _ = rel(f"{ROBOT}/left_tcp", l8)
    ref["gripper"] = {"left_tip_in_link8": tip_l.round(5).tolist(), "right_tip_in_link8": tip_r.round(5).tolist(),
                      "tcp_in_link8": tcp.round(5).tolist()}

    # Objects: oriented bounding box centre/extent in the mount frame
    ref["objects"] = {}
    for name, path in OBJECTS.items():
        b = bbox.ComputeWorldBound(stage.GetPrimAtPath(path))
        rng, mat = b.GetRange(), b.GetMatrix()
        centre_w = np.array(mat.Transform((rng.GetMin() + rng.GetMax()) / 2))
        r_obj = np.array(Gf.Matrix3d(Gf.Transform(mat).GetRotation())).T
        scale = np.array(Gf.Transform(mat).GetScale())
        local_size = (np.array(rng.GetMax()) - np.array(rng.GetMin())) * scale
        size_mount = np.abs(rm.T @ r_obj) @ local_size
        centre = rm.T @ (centre_w - pm)
        ref["objects"][name] = {"centre": centre.round(4).tolist(), "size": size_mount.round(4).tolist(),
                                "bottom_z": round(float(centre[2] - size_mount[2] / 2), 4)}

    floor = bbox.ComputeWorldBound(stage.GetPrimAtPath("/World/Floor")).ComputeAlignedRange()
    ref["mount_height_above_floor"] = round(float(pm[1] - floor.GetMax()[1]), 4)  # stage is Y-up

    OUT.write_text(json.dumps(ref, indent=2))
    print(json.dumps(ref, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
