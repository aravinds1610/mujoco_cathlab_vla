"""Damped least-squares IK for one FR3 arm's TCP site (position + orientation)."""
import mujoco
import numpy as np


def solve_ik(model, data, side, target_pos, target_quat, q_init=None, iters=300, damping=1e-2):
    """Return (q[7], pos_err_m, rot_err_rad). Works on a scratch copy of `data`."""
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    jnt = [model.joint(f"{side}_fr3v2_joint{i}") for i in range(1, 8)]
    qadr = np.array([j.qposadr[0] for j in jnt])
    dadr = np.array([j.dofadr[0] for j in jnt])
    lo, hi = np.array([j.range for j in jnt]).T
    if q_init is not None:
        d.qpos[qadr] = q_init
    site = model.site(f"{side}_tcp").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    cur_quat, err_rot = np.zeros(4), np.zeros(3)

    for _ in range(iters):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)  # mj_jacSite needs cdof from here; without it J == 0
        err_pos = target_pos - d.site_xpos[site]
        mujoco.mju_mat2Quat(cur_quat, d.site_xmat[site])
        mujoco.mju_subQuat(err_rot, target_quat, cur_quat)
        # subQuat gives the error in the TCP's local frame; jacr is in world frame
        err = np.concatenate([err_pos, d.site_xmat[site].reshape(3, 3) @ err_rot])
        if np.linalg.norm(err_pos) < 1e-4 and np.linalg.norm(err_rot) < 1e-3:
            break
        mujoco.mj_jacSite(model, d, jacp, jacr, site)
        J = np.vstack([jacp, jacr])[:, dadr]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + dq, lo, hi)

    mujoco.mj_kinematics(model, d)
    mujoco.mju_mat2Quat(cur_quat, d.site_xmat[site])
    mujoco.mju_subQuat(err_rot, target_quat, cur_quat)
    return d.qpos[qadr].copy(), np.linalg.norm(target_pos - d.site_xpos[site]), np.linalg.norm(err_rot)
