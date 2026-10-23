"""Plan on CPU MuJoCo (process pool), execute in Newton (N worlds per batch), judge with task_stages.

    python -m cathlab_sim.newton.runner check  [--per-object 20] [--worlds 16]   # N2 expert check
    python -m cathlab_sim.newton.runner generate --out <raw dir> [--start 0] [--count 96] [--seed 0]
    python -m cathlab_sim.newton.runner generate --engine mujoco ...             # CPU fallback / baseline

`generate` writes one raw .npz per successful episode (state, action, external_camera,
left_wrist_camera, right_wrist_camera, success, task) and appends every attempt to
<out>/meta/episodes_extra.jsonl. Configs come from the team's generation_order over the train grid
(seed 0 = the 384-cell grid; seeds 1, 2 = resampled mass/friction/light + 1 cm pose jitter).
"""
import argparse
import json
import multiprocessing as mp
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cathlab_sim.newton.task_stages import StageTracker
from cathlab_sim.scene import CAMERAS
from cathlab_sim.spec_io import REPO

INSTRUCTIONS = ("Hand me the {object}", "Please pass the {object}", "Can you give me the {object}")
NOOP_JOINT = 1e-4  # rad (handoff A6)
POSE_JITTER = 0.01  # m, for the extra seeds
NEAR_XY, NEAR_Z = 0.08, (0.06, 0.18)  # near-start episodes: TCP offset from the grasp point (m)
DROP_SETTLE = 45      # frames (1.5 s) the opened gripper holds still while a dropped object comes to rest
DROP_MAX_TILT = 15.0  # deg: a dropped object's long axis must be this close to horizontal (any roll about it is
                      # fine: the trocar rests rolled ~32 deg, the puncture device ~15 deg, even when placed by hand)
DROP_TRAY_MARGIN = 0.03  # m: drops that land on or this close to the tray are rejected
DROP_IN_LIFT = 0.4      # share of drops during the (slow, vertical) lift; the rest during the carry to the tray


def make_task(name: str = "v1"):
    """v1 = right arm, bottle/vial/syringe primitives; v2 = both arms, surgical instrument meshes."""
    if name == "v2":
        from cathlab_sim.newton.instruments import V2Task
        return V2Task()
    from cathlab_sim.newton.task_scene import V1Task
    return V1Task()


def instruction(cfg, task=None) -> str:
    return (task or make_task("v1")).instruction(cfg)


def configs(seed: int = 0, variation=None, task=None):
    """Train-grid configs in generation order. seed > 0 resamples the continuous values."""
    if task is None:
        from cathlab_sim.newton.task_scene import V1Task
        task = V1Task(variation)
    return task.configs(seed)


def versions() -> dict:
    import mujoco
    out = {"mujoco": mujoco.__version__}
    try:
        import mujoco_warp
        import newton
        import warp
        out.update(newton=newton.__version__, warp=warp.__version__,
                   mujoco_warp=getattr(mujoco_warp, "__version__", "3.12.0"))
    except ImportError:
        pass
    out["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                       text=True).stdout.strip()
    out["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "cathlab_sim", "configs"], cwd=REPO,
                                           capture_output=True, text=True).stdout.strip())
    return out


# -- CPU side (worker processes) ----------------------------------------------
_cpu = None


def _init_worker(render: bool = False, task_name: str = "v1"):
    global _cpu
    from cathlab_sim.newton.cpu_sim import CpuTaskSim
    _cpu = CpuTaskSim(render=render, task=make_task(task_name))


def _sample_near_q(cfg, jitter, tries: int = 20):
    """A pose of the acting arm with the TCP pointing down somewhere around the target: +-NEAR_XY sideways,
    NEAR_Z above the grasp height, random yaw. Rejected if IK misses or the settle disturbs an object."""
    from cathlab_sim.expert import PAD_CENTRE_CLOSED, TABLE_CLEARANCE, TIP_CLOSED, _yaw_quat
    from cathlab_sim.ik import solve_ik
    from cathlab_sim.newton.task_scene import TABLE_TOP
    from cathlab_sim.scene import HOME_Q
    rng = np.random.default_rng(cfg.seed + 7_000_003)
    _cpu.reset(cfg, jitter=jitter)
    obs = _cpu.task_obs()
    g = obs["grasp_pos"][0]
    rest = {n: p.copy() for n, (p, _) in obs["objects"].items()}
    grasp_z = max(g[2] + PAD_CENTRE_CLOSED, TABLE_TOP + TIP_CLOSED + TABLE_CLEARANCE)
    for _ in range(tries):
        target = np.array([g[0] + rng.uniform(-NEAR_XY, NEAR_XY), g[1] + rng.uniform(-NEAR_XY, NEAR_XY),
                           grasp_z + rng.uniform(*NEAR_Z)])
        q, pe, re = solve_ik(_cpu.model, _cpu.data, _cpu.side, target, _yaw_quat(rng.uniform(-np.pi, np.pi)),
                             q_init=np.array(HOME_Q))
        if pe > 0.005 or re > 0.03:
            continue
        _cpu.reset(cfg, jitter=jitter, arm_q=q)
        moved = max(np.linalg.norm(_cpu.task_obs()["objects"][n][0] - rest[n]) for n in _cpu.task.objects)
        if moved < 0.003:
            return q
    return None


def _sample_drop_start(cfg, jitter, tries: int = 4):
    """Drop-recovery start state (v2): the expert picks the target, the gripper opens at a random frame between
    the lift and the lowering into the tray, and the instrument falls and settles. Not recorded; the episode
    starts from the result. Returns {"poses", "side", "arm_q", ...} or None if no usable drop was found.
    Rejected: the object was not lifted, lands on or next to the tray, crosses to the other arm's side,
    rests with its long axis tilted (> DROP_MAX_TILT, e.g. propped on another object) or off the table, or moves when the scene is rebuilt from the state."""
    from cathlab_sim.newton.cpu_sim import ARM, GRIP
    from cathlab_sim.newton.expert_fr3 import TaskExpert
    from cathlab_sim.newton.task_scene import TABLE_TOP
    task = _cpu.task
    rng = np.random.default_rng(cfg.seed + 9_000_011)
    tx, ty = task.tray_center
    hx, hy = (h + DROP_TRAY_MARGIN for h in task.tray_outer_half)
    for _ in range(tries):
        task.start_override.pop(cfg.seed, None)
        _cpu.reset(cfg, jitter=jitter)
        side = _cpu.side
        ex = TaskExpert(_cpu.model, task, side)
        actions = ex.plan(_cpu.data, _cpu.task_obs(), cfg.object)
        # a slip just after the grasp, or a drop on the way to the tray (lands between the layout spot and the tray)
        lo, hi = (ex.marks["lift"] + 8, ex.marks["carry"]) if rng.random() < DROP_IN_LIFT else \
            (ex.marks["carry"], ex.marks["lower"])
        t_drop = int(rng.integers(lo, hi))
        _cpu.reset(cfg, jitter=jitter)
        for a in actions[:t_drop]:
            _cpu.step(a)
        height = float(_cpu.task_obs()["target_pos"][0][2] - TABLE_TOP)
        if height < 0.02:
            continue  # not actually holding it
        hold = actions[t_drop - 1].copy()
        hold[GRIP[side]] = 1.0
        for _ in range(DROP_SETTLE):
            _cpu.step(hold)
        obs = _cpu.task_obs()
        pos, quat = obs["objects"][cfg.object]
        R = np.zeros(9)
        import mujoco
        mujoco.mju_quat2Mat(R, quat)
        tilt = float(np.degrees(np.arcsin(min(abs(R[6]), 1.0))))  # long (x) axis vs the table plane
        on_tray = abs(pos[0] - tx) < hx and abs(pos[1] - ty) < hy
        same_side = (pos[1] > 0) == (side == "left")
        if on_tray or not same_side or tilt > DROP_MAX_TILT or pos[2] - TABLE_TOP > 0.01 or obs["target_speed"][0] > 0.01:
            continue
        state = {"poses": {n: (p.tolist(), q.tolist()) for n, (p, q) in obs["objects"].items()}, "side": side,
                 "arm_q": _cpu.data.qpos[[_cpu.model.joint(f"{side}_fr3v2_joint{i}").qposadr[0] for i in range(1, 8)]].tolist()}
        task.start_override[cfg.seed] = state
        _cpu.reset(cfg, jitter=jitter, arm_q=np.array(state["arm_q"]))
        moved = max(np.linalg.norm(_cpu.task_obs()["objects"][n][0] - np.array(p)) for n, (p, _) in state["poses"].items())
        if moved < 0.003:
            return {**state, "t_drop": t_drop, "drop_phase": "lift" if t_drop < ex.marks["carry"] else
                    f"carry {(t_drop - ex.marks['carry']) / (ex.marks['lower'] - ex.marks['carry']):.2f}",
                    "drop_height_m": round(height, 3), "long_axis_tilt_deg": round(tilt, 1)}
    task.start_override.pop(cfg.seed, None)
    return None


def plan_one(job: dict) -> dict:
    """Plan one episode on CPU; optionally execute it on CPU too (sim2sim, fallback recording).
    job["near"]: start the acting arm near the target instead of at HOME_Q (falls back to home if no pose fits)."""
    from cathlab_sim.newton.expert_fr3 import TaskExpert
    from cathlab_sim.newton.variation import EpisodeConfig
    cfg = EpisodeConfig(**job["cfg"])
    jitter = np.array(job.get("jitter", [0.0, 0.0]))
    drop = None
    if job.get("drop"):
        drop = _sample_drop_start(cfg, jitter)
        if drop is None:
            return {"actions": None, "arm_q": None, "side": None, "drop": None}
        arm_q = np.array(drop["arm_q"])  # start_override (poses, side) is now set for cfg.seed
    else:
        arm_q = _sample_near_q(cfg, jitter) if job.get("near") else None
    _cpu.reset(cfg, jitter=jitter, arm_q=arm_q)
    actions = TaskExpert(_cpu.model, _cpu.task, _cpu.side).plan(_cpu.data, _cpu.task_obs(), cfg.object)
    out = {"actions": actions, "arm_q": None if arm_q is None else arm_q.tolist(), "side": _cpu.side, "drop": drop}
    if job.get("run_cpu"):
        tracker = StageTracker(1)
        rec = {"state": [], "success": [], **{c: [] for c in CAMERAS}} if job.get("record") else None
        succ = []
        for a in actions:
            if rec is not None:
                rec["state"].append(_cpu.state())
                for c, img in _cpu.images().items():
                    rec[c].append(img)
            _cpu.step(a)
            succ.append(bool(tracker.update(_cpu.flags())[0]))
        out.update(cpu_success=succ[-1], cpu_failure_stage=None if succ[-1] else (tracker.failure_stage(0) or "hold"),
                   cpu_success_trace=succ, cpu_final_target_pos=_cpu.task_obs()["target_pos"][0].tolist())
        if rec is not None:
            out["record"] = {k: np.array(v) for k, v in rec.items() if k != "success"}
    return out


def make_pool(processes: int, render: bool = False, task_name: str = "v1"):
    return mp.get_context("spawn").Pool(processes, initializer=_init_worker, initargs=(render, task_name))


# -- Newton side ----------------------------------------------------------------
def run_newton_batch(sim, cfgs, plans, jitters=None, record: bool = False, arm_qs=None):
    """Execute up to sim.n planned episodes in parallel worlds. Returns per-episode dicts."""
    n, k = sim.n, len(cfgs)
    cfgs_p = list(cfgs) + [cfgs[0]] * (n - k)
    plans_p = list(plans) + [plans[0]] * (n - k)
    jit = list(jitters if jitters is not None else [np.zeros(2)] * k) + [np.zeros(2)] * (n - k)
    rq = list(arm_qs if arm_qs is not None else [None] * k) + [None] * (n - k)
    sim.reset(cfgs_p, jitters=jit, arm_qs=rq)
    T = np.array([len(p) for p in plans_p])
    tracker = StageTracker(n)
    final = [None] * n
    rec = [{"state": [], "action": [], "success": [], **{c: [] for c in CAMERAS}} for _ in range(k)] if record else None
    nan = np.zeros(n, bool)
    for t in range(T.max()):
        a = np.stack([p[min(t, len(p) - 1)] for p in plans_p])
        if record:
            state, imgs = sim.state(), sim.images()
            for w in range(k):
                if t < T[w]:
                    rec[w]["state"].append(state[w])
                    rec[w]["action"].append(a[w])
                    for c in CAMERAS:
                        rec[w][c].append(imgs[c][w])
        st = sim.step(a)
        nan |= np.isnan(st).any(axis=1)
        obs = sim.task_obs()
        succ = tracker.update(sim.flags(obs)) & ~nan
        for w in range(k):
            if t < T[w] and record:
                rec[w]["success"].append(bool(succ[w]))
            if t == T[w] - 1:
                final[w] = {"success": bool(succ[w]), "failure_stage": None if succ[w] else
                            ("nan" if nan[w] else (tracker.failure_stage(w) or "hold")), "n_frames": int(T[w]),
                            "final_target_pos": obs["target_pos"][w].tolist()}
    out = []
    for w in range(k):
        r = dict(final[w])
        if record:
            r["record"] = {key: np.array(v) for key, v in rec[w].items()}
        out.append(r)
    return out


def drop_noops(rec: dict, success: bool) -> dict:
    """Handoff A6: drop frames whose joint targets move < 1e-4 rad on all joints with an unchanged gripper
    command (vs the previous kept frame). The last kept frame carries the episode outcome."""
    act = rec["action"]
    keep = [0]
    for t in range(1, len(act)):
        d = np.abs(act[t] - act[keep[-1]])
        if (np.delete(d, [7, 15]) >= NOOP_JOINT).any() or d[[7, 15]].max() > 0:
            keep.append(t)
    keep = np.array(keep)
    out = {k: v[keep] for k, v in rec.items()}
    out["success"] = out["success"].copy()
    out["success"][-1] = success
    return out


def save_npz(path: Path, rec: dict, task: str):
    np.savez_compressed(path, state=rec["state"].astype(np.float32), action=rec["action"].astype(np.float32),
                        success=rec["success"].astype(bool), task=np.array(task),
                        **{c: rec[c].astype(np.uint8) for c in CAMERAS})


# -- commands ---------------------------------------------------------------------
def cmd_check(args):
    """N2: expert success per object on the first per_object train configs of each object, in Newton."""
    from cathlab_sim.newton.scene import NewtonTaskSim
    cfgs = configs(0)[:3 * args.per_object]
    pool = make_pool(args.procs)
    t0 = time.time()
    plans = pool.map(plan_one, [{"cfg": c.model_dump(), "run_cpu": True} for c in cfgs])
    pool.close()
    plan_s = time.time() - t0
    sim = NewtonTaskSim(args.worlds, render=False)
    t0 = time.time()
    res = []
    for i in range(0, len(cfgs), args.worlds):
        res += run_newton_batch(sim, cfgs[i:i + args.worlds], [p["actions"] for p in plans[i:i + args.worlds]])
    exec_s = time.time() - t0
    report = {"configs": len(cfgs), "plan_s_cpu_pool": round(plan_s, 1), "newton_exec_s": round(exec_s, 1),
              "worlds": args.worlds, "per_object": {}}
    for obj in ("bottle", "vial", "syringe"):
        idx = [i for i, c in enumerate(cfgs) if c.object == obj]
        fails, cpu_fails = {}, {}
        for i in idx:
            if not res[i]["success"]:
                fails[res[i]["failure_stage"]] = fails.get(res[i]["failure_stage"], 0) + 1
            if not plans[i]["cpu_success"]:
                cpu_fails[plans[i]["cpu_failure_stage"]] = cpu_fails.get(plans[i]["cpu_failure_stage"], 0) + 1
        report["per_object"][obj] = {
            "newton_success": f"{sum(res[i]['success'] for i in idx)}/{len(idx)}",
            "newton_rate": round(np.mean([res[i]["success"] for i in idx]), 3),
            "mean_steps": round(float(np.mean([res[i]["n_frames"] for i in idx])), 1),
            "newton_failure_stages": fails,
            "cpu_success": f"{sum(plans[i]['cpu_success'] for i in idx)}/{len(idx)}",
            "cpu_failure_stages": cpu_fails}
    report["pass"] = all(v["newton_rate"] >= 0.8 for v in report["per_object"].values())
    report["failed_configs"] = [{"i": i, "object": cfgs[i].object, "layout": cfgs[i].layout_id,
                                 "newton": res[i]["failure_stage"], "cpu": plans[i]["cpu_failure_stage"]}
                                for i in range(len(cfgs)) if not (res[i]["success"] and plans[i]["cpu_success"])]
    (REPO / "reports/newton_expert_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failed_configs"}, indent=1))


def _jobs(start: int, count: int, seed: int, task=None):
    cfgs = configs(seed, task=task)
    rng = np.random.default_rng(10_000 + seed)
    # drawn for the whole grid, then sliced: a run resumed with --start gets the same values as a full run
    jit = [rng.uniform(-POSE_JITTER, POSE_JITTER, 2) if seed > 0 else np.zeros(2) for _ in cfgs]
    return cfgs[start:start + count], jit[start:start + count]


def cmd_generate(args):
    out = Path(args.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    task = make_task(args.task)
    cfgs, jit = _jobs(args.start, args.count, args.seed, task)
    ver = versions()
    ver["engine"], ver["task"] = args.engine, task.name
    log = open(out / "meta/episodes_extra.jsonl", "a")
    writer = ThreadPoolExecutor(4)
    pool = make_pool(args.procs, render=args.engine == "mujoco", task_name=args.task)
    sim = None
    if args.engine == "newton":
        from cathlab_sim.newton.scene import NewtonTaskSim
        sim = NewtonTaskSim(args.worlds, task=task)
    t0, saved = time.time(), 0
    batch = args.worlds if args.engine == "newton" else args.procs
    try:
        for i in range(0, len(cfgs), batch):
            if args.max_saved and saved >= args.max_saved:
                break
            cb, jb = cfgs[i:i + batch], jit[i:i + batch]
            jobs = [{"cfg": c.model_dump(), "jitter": j.tolist(), "run_cpu": args.engine == "mujoco",
                     "record": args.engine == "mujoco", "near": args.near, "drop": args.drop} for c, j in zip(cb, jb)]
            plans = pool.map(plan_one, jobs)
            if args.drop:  # configs without a usable drop state are logged and skipped
                for c, j, p in zip(cb, jb, plans):
                    if p["actions"] is None:
                        log.write(json.dumps({"episode": None, **c.model_dump(), "start": "drop", "success": False,
                                              "failure_stage": "no_drop_start", "engine": args.engine, **ver}) + "\n")
                    else:
                        task.start_override[c.seed] = {"poses": p["drop"]["poses"], "side": p["drop"]["side"]}
                keep = [i for i, p in enumerate(plans) if p["actions"] is not None]
                cb, jb, plans = [cb[i] for i in keep], [jb[i] for i in keep], [plans[i] for i in keep]
                if not cb:
                    continue
            if args.engine == "newton":
                res = run_newton_batch(sim, cb, [p["actions"] for p in plans], jb, record=True,
                                       arm_qs=[p["arm_q"] for p in plans])
            else:
                res = []
                for p in plans:
                    r = p.pop("record")
                    r["action"] = p["actions"]
                    r["success"] = np.array(p["cpu_success_trace"])
                    res.append({"success": p["cpu_success"], "failure_stage": p["cpu_failure_stage"],
                                "n_frames": len(p["actions"]), "record": r})
            for cfg, j, r, p in zip(cb, jb, res, plans):
                start = "drop" if p["drop"] else ("near" if p["arm_q"] is not None else "home")
                name = f"{cfg.object}_l{cfg.layout_id}_o{cfg.obj_color_id}_t{cfg.table_color_id}_s{cfg.seed}" + \
                    ("" if start == "home" else f"_{start}")
                kept, over_cap = None, bool(args.max_saved and saved >= args.max_saved)
                if r["success"] and not over_cap:
                    rec = drop_noops(r["record"], True)
                    kept = len(rec["action"])
                    writer.submit(save_npz, out / "episodes" / f"{name}.npz", rec, task.instruction(cfg))
                    saved += 1
                log.write(json.dumps({"episode": name if kept else None, **cfg.model_dump(),
                                      "pose_jitter_m": [round(float(x), 4) for x in j],
                                      "start": start, "arm_q": p["arm_q"],
                                      **({"drop": {k: v for k, v in p["drop"].items() if k != "poses"},
                                          "start_poses": p["drop"]["poses"]} if p["drop"] else {}),
                                      "side": p["side"], "instruction": task.instruction(cfg), "success": r["success"],
                                      "failure_stage": r["failure_stage"], "over_cap": over_cap and r["success"],
                                      "n_frames_raw": r["n_frames"],
                                      "n_frames_kept": kept, "engine": args.engine, **ver}) + "\n")
            log.flush()
            done = min(i + batch, len(cfgs))
            el = time.time() - t0
            print(f"[{done}/{len(cfgs)}] saved {saved}  {el:.0f}s  {3600 * done / el:.0f} episodes/h", flush=True)
    finally:
        writer.shutdown(wait=True)
        pool.close()
        log.close()
    print(f"done: saved {saved}/{len(cfgs)} in {time.time() - t0:.0f}s -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--per-object", type=int, default=20)
    c.add_argument("--worlds", type=int, default=20)
    c.add_argument("--procs", type=int, default=16)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--count", type=int, default=96)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--worlds", type=int, default=16)
    g.add_argument("--procs", type=int, default=16)
    g.add_argument("--engine", choices=["newton", "mujoco"], default="newton")
    g.add_argument("--near", action="store_true", help="start the acting arm near the target (recovery data)")
    g.add_argument("--max-saved", type=int, default=None, help="stop after this many saved episodes")
    g.add_argument("--drop", action="store_true", help="v2: start from a dropped instrument (drop-recovery data)")
    g.add_argument("--task", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()
    {"check": cmd_check, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    main()
