"""Evaluate a PolicyAdapter on seeded episodes in MujocoBackend.

    python -m cathlab_sim.evaluate --policy expert --n 60 --seed0 10000 --out report.json
    python -m cathlab_sim.evaluate --policy smolvla --ckpt outputs/<run>/checkpoints/last/pretrained_model --n 60

Reports success rate with a 95% Wilson confidence interval, failure stage counts,
per-object results, collisions, and policy latency. Use seeds disjoint from the
training seeds (generate.py starts at 0) so the score measures held-out layouts.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from cathlab_sim.contracts import STAGES, PolicyAdapter
from cathlab_sim.generate import DEFAULT_TASK, load_task
from cathlab_sim.spec_io import load_scene_spec

MAX_EPISODE_S = 30.0


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(policy: PolicyAdapter, backend, scene, task, seeds, objects, quiet=False) -> dict:
    rng = np.random.default_rng(seeds[0] if seeds else 0)
    episodes, latencies = [], []
    for i, seed in enumerate(seeds):
        target = objects[i % len(objects)]
        instruction = str(rng.choice(task.instructions(target)))
        obs = backend.reset(scene, seed, target)
        if hasattr(policy, "reset"):
            policy.reset()
        info, collided, steps = None, False, 0
        max_steps = int(MAX_EPISODE_S * policy.meta().control_hz)
        while steps < max_steps and not getattr(policy, "done", False):
            t = time.perf_counter()
            chunk = policy.predict(obs, instruction)
            latencies.append((time.perf_counter() - t) * 1000)
            for a in chunk:
                obs, info = backend.step(a)
                collided |= info.collision
                steps += 1
                if info.success:
                    break
            if info.success:
                break
        failed = next((s for s in STAGES if not info.stage_flags[s]), None)
        episodes.append({"seed": seed, "target": target, "instruction": instruction, "success": info.success,
                         "failure_stage": failed, "collision": collided, "steps": steps})
        if not quiet:
            print(f"seed {seed:6d} {target:10s} {'OK' if info.success else 'FAIL at ' + failed}", flush=True)

    wins = sum(e["success"] for e in episodes)
    lo, hi = wilson_ci(wins, len(episodes))
    return {
        "policy": policy.meta().model_dump(), "scene_id": scene.scene_id, "task_id": task.task_id,
        "n": len(episodes), "successes": wins, "success_rate": wins / max(len(episodes), 1),
        "ci95": [lo, hi],
        "failure_stages": {s: sum(e["failure_stage"] == s for e in episodes) for s in STAGES},
        "per_object": {o: {"successes": sum(e["success"] for e in episodes if e["target"] == o),
                           "n": sum(e["target"] == o for e in episodes)} for o in objects},
        "collision_episodes": sum(e["collision"] for e in episodes),
        "latency_ms": {"mean": float(np.mean(latencies)), "p95": float(np.percentile(latencies, 95))},
        "seeds": [seeds[0], seeds[-1]], "episodes": episodes,
    }


def make_policy(name: str, backend, ckpt=None) -> PolicyAdapter:
    if name == "smolvla":
        if ckpt is None:
            raise SystemExit("--policy smolvla needs --ckpt <run>/checkpoints/last/pretrained_model")
        from cathlab_sim.smolvla_policy import SmolVLAPolicyAdapter
        return SmolVLAPolicyAdapter(ckpt)
    from cathlab_sim.expert_policy import ScriptedExpertPolicy
    return ScriptedExpertPolicy(backend)


def main():
    from cathlab_sim.backend import MujocoBackend

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["expert", "smolvla"], default="expert")
    ap.add_argument("--ckpt", default=None, help="SmolVLA checkpoint folder (pretrained_model)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=10000, help="first seed (keep disjoint from training seeds)")
    ap.add_argument("--objects", nargs="+", default=None)
    ap.add_argument("--render", action="store_true", help="render camera images (needed by learned policies)")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args()

    scene, task = load_scene_spec(), load_task(DEFAULT_TASK)
    objects = args.objects or [o for o in task.objects if o in scene.objects]
    backend = MujocoBackend(render_images=args.render or args.policy != "expert")  # learned policies need images
    policy = make_policy(args.policy, backend, args.ckpt)
    report = evaluate(policy, backend, scene, task, list(range(args.seed0, args.seed0 + args.n)), objects)
    backend.close()
    lo, hi = report["ci95"]
    print(f"\n{report['policy']['name']}: {report['successes']}/{report['n']} = {report['success_rate']:.1%} "
          f"(95% CI {lo:.1%}-{hi:.1%})")
    print("failure stages:", {k: v for k, v in report["failure_stages"].items() if v})
    print("per object:", {o: f"{r['successes']}/{r['n']}" for o, r in report["per_object"].items()})
    print(f"collision episodes: {report['collision_episodes']}, latency mean {report['latency_ms']['mean']:.1f} ms")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
