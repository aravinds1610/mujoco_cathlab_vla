"""Convex collision pieces for converted mesh props (output of usd_assets.py), via CoACD.

    python -m cathlab_sim.newton.collision_hulls external/i4h_mujoco/scissors [--threshold 0.04] [--max-hulls 16]

Writes <dir>/collision/hullNN.obj in the same frame as the visual parts, and adds "collision" to manifest.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

MAX_HULL_VERTS = 64


def decompose(obj_dir: Path, threshold: float = 0.04, max_hulls: int = 16) -> list[str]:
    import coacd
    import trimesh
    man = json.loads((obj_dir / "manifest.json").read_text())
    parts = [trimesh.load(obj_dir / p["obj"], force="mesh", process=False) for p in man["parts"]]
    whole = trimesh.util.concatenate(parts)
    whole.merge_vertices()
    coacd.set_log_level("error")
    hulls = coacd.run_coacd(coacd.Mesh(whole.vertices, whole.faces), threshold=threshold, max_convex_hull=max_hulls,
                            preprocess_mode="auto", max_ch_vertex=MAX_HULL_VERTS)
    out = obj_dir / "collision"
    out.mkdir(exist_ok=True)
    for f in out.glob("hull*.obj"):
        f.unlink()
    names = []
    for i, (v, f) in enumerate(hulls):
        h = trimesh.Trimesh(np.asarray(v), np.asarray(f)).convex_hull
        name = f"collision/hull{i:02d}.obj"
        h.export(obj_dir / name)
        names.append(name)
    man["collision"] = names
    (obj_dir / "manifest.json").write_text(json.dumps(man, indent=2))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--threshold", type=float, default=0.04, help="CoACD concavity threshold (lower = finer)")
    ap.add_argument("--max-hulls", type=int, default=16)
    args = ap.parse_args()
    for d in args.dirs:
        names = decompose(d, args.threshold, args.max_hulls)
        print(f"{d.name:18s} {len(names)} convex pieces")


if __name__ == "__main__":
    main()
