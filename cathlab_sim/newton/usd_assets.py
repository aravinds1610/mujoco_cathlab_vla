"""Convert textured USD props (NVIDIA Isaac for Healthcare / Lightwheel, OmniPBR materials) to MuJoCo meshes.

One OBJ per material (with its base-colour PNG), in metres, Z-up, centred in xy on the asset's bounding box
and resting on z = 0. Normal/roughness maps are dropped (MuJoCo uses only the colour texture).
Runs where usd-core is installed (.venv):

    python -m cathlab_sim.newton.usd_assets external/i4h_assets/Lightwheel/Assets/SurgicalTweezers/AngledTweezers001.usd out/tweezers

Lightwheel assets are CC BY-NC 4.0 (see the LICENSE.txt next to each asset).
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _material_info(prim, usd_dir: Path) -> tuple[str, Path | None, list[float]]:
    from pxr import Sdf, UsdShade
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return "default", None, [0.8, 0.8, 0.8]
    for p in mat.GetPrim().GetChildren():
        sh = UsdShade.Shader(p)
        if not sh:
            continue
        ins = {i.GetBaseName(): i.Get() for i in sh.GetInputs()}
        tex = ins.get("diffuse_texture") or ins.get("file")
        col = ins.get("diffuse_color_constant") or ins.get("diffuseColor")
        tex_path = (usd_dir / tex.path).resolve() if isinstance(tex, Sdf.AssetPath) and tex.path else None
        rgb = [float(c) for c in col] if col is not None else [0.8, 0.8, 0.8]
        return mat.GetPrim().GetName(), tex_path if tex_path and tex_path.exists() else None, rgb
    return mat.GetPrim().GetName(), None, [0.8, 0.8, 0.8]


def convert(usd_file: str | Path, out_dir: str | Path) -> dict:
    """Write <material>.obj (+ .png) files to out_dir; return a manifest (also saved as manifest.json)."""
    import trimesh
    from pxr import Usd, UsdGeom
    usd_file, out = Path(usd_file), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = Usd.Stage.Open(str(usd_file))
    mpu = UsdGeom.GetStageMetersPerUnit(st)
    groups = defaultdict(lambda: {"v": [], "uv": [], "tex": None, "rgb": None})
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        m = UsdGeom.Mesh(prim)
        pts, counts, idx = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        p = (np.c_[np.array(pts, float), np.ones(len(pts))] @ xf.T)[:, :3] * mpu
        idx, counts = np.array(idx), np.array(counts)
        st_pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        uv_vals = np.array(st_pv.Get(), float) if st_pv and st_pv.Get() is not None else None
        uv_idx = np.array(st_pv.GetIndices()) if uv_vals is not None and st_pv.GetIndices() else None
        interp = st_pv.GetInterpolation() if uv_vals is not None else None
        # per face-corner uv
        if uv_vals is None:
            corner_uv = np.zeros((len(idx), 2))
        elif interp == "faceVarying":
            corner_uv = uv_vals[uv_idx] if uv_idx is not None else uv_vals
        else:  # vertex / varying
            corner_uv = (uv_vals[uv_idx] if uv_idx is not None else uv_vals)[idx]
        # fan-triangulate polygons, expanding to one vertex per corner (keeps uv seams)
        name, tex, rgb = _material_info(prim, usd_file.parent)
        g = groups[name]
        g["tex"], g["rgb"] = tex, rgb
        start = np.r_[0, np.cumsum(counts)[:-1]]
        for s, c in zip(start, counts):
            for k in range(1, c - 1):
                for corner in (s, s + k, s + k + 1):
                    g["v"].append(p[idx[corner]])
                    g["uv"].append(corner_uv[corner])
    allv = np.concatenate([np.array(g["v"]) for g in groups.values()])
    lo, hi = allv.min(0), allv.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    manifest = {"source": str(usd_file), "size_m": (hi - lo).round(4).tolist(), "parts": []}
    for i, (name, g) in enumerate(groups.items()):
        v = np.array(g["v"]) + shift
        faces = np.arange(len(v)).reshape(-1, 3)
        part = f"part{i:02d}"
        visual = trimesh.visual.TextureVisuals(uv=np.array(g["uv"])) if g["tex"] else None
        trimesh.Trimesh(v, faces, visual=visual, process=False).export(out / f"{part}.obj", include_texture=False)
        tex_name = None
        if g["tex"]:
            tex_name = f"{part}.png"
            shutil.copy(g["tex"], out / tex_name)
        manifest["parts"].append({"obj": f"{part}.obj", "texture": tex_name, "rgb": g["rgb"], "material": name,
                                  "triangles": len(faces)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    m = convert(sys.argv[1], sys.argv[2])
    print(f"{Path(sys.argv[1]).name}: size {m['size_m']} m, {len(m['parts'])} parts, "
          f"{sum(p['triangles'] for p in m['parts'])} triangles, textures {sum(bool(p['texture']) for p in m['parts'])}")
