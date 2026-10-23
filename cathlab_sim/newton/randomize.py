"""N3: per-world domain randomisation preview and check.

Where each EpisodeConfig value is applied (both engines read task_scene.episode_params):
| Value                  | Newton (scene.NewtonTaskSim.reset)                 | CPU MuJoCo (cpu_sim)  |
|------------------------|----------------------------------------------------|-----------------------|
| layout poses (+jitter) | free-joint joint_q per world                        | qpos                  |
| object palette, table  | model.shape_color per shape (per world)            | geom_rgba             |
| mass x scale           | body_mass / inertia (+ inverses), notify_model_changed | body_mass / inertia |
| friction               | shape_material_mu of the object shapes             | geom_friction[:, 0]   |
| light intensity, K     | linear HDR image x gain, then sRGB (no light colour in the tiled renderer) | light_diffuse |

Per-world colours work in the tiled renderer, so no batching by (palette, table colour) is needed.

    python -m cathlab_sim.newton.randomize     # -> reports/newton_n3_preview.png, newton_n3_world0.png
"""
import json

import numpy as np
from PIL import Image

from cathlab_sim.newton.runner import configs
from cathlab_sim.newton.scene import NewtonTaskSim
from cathlab_sim.scene import CAMERAS
from cathlab_sim.spec_io import REPO


def main(n: int = 8):
    sim = NewtonTaskSim(n)
    cfgs = configs(0)[::47][:n]  # spread over the grid: different layouts, palettes, tables, lights
    sim.reset(cfgs)
    imgs = sim.images()
    ext = imgs["external_camera"]
    grid = np.concatenate([np.concatenate(list(ext[r * 4:(r + 1) * 4]), axis=1) for r in range(n // 4)], axis=0)
    Image.fromarray(grid).save(REPO / "reports/newton_n3_preview.png")
    Image.fromarray(np.concatenate([imgs[c][0] for c in CAMERAS], axis=1)).save(REPO / "reports/newton_n3_world0.png")
    mass = sim.model.body_mass.numpy()
    mu = sim.model.shape_material_mu.numpy()
    rows = []
    for w, c in enumerate(cfgs):
        b = sim.obj_body[c.object][w]
        s = sim.obj_shapes[c.object][w][0]
        rows.append({"world": w, "object": c.object, "layout": c.layout_id, "palette": c.obj_color_id,
                     "table": c.table_color_id, "mass_scale": round(c.mass_scale, 3),
                     "mass_applied": round(float(mass[b] / sim.base_mass[b]), 3),
                     "friction": round(c.friction, 3), "friction_applied": round(float(mu[s]), 3),
                     "light": round(c.light_intensity, 2), "temp_k": round(c.light_temp_k)})
    ok = all(abs(r["mass_applied"] - r["mass_scale"]) < 1e-3 and abs(r["friction_applied"] - r["friction"]) < 1e-3
             for r in rows)
    print(json.dumps(rows, indent=0).replace("\n", " "))
    print("applied values match configs:", ok)


if __name__ == "__main__":
    main()
