"""Preview GIF from raw episodes: external + right wrist camera side by side, one episode per object.

    python -m cathlab_sim.newton.preview --raw datasets/newton_raw/v1 --out reports/newton_preview.gif
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

STRIDE = 3  # 30 fps -> 10 fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    log = [json.loads(line) for line in (args.raw / "meta/episodes_extra.jsonl").read_text().splitlines()]
    frames = []
    for obj in ("bottle", "vial", "syringe"):
        name = next(e["episode"] for e in log if e["episode"] and e["object"] == obj)
        with np.load(args.raw / "episodes" / f"{name}.npz") as z:
            ext, wrist = z["external_camera"], z["right_wrist_camera"]
        frames += [Image.fromarray(np.concatenate([ext[t], wrist[t]], axis=1)) for t in range(0, len(ext), STRIDE)]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=1000 // (30 // STRIDE), loop=0)
    print(f"{len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
