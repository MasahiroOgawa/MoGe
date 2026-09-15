#!/usr/bin/env python3
"""Run MoGe-3 over a directory of pre-dumped clips and save metric depth per frame.

Written for cross-method comparison: a consumer dumps byte-identical clips once, every
method infers over the same files, and nobody re-derives "the test set" from its own
loader with a subtly different resize, crop or ordering.

MoGe-3 is MONOCULAR -- one image in, one geometry out -- so each frame of a clip is
inferred independently. A consumer comparing against a multi-frame method must say so:
the multi-frame method is given strictly more information.

Input layout (produced by the consumer)::

    <in>/clips/<clip:06d>/<frame:02d>.png
    <in>/manifest.json

Output, mirroring it so a clip's predictions are found by path alone::

    <out>/<clip:06d>/<frame:02d>.npy      float32 (H, W) metric depth, 0 = invalid

Usage::

    uv run python scripts/infer_clip_dir.py --in <dir> --out <dir> --model Ruicheng/moge-3-vitl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="dumped clip dir")
    ap.add_argument("--out", required=True, help="where to write depth .npy")
    ap.add_argument("--model", default="Ruicheng/moge-3-vitl",
                    help="HF id, e.g. Ruicheng/moge-3-vitl or Ruicheng/moge-3-vitg")
    ap.add_argument("--refine-steps", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from moge.model.v3 import MoGeModel

    model = MoGeModel.from_pretrained(args.model).to(args.device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[moge] {args.model}: {n_par/1e6:.0f}M params on {args.device}", flush=True)

    inp, out = Path(args.inp), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted((inp / "clips").iterdir())
    (out / "model_info.json").write_text(json.dumps(
        {"model": args.model, "params": n_par, "refine_steps": args.refine_steps}, indent=1))

    t0, n_frames = time.time(), 0
    for ci, cdir in enumerate(clips):
        odir = out / cdir.name
        odir.mkdir(exist_ok=True)
        for png in sorted(cdir.glob("*.png")):
            dst = odir / f"{png.stem}.npy"
            if dst.exists():                       # resumable: skip completed frames
                continue
            arr = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1).to(args.device)
            with torch.no_grad():
                pred = model.infer(img, refine_steps=args.refine_steps)
            depth = pred["depth"].float()
            # `mask` marks where the geometry is defined; zero elsewhere so a consumer's
            # "valid = depth > 0" convention holds without it needing MoGe's mask too.
            if pred.get("mask") is not None:
                depth = torch.where(pred["mask"], depth, torch.zeros_like(depth))
            np.save(dst, torch.nan_to_num(depth, nan=0.0, posinf=0.0).cpu().numpy())
            n_frames += 1
        if (ci + 1) % 20 == 0:
            print(f"  {ci + 1}/{len(clips)} clips, {n_frames} frames, "
                  f"{time.time()-t0:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"[moge] {len(clips)} clips / {n_frames} frames in {dt:.0f}s "
          f"({n_frames/max(dt,1e-6):.2f} fps) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
