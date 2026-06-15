#!/usr/bin/env python
"""B2 experiment (generation step): feed CADGenBench drawings to the pretrained
cadrille (Qwen2-VL-2B) specialist model and dump the generated CadQuery code.

Motivation: cadrille was trained on clean 2x2-grid (268px) shaded multi-view
*renders* of 3D meshes. CADGenBench inputs are real A2 dimensioned engineering
drawings. This measures the distribution break ("specialist model collapses on
real drawings"). Execution/scoring of the produced code is a separate step.

Feed strategies (how the single drawing sheet is mapped to cadrille's image slot):
  whole268     : resize the entire sheet to 268x268 (matches training grid size), 1 frame
  whole_native : pass the sheet as-is; Qwen2-VL processor resizes via min/max_pixels

Run (GPU1 is usually free; GPU0 busy):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/ryotaro/github/cadrille \
    python scripts/b2_cadrille_generate.py --feed whole268 --n 5
"""
import os, glob, json, argparse, time
from PIL import Image

import torch
from transformers import AutoProcessor
from functools import partial

from cadrille import Cadrille, collate  # from PYTHONPATH=~/github/cadrille

DATA_GLOB = ("/home/ryotaro/.cache/huggingface/hub/"
             "datasets--HuggingAI4Engineering--cadgenbench-data/snapshots/*/")


def find_generation_fixtures():
    """Return sorted list of (fixture_id, input_png_path) for generation fixtures."""
    snaps = sorted(glob.glob(DATA_GLOB))
    assert snaps, f"no cadgenbench-data snapshot under {DATA_GLOB}"
    root = snaps[-1]
    out = []
    for d in sorted(os.listdir(root)):
        png = os.path.join(root, d, "input.png")
        if os.path.isfile(png):
            out.append((d, png))
    return out


def make_image(png_path, feed):
    img = Image.open(png_path).convert("RGB")
    if feed == "whole268":
        # letterbox to square (white pad) then resize to the training grid size (268)
        w, h = img.size
        s = max(w, h)
        sq = Image.new("RGB", (s, s), (255, 255, 255))
        sq.paste(img, ((s - w) // 2, (s - h) // 2))
        img = sq.resize((268, 268), Image.LANCZOS)
    elif feed == "whole_native":
        # keep aspect, but cap longest side at 1024 (full A2 ~6.5MP OOMs the ViT
        # full-attention on a 24GB card -> tried to alloc 66GiB). 1024 stays legible-ish.
        w, h = img.size
        s = max(w, h)
        if s > 1024:
            img = img.resize((round(w * 1024 / s), round(h * 1024 / s)), Image.LANCZOS)
    else:
        raise ValueError(feed)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", default="whole268", choices=["whole268", "whole_native"])
    ap.add_argument("--n", type=int, default=5, help="number of generation fixtures")
    ap.add_argument("--ids", type=str, default="", help="comma-separated fixture ids (overrides --n)")
    ap.add_argument("--checkpoint", default="maksimko123/cadrille")
    ap.add_argument("--out", default="/home/ryotaro/my_works/drawing2cad/experiments/b2_cadrille")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.feed)
    os.makedirs(out_dir, exist_ok=True)

    fixtures = find_generation_fixtures()
    if args.ids:
        want = set(args.ids.split(","))
        fixtures = [f for f in fixtures if f[0] in want]
    else:
        fixtures = fixtures[: args.n]
    print(f"[b2] {len(fixtures)} fixtures, feed={args.feed}: {[f[0] for f in fixtures]}")

    print("[b2] loading cadrille ...")
    model = Cadrille.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28, padding_side="left")

    summary = []
    for fid, png in fixtures:
        item = {"video": [make_image(png, args.feed)],
                "description": "Generate cadquery code",
                "file_name": fid}
        batch = collate([item], processor=processor, n_points=256, eval=True)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                point_clouds=batch["point_clouds"].to(model.device),
                is_pc=batch["is_pc"].to(model.device),
                is_img=batch["is_img"].to(model.device),
                pixel_values_videos=batch["pixel_values_videos"].to(model.device)
                    if batch.get("pixel_values_videos") is not None else None,
                video_grid_thw=batch["video_grid_thw"].to(model.device)
                    if batch.get("video_grid_thw") is not None else None,
                max_new_tokens=args.max_new_tokens)
        trimmed = gen[0][batch["input_ids"].shape[1]:]
        code = processor.decode(trimmed, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        dt = time.time() - t0
        code_path = os.path.join(out_dir, f"{fid}.py")
        with open(code_path, "w") as f:
            f.write(code)
        # cheap static sanity: does it parse, does it mention cadquery / a result var?
        parses = True
        try:
            compile(code, code_path, "exec")
        except Exception as e:
            parses = False
        rec = {"id": fid, "chars": len(code), "lines": code.count("\n") + 1,
               "parses": parses, "has_cadquery": ("cadquery" in code or "cq." in code),
               "secs": round(dt, 1)}
        summary.append(rec)
        print(f"[b2] {fid}: {rec}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[b2] wrote {len(summary)} codes + summary.json to {out_dir}")


if __name__ == "__main__":
    main()
