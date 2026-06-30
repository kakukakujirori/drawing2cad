#!/usr/bin/env python
"""Run a (trained or zero-shot) cadrille-2D model on drawings -> AMVDG graph.

Mirrors cadrille/test.py generation, but the target is our compact graph text; each
prediction is parsed back to a graph JSON so `score.py` can score it against the
renderer GT. Run from this directory (or add it to PYTHONPATH) for the sibling imports.

  PYTHONPATH=$CADRILLE_REPO:. python infer.py \
    --data-root data_bundle --split-file val.jsonl \
    --checkpoint runs/2d_v0/final --out runs/2d_v0/pred
"""
import os, json, argparse
from functools import partial

import torch
from transformers import AutoProcessor
from torch.utils.data import DataLoader

from cadrille import Cadrille, collate
from dataset import DrawingGraphDataset
from serialize import graph_from_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--split-file', default='val.jsonl')
    ap.add_argument('--checkpoint', default='maksimko123/cadrille')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-new-tokens', type=int, default=6144)
    ap.add_argument('--max-pixels', type=int, default=1280 * 28 * 28)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--attn', default='sdpa', choices=['sdpa', 'flash_attention_2', 'eager'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = Cadrille.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn, device_map='auto')
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct',
        min_pixels=256 * 28 * 28, max_pixels=args.max_pixels, padding_side='left')

    ds = DrawingGraphDataset(args.data_root, args.split_file, split='test')
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=4,
                    collate_fn=partial(collate, processor=processor, n_points=256, eval=True))

    n_ok = 0
    for batch in dl:
        gen = model.generate(
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device),
            point_clouds=batch['point_clouds'].to(model.device),
            is_pc=batch['is_pc'].to(model.device),
            is_img=batch['is_img'].to(model.device),
            pixel_values_videos=batch['pixel_values_videos'].to(model.device) if batch.get('pixel_values_videos') is not None else None,
            video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get('video_grid_thw') is not None else None,
            max_new_tokens=args.max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(batch['input_ids'], gen)]
        texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for stem, txt in zip(batch['file_name'], texts):
            with open(os.path.join(args.out, f'{stem}.txt'), 'w') as f:
                f.write(txt)
            try:
                graph = graph_from_text(txt)
                json.dump(graph, open(os.path.join(args.out, f'{stem}.pred.json'), 'w'), ensure_ascii=False)
                n_ok += 1
                status = 'ok'
            except Exception as e:
                status = f'PARSE_FAIL: {e}'
            print(f'[{stem}] {status} ({len(txt)} chars)')

    print(f'[infer_2d] parseable {n_ok}/{len(ds)} -> {args.out}')


if __name__ == '__main__':
    main()
