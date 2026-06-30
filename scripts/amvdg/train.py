#!/usr/bin/env python
"""Fine-tune cadrille for the 2D leg (drawing image -> AMVDG graph text).

Imports cadrille's own Cadrille model + collate unchanged; only the dataset and the
generation target differ. Init from the cadrille SFT checkpoint (so we inherit its
engineering-image + structured-code priors) or from base Qwen2-VL-2B for an ablation.

Requires CADRILLE_REPO on PYTHONPATH (the col14m/cadrille checkout) for
`from cadrille import Cadrille, collate`. Img-only path needs no pytorch3d/open3d.
Run from this directory (or add it to PYTHONPATH) so the sibling `dataset`/`serialize`
modules resolve; `run_train2d.sh` wraps build-bundle -> train -> infer end-to-end.

Example (drawing2cad-ml env):
  PYTHONPATH=$CADRILLE_REPO:. python train.py \
    --data-root data_bundle --out runs/2d_v0 --init cadrille --max-steps 400
"""
import os, argparse
from functools import partial

import torch
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

from cadrille import Cadrille, collate          # from $CADRILLE_REPO
from dataset import DrawingGraphDataset


class LogCb(TrainerCallback):
    def on_log(self, args, state, control, logs, **kw):
        if state.is_world_process_zero:
            os.makedirs(args.logging_dir, exist_ok=True)
            with open(os.path.join(args.logging_dir, 'log.txt'), 'a') as f:
                f.write(str(logs) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', choices=['cadrille', 'base'], default='cadrille')
    ap.add_argument('--cadrille-weights', default='maksimko123/cadrille')
    ap.add_argument('--base-weights', default='Qwen/Qwen2-VL-2B-Instruct')
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--max-pixels', type=int, default=1280 * 28 * 28)
    ap.add_argument('--save-steps', type=int, default=200)
    ap.add_argument('--attn', default='sdpa', choices=['sdpa', 'flash_attention_2', 'eager'])
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.base_weights,
        min_pixels=256 * 28 * 28, max_pixels=args.max_pixels, padding_side='left')

    weights = args.cadrille_weights if args.init == 'cadrille' else args.base_weights
    model = Cadrille.from_pretrained(
        weights, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_ds = DrawingGraphDataset(args.data_root, 'train.jsonl', split='train')
    val_ds = DrawingGraphDataset(args.data_root, 'val.jsonl', split='val')
    print(f'[train_2d] init={args.init} weights={weights} '
          f'train={len(train_ds)} val={len(val_ds)}')

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            logging_dir=os.path.join(args.out, 'logs'),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accum,
            dataloader_num_workers=4,
            max_steps=args.max_steps,
            lr_scheduler_type='cosine',
            learning_rate=args.lr,
            warmup_ratio=0.05,
            weight_decay=0.01,
            bf16=True,
            remove_unused_columns=False,
            logging_steps=10,
            save_total_limit=2,
            save_strategy='steps',
            save_steps=args.save_steps,
            report_to=None),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate, processor=processor, n_points=256),
        tokenizer=processor,
        callbacks=[LogCb()])
    trainer.train()
    trainer.save_model(os.path.join(args.out, 'final'))
    processor.save_pretrained(os.path.join(args.out, 'final'))
    print('[train_2d] done ->', os.path.join(args.out, 'final'))


if __name__ == '__main__':
    main()
