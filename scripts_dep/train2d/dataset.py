"""Dataset for cadrille 2D->2D: drawing image -> intra-view graph text.

Mirrors cadrille's image branch (returns a 'video' list of frames) but the frame
is our actual engineering drawing and the 'answer' is the compact intra-view graph
target instead of CadQuery code. Reuses cadrille.collate unchanged (is_pc=0).
"""

import os
import json
from PIL import Image, ImageOps
from torch.utils.data import Dataset

try:
    from serialize import PROMPT
except Exception:  # serialize.py sits next to this file
    PROMPT = "Extract the engineering drawing as an intra-view graph. Output compact JSON only."


class DrawingGraphDataset(Dataset):
    def __init__(self, data_root, split_file, max_side=1280, split="train"):
        self.data_root = data_root
        self.split = split
        self.max_side = max_side
        self.rows = [
            json.loads(l)
            for l in open(os.path.join(data_root, split_file))
            if l.strip()
        ]

    def __len__(self):
        return len(self.rows)

    def _load_img(self, rel):
        img = Image.open(os.path.join(self.data_root, rel)).convert("RGB")
        # cap the long side so a full A-size sheet fits the VL token budget while
        # keeping dimension text legible (Qwen processor does the 28px tiling).
        w, h = img.size
        s = max(w, h)
        if s > self.max_side:
            r = self.max_side / s
            img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
        return ImageOps.expand(img, border=3, fill="black")

    def __getitem__(self, i):
        r = self.rows[i]
        item = {
            "video": [self._load_img(r["image"])],
            "description": PROMPT,
            "file_name": r["file_name"],
        }
        if self.split in ("train", "val"):
            item["answer"] = r["target"]
        return item
