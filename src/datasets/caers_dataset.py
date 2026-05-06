from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from datasets.transforms import default_transform
from utils.io_utils import read_jsonl


def _apply_face_mask(image: Image.Image, face_bbox: Any) -> Image.Image:
    if not face_bbox:
        return image.copy()

    if not isinstance(face_bbox, (list, tuple)) or len(face_bbox) != 4:
        return image.copy()

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle([x1, y1, x2, y2], fill=(127, 127, 127))
    return masked


class CAERSTwoStreamDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        split: str,
        image_size: int,
        transform: Any = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform if transform is not None else default_transform(image_size)

        rows = read_jsonl(manifest_path)
        self.rows = [r for r in rows if str(r.get("split")) == split]
        if len(self.rows) == 0:
            raise ValueError(f"No rows found for split={split} in manifest={manifest_path}")

        labels = sorted({str(r["label"]) for r in rows})
        self.label_to_index = {label: idx for idx, label in enumerate(labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rel_path = str(row["image_path"])
        image_path = self.dataset_root / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        image = Image.open(image_path).convert("RGB")
        context_image = _apply_face_mask(image, row.get("face_bbox"))

        face_tensor = self.transform(image)
        context_tensor = self.transform(context_image)

        label_name = str(row["label"])
        label_idx = self.label_to_index[label_name]

        return {
            "sample_id": str(row["sample_id"]),
            "face_image": face_tensor,
            "context_image": context_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_name": label_name,
            "image_path": rel_path,
        }
