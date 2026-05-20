from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from datasets.transforms import default_transform
from utils.io_utils import read_jsonl


def _apply_face_mask(image: Image.Image, face_bbox: Any, fill_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Mask face region in context image. Default fill is black (0,0,0) per paper."""
    if not face_bbox:
        return image.copy()

    if not isinstance(face_bbox, (list, tuple)) or len(face_bbox) != 4:
        return image.copy()

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle([x1, y1, x2, y2], fill=fill_color)
    return masked


class CAERSTwoStreamDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        split: str,
        image_size: int,
        face_size: int | None = None,
        transform: Any = None,
        face_transform: Any = None,
        context_transform: Any = None,
        synchronized_flip: bool = False,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.image_size = image_size
        self.face_size = face_size or image_size
        self.synchronized_flip = synchronized_flip

        # Fallback to unified transform if per-stream transforms not provided
        if face_transform is not None and context_transform is not None:
            self.face_transform = face_transform
            self.context_transform = context_transform
        else:
            t = transform if transform is not None else default_transform(image_size)
            self.face_transform = t
            self.context_transform = t

        rows = read_jsonl(manifest_path)
        if isinstance(split, (list, tuple)):
            self.rows = [r for r in rows if str(r.get("split")) in split]
        else:
            self.rows = [r for r in rows if str(r.get("split")) == split]
        if len(self.rows) == 0:
            raise ValueError(f"No rows found for split={split} in manifest={manifest_path}")

        labels = sorted({str(r["label"]) for r in rows})
        self.label_to_index = {label: idx for idx, label in enumerate(labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def _crop_face(self, image: Image.Image, face_bbox: Any) -> Image.Image:
        """Crop face region and resize to face_size."""
        if not face_bbox or not isinstance(face_bbox, (list, tuple)) or len(face_bbox) != 4:
            # Fallback: center crop if no bbox
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            face_crop = image.crop((left, top, left + min_dim, top + min_dim))
        else:
            x1, y1, x2, y2 = [int(v) for v in face_bbox]
            face_crop = image.crop((x1, y1, x2, y2))
        return face_crop.resize((self.face_size, self.face_size), Image.BILINEAR)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rel_path = str(row["image_path"])
        image_path = self.dataset_root / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        image = Image.open(image_path).convert("RGB")
        face_bbox = row.get("face_bbox")

        # Synchronized random horizontal flip on original image
        if self.synchronized_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if face_bbox and isinstance(face_bbox, (list, tuple)) and len(face_bbox) == 4:
                w, h = image.size
                x1, y1, x2, y2 = [int(v) for v in face_bbox]
                face_bbox = [w - x2, y1, w - x1, y2]

        # Face branch: cropped face (paper uses 96x96 for CAER-Net/GLAMOR)
        face_crop = self._crop_face(image, face_bbox)
        face_tensor = self.face_transform(face_crop)

        # Context branch: full image with face masked out
        context_image = _apply_face_mask(image, face_bbox)
        context_tensor = self.context_transform(context_image)

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
