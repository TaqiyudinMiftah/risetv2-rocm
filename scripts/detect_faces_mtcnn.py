"""
Regenerate CAER-S manifest with MTCNN face bounding boxes.
MTCNN runs on CPU because some ops are not compatible with AMD ROCm.
Usage: uv run python scripts/detect_faces_mtcnn.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facenet_pytorch import MTCNN


def detect_mtcnn(image_path: Path, mtcnn: MTCNN) -> tuple[int, int, int, int] | None:
    """Detect face with MTCNN. Returns (x1, y1, x2, y2) or None."""
    try:
        img = Image.open(image_path).convert("RGB")
        boxes, _ = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return None
        # Pick largest face
        import numpy as np
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[idx]
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None


def main():
    dataset_root = Path("/home/taqiyudinmiftah/risetv2/caer_dataset/CAER-S")
    manifest_path = Path("artifacts/caers/manifest_caers.jsonl")
    output_path = Path("artifacts/caers/manifest_caers_mtcnn.jsonl")

    print("Loading manifest...")
    rows: list[dict[str, Any]] = []
    with open(manifest_path) as f:
        for line in f:
            rows.append(json.loads(line))

    print("Initializing MTCNN (CPU)...")
    mtcnn = MTCNN(keep_all=True, device="cpu")

    updated = 0
    missed = 0

    print(f"Processing {len(rows)} images...")
    for row in tqdm(rows):
        image_path = dataset_root / row["image_path"]
        if not image_path.exists():
            row["face_bbox"] = None
            missed += 1
            continue

        bbox = detect_mtcnn(image_path, mtcnn)
        row["face_bbox"] = bbox
        if bbox is not None:
            updated += 1
        else:
            missed += 1

    print(f"\nMTCNN detections: {updated}/{len(rows)} ({updated/len(rows)*100:.1f}%)")
    print(f"Missed: {missed}/{len(rows)} ({missed/len(rows)*100:.1f}%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Manifest saved to {output_path}")


if __name__ == "__main__":
    main()
