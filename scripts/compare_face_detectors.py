"""
Compare MTCNN vs OpenCV Haar Cascade face detection on CAER-S samples.
Run: uv run python scripts/compare_face_detectors.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facenet_pytorch import MTCNN


def load_manifest_sample(manifest_path: str, n: int = 100) -> list[dict]:
    rows = []
    with open(manifest_path) as f:
        for line in f:
            rows.append(json.loads(line))
    random.seed(42)
    return random.sample(rows, min(n, len(rows)))


def detect_opencv(image_path: Path) -> tuple[int, int, int, int] | None:
    """OpenCV Haar Cascade detection (same as current pipeline)."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    # Pick largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return int(x), int(y), int(x + w), int(y + h)


def detect_mtcnn(image_path: Path, mtcnn: MTCNN) -> tuple[int, int, int, int] | None:
    """MTCNN detection."""
    img = Image.open(image_path).convert("RGB")
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return None
    # Pick largest face
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[idx]
    return int(x1), int(y1), int(x2), int(y2)


def iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    dataset_root = Path("/home/taqiyudinmiftah/risetv2/caer_dataset/CAER-S")
    manifest_path = "artifacts/caers/manifest_caers.jsonl"
    samples = load_manifest_sample(manifest_path, n=200)

    print("Initializing MTCNN (CPU)...")
    # MTCNN has CUDA-specific ops that crash on AMD ROCm; force CPU
    mtcnn = MTCNN(keep_all=True, device="cpu")

    opencv_hits = 0
    mtcnn_hits = 0
    ious = []
    mismatches = []

    print(f"Comparing {len(samples)} samples...")
    for row in samples:
        image_path = dataset_root / row["image_path"]
        if not image_path.exists():
            continue

        opencv_box = detect_opencv(image_path)
        mtcnn_box = detect_mtcnn(image_path, mtcnn)

        if opencv_box is not None:
            opencv_hits += 1
        if mtcnn_box is not None:
            mtcnn_hits += 1

        if opencv_box is not None and mtcnn_box is not None:
            iou_val = iou(opencv_box, mtcnn_box)
            ious.append(iou_val)
            if iou_val < 0.5:
                mismatches.append({
                    "path": str(image_path),
                    "opencv": opencv_box,
                    "mtcnn": mtcnn_box,
                    "iou": iou_val,
                })

    print("\n=== Results ===")
    print(f"OpenCV Haar hits: {opencv_hits}/{len(samples)} ({opencv_hits/len(samples)*100:.1f}%)")
    print(f"MTCNN hits:       {mtcnn_hits}/{len(samples)} ({mtcnn_hits/len(samples)*100:.1f}%)")
    if ious:
        print(f"Mean IoU (when both detect): {sum(ious)/len(ious):.3f}")
        print(f"Median IoU: {sorted(ious)[len(ious)//2]:.3f}")
        print(f"IoU < 0.5 mismatches: {len(mismatches)}")
    else:
        print("No overlapping detections to compare.")

    if mismatches:
        print("\nSample mismatches (IoU < 0.5):")
        for m in mismatches[:5]:
            print(f"  {m['path']}: OpenCV={m['opencv']} MTCNN={m['mtcnn']} IoU={m['iou']:.3f}")


if __name__ == "__main__":
    import torch
    main()
