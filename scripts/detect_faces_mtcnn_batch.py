"""
Regenerate CAER-S manifest with MTCNN face bounding boxes (batch processing with checkpointing).
MTCNN runs on CPU because some ops are not compatible with AMD ROCm.
Usage: uv run python scripts/detect_faces_mtcnn_batch.py
"""

from __future__ import annotations

import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facenet_pytorch import MTCNN


def create_mtcnn():
    """Create fresh MTCNN instance to avoid memory leak."""
    return MTCNN(keep_all=True, device="cpu")


def detect_mtcnn(image_path: Path, mtcnn: MTCNN) -> tuple[int, int, int, int] | None:
    """Detect face with MTCNN. Returns (x1, y1, x2, y2) or None."""
    try:
        img = Image.open(image_path).convert("RGB")
        boxes, _ = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return None
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
    checkpoint_path = Path("artifacts/caers/mtcnn_checkpoint.json")
    batch_size = 500
    restart_every_n_batches = 10  # Recreate MTCNN every N batches to prevent memory leak

    print("Loading manifest...")
    rows: list[dict[str, Any]] = []
    with open(manifest_path) as f:
        for line in f:
            rows.append(json.loads(line))

    # Resume from checkpoint if exists
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_idx = ckpt.get("next_idx", 0)
        if start_idx > 0 and output_path.exists():
            with open(output_path) as f:
                processed = [json.loads(line) for line in f]
            # Merge processed rows back into rows
            for i, row in enumerate(processed):
                if i < len(rows):
                    rows[i] = row
            print(f"Resumed from checkpoint: {start_idx}/{len(rows)} already processed")

    print("Initializing MTCNN (CPU)...")
    mtcnn = create_mtcnn()
    batches_since_restart = 0

    for batch_start in range(start_idx, len(rows), batch_size):
        batch_end = min(batch_start + batch_size, len(rows))
        print(f"\nProcessing batch {batch_start}-{batch_end}...")
        
        for i in tqdm(range(batch_start, batch_end), desc=f"batch_{batch_start}"):
            try:
                row = rows[i]
                image_path = dataset_root / row["image_path"]
                if not image_path.exists():
                    row["face_bbox"] = None
                    continue
                bbox = detect_mtcnn(image_path, mtcnn)
                row["face_bbox"] = bbox
            except Exception as e:
                print(f"\nError processing {row.get('image_path', 'unknown')}: {e}")
                traceback.print_exc()
                rows[i]["face_bbox"] = None

        # Write checkpoint after each batch
        try:
            with open(output_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            
            with open(checkpoint_path, "w") as f:
                json.dump({"next_idx": batch_end}, f)
            
            print(f"Checkpoint saved: {batch_end}/{len(rows)}")
        except Exception as e:
            print(f"\nError writing checkpoint: {e}")
            traceback.print_exc()
            # Try to write emergency checkpoint
            try:
                with open(checkpoint_path, "w") as f:
                    json.dump({"next_idx": batch_end}, f)
                print(f"Emergency checkpoint saved at {batch_end}")
            except Exception as e2:
                print(f"Failed to save emergency checkpoint: {e2}")

        batches_since_restart += 1
        
        # Recreate MTCNN periodically to prevent memory leak
        if batches_since_restart >= restart_every_n_batches:
            print("Recreating MTCNN to free memory...")
            del mtcnn
            gc.collect()
            mtcnn = create_mtcnn()
            batches_since_restart = 0
            print("MTCNN recreated.")

    # Final stats
    detected = sum(1 for r in rows if r["face_bbox"] is not None)
    print(f"\nMTCNN detections: {detected}/{len(rows)} ({detected/len(rows)*100:.1f}%)")
    print(f"Manifest saved to {output_path}")


if __name__ == "__main__":
    main()
