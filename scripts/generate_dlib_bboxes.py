"""
Generate face bounding boxes for CAER-S test set using dlib CNN face detector.
This matches the original paper's face detection method.
Usage: uv run python scripts/generate_dlib_bboxes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import dlib
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.io_utils import read_jsonl


def detect_dlib(image_path: Path, detector) -> tuple[int, int, int, int] | None:
    """Detect face with dlib CNN. Returns (x1, y1, x2, y2) or None."""
    try:
        img = dlib.load_rgb_image(str(image_path))
        detections = detector(img, 1)  # upsample once
        if len(detections) == 0:
            return None
        
        # Take the largest face
        areas = [(d.rect.width() * d.rect.height(), d) for d in detections]
        areas.sort(reverse=True)
        best = areas[0][1]
        
        x1, y1, x2, y2 = best.rect.left(), best.rect.top(), best.rect.right(), best.rect.bottom()
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None


def main():
    dataset_root = Path("/home/taqiyudinmiftah/risetv2/caer_dataset/CAER-S")
    manifest_path = Path("artifacts/caers/manifest_caers_mtcnn.jsonl")
    output_path = Path("artifacts/caers/manifest_caers_dlib_test.jsonl")
    model_path = Path("checkpoints/dlib/mmod_human_face_detector.dat")
    
    print("Loading dlib CNN face detector...")
    detector = dlib.cnn_face_detection_model_v1(str(model_path))
    
    print("Loading manifest...")
    rows = read_jsonl(manifest_path)
    
    # Filter test split only
    test_rows = [r for r in rows if str(r.get("split")) == "test"]
    print(f"Test images to process: {len(test_rows)}")
    
    # Build lookup for existing rows
    all_rows_dict = {r["sample_id"]: r for r in rows}
    
    detected_count = 0
    for row in tqdm(test_rows, desc="dlib test detection"):
        image_path = dataset_root / row["image_path"]
        bbox = detect_dlib(image_path, detector)
        if bbox:
            all_rows_dict[row["sample_id"]]["face_bbox"] = bbox
            detected_count += 1
        else:
            # Keep existing MTCNN bbox if dlib fails
            pass
    
    # Write output
    output_rows = [all_rows_dict[r["sample_id"]] for r in rows]
    with open(output_path, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")
    
    print(f"\nDone! Detected {detected_count}/{len(test_rows)} test faces with dlib")
    print(f"Output: {output_path}")
    
    # Stats
    total_test = len(test_rows)
    print(f"Coverage: {detected_count}/{total_test} ({detected_count/total_test*100:.1f}%)")


if __name__ == "__main__":
    main()
