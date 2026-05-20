"""
Build manifest with official dlib CNN face detections from the paper.
Downloaded from: https://github.com/ndkhanh360/CAER
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_detection_file(file_path: Path) -> dict[str, tuple[int, int, int, int]]:
    """Parse detection file: path,label_idx,x1,y1,x2,y2"""
    detections = {}
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                continue
            rel_path = parts[0]  # e.g., "Surprise/2492.png"
            x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            detections[rel_path] = (x1, y1, x2, y2)
    return detections


def main():
    dlib_dir = Path("checkpoints/dlib")
    output_path = Path("artifacts/caers/manifest_caers_dlib.jsonl")
    dataset_root = Path("/home/taqiyudinmiftah/risetv2/caer_dataset/CAER-S")
    
    # Load official detections
    print("Loading official dlib detections...")
    train_det = parse_detection_file(dlib_dir / "train_detect.txt")
    val_det = parse_detection_file(dlib_dir / "val_detect.txt")
    test_det = parse_detection_file(dlib_dir / "test_detect.txt")
    
    print(f"Train detections: {len(train_det)}")
    print(f"Val detections: {len(val_det)}")
    print(f"Test detections: {len(test_det)}")
    
    # Build manifest - find actual file paths
    def build_rows(split: str, detections: dict):
        rows = []
        for rel_path, (x1, y1, x2, y2) in detections.items():
            # rel_path is like "Surprise/2492.png"
            folder_label = rel_path.split("/")[0]  # actual folder name in dataset
            filename = rel_path.split("/")[1]
            
            # Normalize label: official uses "Anger", but we use "Angry" consistently
            label_name = "Angry" if folder_label == "Anger" else folder_label
            
            # Try different paths to find the actual file
            possible_paths = [
                dataset_root / split / folder_label / filename,
            ]
            
            # For val, images might be in train folder
            if split == "val":
                possible_paths.insert(0, dataset_root / "train" / folder_label / filename)
            
            found_path = None
            for p in possible_paths:
                if p.exists():
                    found_path = p
                    break
            
            if found_path is None:
                # Try mapping Anger -> Angry for train split
                if folder_label == "Anger":
                    alt_path = dataset_root / "train" / "Angry" / filename
                    if alt_path.exists():
                        folder_label = "Angry"
                        found_path = alt_path
                if found_path is None:
                    print(f"  Warning: could not find {rel_path} for {split}")
                    continue
            
            # Build image_path relative to dataset_root
            if split == "val" and found_path.parent.parent.name == "train":
                # Val images that exist in train folder
                image_path = f"train/{folder_label}/{filename}"
            else:
                image_path = f"{split}/{folder_label}/{filename}"
            
            sample_id = f"{split}__{folder_label}__{filename.replace('.png', '_png')}"
            
            rows.append({
                "sample_id": sample_id,
                "image_path": image_path,
                "label": label_name,
                "split": split,
                "face_bbox": [x1, y1, x2, y2],
            })
        return rows
    
    train_rows = build_rows("train", train_det)
    val_rows = build_rows("val", val_det)
    test_rows = build_rows("test", test_det)
    
    all_rows = train_rows + val_rows + test_rows
    
    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    
    print(f"\nManifest saved: {output_path}")
    print(f"Total rows: {len(all_rows)}")
    print(f"  Train: {len(train_rows)}")
    print(f"  Val: {len(val_rows)}")
    print(f"  Test: {len(test_rows)}")
    
    # Verify
    counts = {"train": 0, "val": 0, "test": 0}
    for row in all_rows:
        counts[row["split"]] += 1
    print(f"Split counts: {counts}")


if __name__ == "__main__":
    main()
