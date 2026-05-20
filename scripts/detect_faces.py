from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


def detect_face_bbox(image_path: Path) -> list[int] | None:
    """
    Detect face bounding box using OpenCV Haar Cascade.
    Returns [x1, y1, x2, y2] or None if no face detected.
    """
    # Load Haar Cascade (included with opencv-python-headless)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return None

    # Take the largest face (by area)
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest

    return [int(x), int(y), int(x + w), int(y + h)]


def main():
    if len(sys.argv) < 3:
        print("Usage: python detect_faces.py <manifest_path> <dataset_root>")
        sys.exit(1)

    manifest_path = Path(sys.argv[1])
    dataset_root = Path(sys.argv[2]).expanduser()

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    # Read manifest
    with manifest_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    print(f"Processing {len(rows)} images for face detection...")

    detected = 0
    failed = 0

    for row in tqdm(rows, desc="Detecting faces"):
        image_path = dataset_root / row["image_path"]

        if row.get("face_bbox") is not None:
            detected += 1
            continue

        bbox = detect_face_bbox(image_path)
        if bbox is not None:
            row["face_bbox"] = bbox
            detected += 1
        else:
            failed += 1

    # Write updated manifest
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nDone!")
    print(f"  Total: {len(rows)}")
    print(f"  With bbox: {detected}")
    print(f"  Failed (no face): {failed}")
    print(f"  Success rate: {detected/len(rows)*100:.2f}%")


if __name__ == "__main__":
    main()
