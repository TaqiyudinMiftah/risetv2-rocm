"""
datasets/caers.py
CAER-S two-stream dataset for CAER-Net-S (static model).

Two-stream preprocessing (Section 3.2.1):
  Face stream    – detect face → crop (padded) → resize to image_size×image_size
  Context stream – detect face → black-out face region → resize to image_size×image_size
  Fallback       – if no face detected: centre-crop as face, full image as context

Face detector: OpenCV Haar Cascade (bundled inside opencv-python, no downloads).

Dataset layout expected (either):
  dataset_root/
    train/
      Anger/ Disgust/ Fear/ Happy/ Neutral/ Sad/ Surprise/
    test/
      ...
  ─── or ───
  dataset_root/
    Anger/ ... Surprise/     (no split dirs; caller must split manually)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion label map  (paper: 7 categories)
# ---------------------------------------------------------------------------
EMOTION_CLASSES: List[str] = [
    "Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
]
_CLS2IDX: Dict[str, int] = {c: i for i, c in enumerate(EMOTION_CLASSES)}
# accept lower-case folder names too
_CLS2IDX.update({c.lower(): i for c, i in _CLS2IDX.items()})
NUM_CLASSES = len(EMOTION_CLASSES)


# ---------------------------------------------------------------------------
# Face detector
# ---------------------------------------------------------------------------
class FaceDetector:
    """
    OpenCV Haar-Cascade face detector.
    Uses the XML bundled with opencv-python (cv2.data.haarcascades).
    No external downloads required.
    """

    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.clf = cv2.CascadeClassifier(path)
        if self.clf.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from '{path}'. "
                "Ensure opencv-python is properly installed."
            )
        self._miss = 0
        self._total = 0

    def detect(
        self, img_bgr: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in *img_bgr*.
        Returns (x, y, w, h) or None.
        """
        self._total += 1
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            self._miss += 1
            return None
        # Largest face by area
        idx  = int(np.argmax([w * h for x, y, w, h in faces]))
        return tuple(int(v) for v in faces[idx])

    @property
    def miss_rate(self) -> float:
        return self._miss / max(1, self._total)


# ---------------------------------------------------------------------------
# CAER-S Dataset
# ---------------------------------------------------------------------------
class CAERSDataset(Dataset):
    """
    Two-stream CAER-S dataset.

    Parameters
    ----------
    samples        : list of (image_path, label_idx)
    image_size     : int   – both streams are resized to image_size×image_size
    augment        : bool  – apply paper augmentations (train only)
    face_detector  : FaceDetector or None (creates one if None)
    pad_ratio      : float – extra padding around the detected face crop
    """

    # ImageNet normalisation (used for pretrained backbones)
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        samples:       List[Tuple[str, int]],
        image_size:    int  = 224,
        augment:       bool = True,
        face_detector: Optional[FaceDetector] = None,
        pad_ratio:     float = 0.20,
    ) -> None:
        super().__init__()
        self.samples       = samples
        self.image_size    = image_size
        self.augment       = augment
        self.face_detector = face_detector if face_detector else FaceDetector()
        self.pad_ratio     = pad_ratio

        # Colour jitter (paper: "contrast and color changes")
        self._jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

        # Final normalise+ToTensor (no size here – done per-item)
        self._normalise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_bgr(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return img

    def _crop_face(
        self, img: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Return a padded square face crop."""
        H, W = img.shape[:2]
        x, y, w, h = bbox
        px, py = int(w * self.pad_ratio), int(h * self.pad_ratio)
        x1 = max(0,  x - px)
        y1 = max(0,  y - py)
        x2 = min(W,  x + w + px)
        y2 = min(H,  y + h + py)
        return img[y1:y2, x1:x2]

    @staticmethod
    def _hide_face(
        img: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Return a copy of *img* with the face region blacked out."""
        out = img.copy()
        x, y, w, h = bbox
        out[y : y + h, x : x + w] = 0
        return out

    @staticmethod
    def _centre_crop(img: np.ndarray) -> np.ndarray:
        """Fallback square centre crop."""
        H, W = img.shape[:2]
        s = min(H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        return img[y0 : y0 + s, x0 : x0 + s]

    def _bgr_to_pil(self, img: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _apply_transforms(
        self, face_pil: Image.Image, ctx_pil: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment and normalise both streams."""
        sz = self.image_size

        if self.augment:
            # Synchronised horizontal flip (paper: "flips")
            if random.random() < 0.5:
                face_pil = TF.hflip(face_pil)
                ctx_pil  = TF.hflip(ctx_pil)

            # Independent colour jitter (paper: "contrast, color changes")
            face_pil = self._jitter(face_pil)
            ctx_pil  = self._jitter(ctx_pil)

            # Small random crop after resize (paper: random crop on context)
            face_pil = TF.resize(face_pil, [int(sz * 1.1), int(sz * 1.1)])
            ctx_pil  = TF.resize(ctx_pil,  [int(sz * 1.1), int(sz * 1.1)])
            i, j, h, w = transforms.RandomCrop.get_params(
                face_pil, output_size=(sz, sz)
            )
            face_pil = TF.crop(face_pil, i, j, h, w)
            # Same crop position for context
            ctx_pil  = TF.crop(ctx_pil,  i, j, h, w)

        else:
            face_pil = TF.resize(face_pil, [sz, sz])
            ctx_pil  = TF.resize(ctx_pil,  [sz, sz])

        return self._normalise(face_pil), self._normalise(ctx_pil)

    # ------------------------------------------------------------------ #
    # Dataset interface                                                    #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label = self.samples[idx]

        try:
            img_bgr = self._load_bgr(path)
        except FileNotFoundError:
            logger.warning(f"Skipping unreadable image: {path}")
            dummy = torch.zeros(3, self.image_size, self.image_size)
            return {"face": dummy, "context": dummy, "label": torch.tensor(label)}

        bbox = self.face_detector.detect(img_bgr)

        if bbox is not None:
            face_bgr = self._crop_face(img_bgr, bbox)
            ctx_bgr  = self._hide_face(img_bgr, bbox)
        else:
            # Fallback: centre-crop as face, full image as context
            face_bgr = self._centre_crop(img_bgr)
            ctx_bgr  = img_bgr

        face_pil = self._bgr_to_pil(face_bgr)
        ctx_pil  = self._bgr_to_pil(ctx_bgr)

        face_t, ctx_t = self._apply_transforms(face_pil, ctx_pil)

        return {
            "face":    face_t,                              # [3, H, W]
            "context": ctx_t,                               # [3, H, W]
            "label":   torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Sample discovery helpers
# ---------------------------------------------------------------------------

def discover_samples(
    root: str,
    image_extensions: List[str],
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Walk *root* and return (train_samples, test_samples).

    Handles two layouts:
      root/train/<Class>/<img>  +  root/test/<Class>/<img>
      root/<Class>/<img>         (no split dirs – test list is empty)
    """
    root   = Path(root)
    exts   = {e.lower() for e in image_extensions}

    def _scan(split_dir: Path) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        if not split_dir.exists():
            return samples
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = _CLS2IDX.get(cls_dir.name) if cls_dir.name in _CLS2IDX \
                else _CLS2IDX.get(cls_dir.name.lower())
            if label is None:
                logger.warning(f"Unrecognised class folder '{cls_dir.name}' – skipped.")
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in exts:
                    samples.append((str(img_path), label))
        return samples

    train_dir = root / "train"
    test_dir  = root / "test"

    if train_dir.exists():
        train_s = _scan(train_dir)
        test_s  = _scan(test_dir)
    else:
        # Flat layout: treat everything as train; test split done externally
        train_s = _scan(root)
        test_s  = []
        logger.info("No train/ dir found – using flat layout; test set will be empty.")

    logger.info(
        f"Discovered {len(train_s):,} train  /  {len(test_s):,} test samples "
        f"under '{root}'."
    )
    return train_s, test_s


def split_train_val(
    samples: List[Tuple[str, int]],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Stratified (per-class) train / val split to keep class distribution.
    """
    from collections import defaultdict
    rng = random.Random(seed)

    by_class: Dict[int, List] = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)

    train_out: List[Tuple[str, int]] = []
    val_out:   List[Tuple[str, int]] = []

    for cls_samples in by_class.values():
        shuffled = cls_samples[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_out.extend(shuffled[:n_val])
        train_out.extend(shuffled[n_val:])

    rng.shuffle(train_out)
    rng.shuffle(val_out)

    logger.info(
        f"Stratified split → train {len(train_out):,}  val {len(val_out):,}"
    )
    return train_out, val_out
