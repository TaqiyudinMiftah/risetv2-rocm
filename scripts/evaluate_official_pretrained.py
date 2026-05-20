"""
Evaluate official pretrained CAER-Net on CAER-S test set.
Model from: https://github.com/ndkhanh360/CAER.git
"""

import sys
sys.path.insert(0, '/home/taqiyudinmiftah/risetv2/src')

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.caers_dataset import CAERSTwoStreamDataset
from datasets.transforms import caer_official_transforms
from models.caernet_official import CAERNetOfficial


def evaluate_split(model, device, face_t, ctx_t, split, manifest_path=None):
    if manifest_path is None:
        manifest_path = Path("/home/taqiyudinmiftah/risetv2/artifacts/caers/manifest_caers_mtcnn.jsonl")
    ds = CAERSTwoStreamDataset(
        manifest_path=manifest_path,
        dataset_root=Path("/home/taqiyudinmiftah/risetv2/caer_dataset/CAER-S"),
        split=split,
        image_size=224,
        face_size=96,
        face_transform=face_t,
        context_transform=ctx_t,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    print(f"{split.upper()} samples: {len(ds)}")

    correct = 0
    total = 0
    per_class_correct = {label: 0 for label in ds.label_to_index.keys()}
    per_class_total = {label: 0 for label in ds.label_to_index.keys()}

    with torch.no_grad():
        for batch in tqdm(loader, desc=split):
            face = batch["face_image"].to(device)
            context = batch["context_image"].to(device)
            labels = batch["label"].to(device)

            out = model(face, context)
            preds = out["logits"].argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(preds, labels):
                label_name = ds.index_to_label[label.item()]
                per_class_total[label_name] += 1
                if pred == label:
                    per_class_correct[label_name] += 1

    accuracy = 100.0 * correct / total
    print(f"{split.upper()} Accuracy: {accuracy:.2f}%")
    return accuracy


def evaluate_official():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = CAERNetOfficial(num_classes=7, dropout=0.5).to(device)
    model.eval()

    # Load pretrained weights
    ckpt = torch.load('/home/taqiyudinmiftah/risetv2/checkpoints/caernet_official_state_dict.pt', map_location=device)
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    print("Loaded pretrained weights")

    face_t, ctx_t = caer_official_transforms(train=False)

    manifest_path = Path("/home/taqiyudinmiftah/risetv2/artifacts/caers/manifest_caers_dlib.jsonl")
    if manifest_path.exists():
        print(f"Using dlib manifest: {manifest_path}")
        manifest_dlib = manifest_path
    else:
        manifest_dlib = Path("/home/taqiyudinmiftah/risetv2/artifacts/caers/manifest_caers_mtcnn.jsonl")

    # Evaluate train split
    train_acc = evaluate_split(model, device, face_t, ctx_t, "train", manifest_dlib)
    # Evaluate val split
    val_acc = evaluate_split(model, device, face_t, ctx_t, "val", manifest_dlib)
    # Evaluate test split
    test_acc = evaluate_split(model, device, face_t, ctx_t, "test", manifest_dlib)

    print(f"\n{'='*50}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Val Accuracy:   {val_acc:.2f}%")
    print(f"Test Accuracy:  {test_acc:.2f}%")
    print(f"{'='*50}")
    print(f"\nPaper reported: 73.51%")
    print(f"Difference from paper: {test_acc - 73.51:+.2f}%")

    if train_acc > test_acc + 5:
        print("\nTrain >> Test gap detected: model is overfitting to different face bboxes!")
        print("Likely cause: pretrained model trained with dlib, tested with MTCNN")
    else:
        print("\nTrain ≈ Test: issue is likely model architecture or training procedure")


if __name__ == "__main__":
    evaluate_official()
