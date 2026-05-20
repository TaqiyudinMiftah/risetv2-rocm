from __future__ import annotations

from typing import Any

from torchvision import transforms


def default_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def augmented_transform(image_size: int, flip: bool = True) -> transforms.Compose:
    """Training transform with light augmentation."""
    ops: list[Any] = [
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
    ]
    if flip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


def caer_net_transforms(image_size: int = 224, face_size: int = 96, augment: bool = False) -> tuple[Any, Any]:
    """
    Paper-specific transforms for CAER-Net and GLAMOR-Net.
    Face branch: 96x96 (cropped face per bbox)
    Context branch: 224x224 (full image with face blacked out)
    """
    if augment:
        face_t = transforms.Compose([
            transforms.Resize((face_size + 8, face_size + 8)),
            transforms.RandomCrop(face_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ctx_t = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        face_t = transforms.Compose([
            transforms.Resize((face_size, face_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ctx_t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return face_t, ctx_t


def caer_official_transforms(train: bool = True) -> tuple[Any, Any]:
    """
    Official CAER-Net transforms from ndkhanh360/CAER.git
    Face: 96x96
    Context: resize to (128, 171), then RandomCrop 112 (train) or CenterCrop 112 (test)
    """
    face_t = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if train:
        ctx_t = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.RandomCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        ctx_t = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return face_t, ctx_t
