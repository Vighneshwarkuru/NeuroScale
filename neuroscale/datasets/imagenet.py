"""ImageNet (ILSVRC 2012) dataset loader."""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path


def get_imagenet_loaders(
    data_dir: str = "./data/imagenet",
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Get ImageNet train and validation data loaders.

    Expects the standard ImageNet directory layout:
        data_dir/
            train/
                n01440764/
                    *.JPEG
                ...
            val/
                n01440764/
                    *.JPEG
                ...

    Args:
        data_dir: Root directory containing 'train' and 'val' folders.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"ImageNet train directory not found: {train_dir}\n"
            "Please download ImageNet and organize it into train/val folders."
        )

    # Standard ImageNet augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=str(train_dir), transform=train_transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=str(val_dir), transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_imagenet_calibration_loader(
    data_dir: str = "./data/imagenet",
    num_samples: int = 1024,
    batch_size: int = 64,
) -> DataLoader:
    """Get a subset of ImageNet training data for SNN calibration.

    Args:
        data_dir: Root directory containing ImageNet.
        num_samples: Number of calibration samples.
        batch_size: Batch size.

    Returns:
        DataLoader with a random subset (no augmentation).
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = torchvision.datasets.ImageFolder(root=str(train_dir), transform=transform)

    indices = torch.randperm(len(dataset))[:num_samples]
    subset = torch.utils.data.Subset(dataset, indices)

    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)
