"""CIFAR-10 and CIFAR-100 dataset loaders."""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test data loaders.

    Args:
        data_dir: Root directory for dataset storage.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    # Standard CIFAR-10 augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def get_cifar100_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-100 train and test data loaders.

    Args:
        data_dir: Root directory for dataset storage.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    # Standard CIFAR-100 augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def get_cifar_calibration_loader(
    dataset_name: str = "cifar10",
    data_dir: str = "./data",
    num_samples: int = 1024,
    batch_size: int = 64,
) -> DataLoader:
    """Get a subset loader for SNN calibration (no augmentation).

    Args:
        dataset_name: 'cifar10' or 'cifar100'.
        data_dir: Root directory for dataset storage.
        num_samples: Number of calibration samples.
        batch_size: Batch size.

    Returns:
        DataLoader with a subset of training data (no augmentation).
    """
    if dataset_name == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        dataset_cls = torchvision.datasets.CIFAR10
    else:
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
        dataset_cls = torchvision.datasets.CIFAR100

    transform = transforms.Compose([transforms.ToTensor(), normalize])

    dataset = dataset_cls(root=data_dir, train=True, download=True, transform=transform)

    # Take a random subset
    indices = torch.randperm(len(dataset))[:num_samples]
    subset = torch.utils.data.Subset(dataset, indices)

    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
