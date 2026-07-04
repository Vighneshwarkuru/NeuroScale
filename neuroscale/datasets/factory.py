"""Dataset factory — single entry point for all datasets."""

from torch.utils.data import DataLoader
from .cifar import get_cifar10_loaders, get_cifar100_loaders
from .imagenet import get_imagenet_loaders


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create data loaders based on config.

    Args:
        config: Configuration dictionary with 'dataset' and 'model' keys.

    Returns:
        Tuple of (train_loader, test/val_loader).
    """
    dataset_cfg = config["dataset"]
    name = dataset_cfg["name"]
    data_dir = dataset_cfg.get("data_dir", "./data")
    batch_size = config["model"].get("batch_size", 128)

    if name == "cifar10":
        return get_cifar10_loaders(data_dir=data_dir, batch_size=batch_size)
    elif name == "cifar100":
        return get_cifar100_loaders(data_dir=data_dir, batch_size=batch_size)
    elif name == "imagenet":
        return get_imagenet_loaders(data_dir=data_dir, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: cifar10, cifar100, imagenet")
