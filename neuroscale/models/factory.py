"""Model factory — create ANN models from config."""

import torch.nn as nn
from .resnet_cifar import resnet20, resnet32
from .vgg_cifar import vgg16_cifar
from .resnet_imagenet import resnet34


def get_ann_model(config: dict) -> nn.Module:
    """Create an ANN model based on configuration.

    Args:
        config: Configuration dictionary with 'model' and 'dataset' keys.

    Returns:
        Instantiated ANN model.
    """
    model_name = config["model"]["ann"]
    num_classes = config["dataset"]["num_classes"]

    model_map = {
        "resnet20": lambda: resnet20(num_classes=num_classes),
        "resnet32": lambda: resnet32(num_classes=num_classes),
        "vgg16": lambda: vgg16_cifar(num_classes=num_classes),
        "resnet34": lambda: resnet34(num_classes=num_classes),
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(model_map.keys())}"
        )

    return model_map[model_name]()
