"""VGG-16 for CIFAR (32x32 images).

Adapted from the standard VGG-16 architecture for smaller inputs.
Uses BatchNorm after each conv layer (VGG-16-BN variant).
"""

import torch
import torch.nn as nn


# VGG-16 configuration: numbers are output channels, 'M' is MaxPool
VGG16_CONFIG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGG16CIFAR(nn.Module):
    """VGG-16 with BatchNorm for CIFAR-10/100.

    Compared to ImageNet VGG-16:
    - No 4096-dim fully connected layers (would be overkill for 32x32)
    - Uses a smaller classifier head
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _make_layers(self) -> nn.Sequential:
        layers = []
        in_channels = 3
        for v in VGG16_CONFIG:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features after each MaxPool layer."""
        features = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                features.append(x)
        return features


def vgg16_cifar(num_classes: int = 10) -> VGG16CIFAR:
    """VGG-16-BN for CIFAR."""
    return VGG16CIFAR(num_classes=num_classes)
