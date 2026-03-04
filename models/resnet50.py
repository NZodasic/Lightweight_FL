"""
resnet50.py
===========
ResNet-50 model for binary network traffic classification.

The classification head (fc layer) is replaced with a 2-class linear layer
to match the binary classification setup in the paper:
  - Class 0: Benign
  - Class 1: Malicious

Paper: "Lightweight Federated Learning for Efficient Network Intrusion Detection"
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build ResNet-50 with a binary classification head.

    Args:
        num_classes: Number of output classes (default=2 for binary).
        pretrained: Use ImageNet pre-trained weights (default=True).

    Returns:
        model: ResNet-50 with replaced fc layer.
    """
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    # Replace the final fully-connected layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes),
    )

    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze all layers except the classification head (fc)."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
    return model
