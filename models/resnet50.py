import torch
import torch.nn as nn
from torchvision.models import resnet50

def get_resnet50(num_classes=2):
    """
    Returns a ResNet-50 model adapted for 2 classes and 32x32 input images.
    """
    model = resnet50(weights=None)
    
    # Adapt the first convolutional layer for 32x32 images
    # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Replaced to avoid excessive downsampling early on.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # We ideally remove the maxpool to preserve spatial dimensions for 32x32
    # model.maxpool = nn.Identity()
    
    # Change fully connected layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
