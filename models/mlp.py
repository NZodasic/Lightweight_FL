import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=9, num_classes=2):
        super(MLP, self).__init__()
        # A simple linear network
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # Flatten x just in case it's batched as (B, 1, 9)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def get_mlp(input_dim=9, num_classes=2):
    """
    Returns an MLP model adapted for tabular input arrays.
    """
    return MLP(input_dim=input_dim, num_classes=num_classes)
