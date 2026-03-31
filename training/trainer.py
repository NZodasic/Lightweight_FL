import copy
import torch
import torch.nn as nn
import torch.optim as optim
from pruning.baseline import apply_mask, apply_mask_to_gradients

class FLClient:
    def __init__(self, client_id, dataloader, device):
        self.id = client_id
        self.dataloader = dataloader
        self.device = device
        
    def train(self, global_model, masks, epochs, lr, momentum, weight_decay):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        # Apply mask initially
        apply_mask(model, masks)
        
        running_loss = 0.0
        for ep in range(epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply mask to gradient strictly before step to ensure masked weights stay 0
                apply_mask_to_gradients(model, masks)
                optimizer.step()
                running_loss += loss.item()
                
        avg_loss = running_loss / (epochs * len(self.dataloader))
        return model.state_dict(), avg_loss


def fedavg(client_weights):
    """Simple FedAvg over equivalent sized clients."""
    global_weights = copy.deepcopy(client_weights[0])
    num_clients = len(client_weights)
    for key in global_weights.keys():
        for i in range(1, num_clients):
            global_weights[key] += client_weights[i][key]
        global_weights[key] = torch.div(global_weights[key], float(num_clients))
    return global_weights


class FLServer:
    def __init__(self, model, config, device, masks):
        self.global_model = model.to(device)
        self.config = config
        self.device = device
        self.masks = masks
        
    def aggregate(self, client_weights_list):
        averaged_weights = fedavg(client_weights_list)
        self.global_model.load_state_dict(averaged_weights)
        # Ensure mask maintains perfectly
        apply_mask(self.global_model, self.masks)
        
    def get_model(self):
        return self.global_model
