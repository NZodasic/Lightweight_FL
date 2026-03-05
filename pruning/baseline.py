import torch

def compute_structured_mask(model, sparsity):
    """
    Computes a binary mask for each Conv2d layer's weights based on L1-norm filter pruning.
    """
    if sparsity <= 0.0:
        return {}
        
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight.data
            
            # Compute L1 norm of each filter (out_channels)
            # shape: (out_channels, in_channels, kH, kW)
            filter_norms = torch.sum(torch.abs(weight), dim=(1, 2, 3))
            
            # Number of filters to prune
            k = int(len(filter_norms) * sparsity)
            if k == 0:
                masks[name + '.weight'] = torch.ones_like(weight)
                continue
                
            sorted_norms, _ = torch.sort(filter_norms)
            threshold = sorted_norms[k - 1]
            
            # 1 if norm > threshold else 0
            mask_1d = (filter_norms > threshold).float()
            
            # If all are somehow <= threshold, at least keep 1 to avoid crash
            if mask_1d.sum() == 0:
                mask_1d[torch.argmax(filter_norms)] = 1.0
                
            mask = mask_1d.view(-1, 1, 1, 1).expand_as(weight)
            masks[name + '.weight'] = mask
            
    return masks

def apply_mask(model, masks):
    """Zeroes out weights according to the mask."""
    if not masks:
        return
    for name, param in model.named_parameters():
        if name in masks:
            param.data.mul_(masks[name].to(param.device))
            
def apply_mask_to_gradients(model, masks):
    """Zeroes out gradients according to the mask."""
    if not masks:
        return
    for name, param in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad.data.mul_(masks[name].to(param.device))

def calculate_sparsity(model, masks):
    """Return the achieved sparsity of masked model."""
    if not masks:
        return 0.0
    
    total = 0
    zeros = 0
    for name, param in model.named_parameters():
        if name in masks:
            total += param.numel()
            zeros += torch.sum(masks[name] == 0).item()
    return zeros / total if total > 0 else 0.0
