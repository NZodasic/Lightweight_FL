import torch
import time

def compute_model_complexity(model, input_size=(1, 3, 32, 32), device="cpu"):
    """
    Computes parameters, FLOPs, model size in MB, and inference latency.
    """
    model.eval()
    model.to(device)
    
    # 1. Number of parameters (Total and Non-Zero)
    total_params = 0
    nonzero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += torch.sum(param != 0).item()
        
    # 2. Model Size (MB)
    model_size_mb = (nonzero_params * 4) / (1024 ** 2) # Assuming float32 (4 bytes)
    
    # 3. Inference Latency
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Measure latency
    start_time = time.time()
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    
    # 4. Rough FLOPs estimation (using standard libraries if available, else structural approximation)
    # We will compute a rough MACs (Multiply-Accumulates) and report FLOPs = 2 * MACs
    macs = estimate_macs(model, input_size, device)
    flops = 2 * macs
    
    return {
        "params": total_params,
        "nonzero_params": nonzero_params,
        "size_mb": model_size_mb,
        "latency_ms": latency_ms,
        "flops": flops
    }

def estimate_macs(model, input_size, device):
    """Simple MACs estimator for Conv2d and Linear layers."""
    macs = 0
    
    # A simple hook-based MACs counter
    def conv2d_hook(module, input, output):
        batch_size, out_channels, out_h, out_w = output.shape
        in_channels = input[0].shape[1]
        kernel_h, kernel_w = module.kernel_size
        groups = module.groups
        
        ops = (out_h * out_w) * (in_channels // groups) * out_channels * kernel_h * kernel_w
        module.__macs__ = ops
        
    def linear_hook(module, input, output):
        weight_ops = module.weight.numel()
        module.__macs__ = weight_ops
        
    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
            
    dummy_input = torch.randn(*input_size).to(device)
    with torch.no_grad():
        model(dummy_input)
        
    for m in model.modules():
        if hasattr(m, '__macs__'):
            macs += m.__macs__
            del m.__macs__
            
    for hook in hooks:
        hook.remove()
        
    return macs
