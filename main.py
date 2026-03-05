import os
import yaml
import torch
import random
from data.data_loader import get_dataset, partition_data, get_dataloaders
from models.resnet50 import get_resnet50
from models.model_utils import compute_model_complexity
from pruning.baseline import compute_structured_mask, calculate_sparsity
from training.trainer import FLServer, FLClient
from training.evaluator import evaluate_global_model
from utils.system import set_seed, get_device, get_optimal_batch_size
from utils.logger import FL_Logger
from utils.viz import plot_training_curves

def main():
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = get_device()
    batch_size = get_optimal_batch_size(config['dataset']['batch_size'])
    
    logger = FL_Logger(log_dir="EXPERIMENT")
    
    # 1. Dataset Loading
    dataset = get_dataset(config['dataset']['archive_path'], img_size=config['dataset']['img_size'])
    total_samples = len(dataset)
    
    # Simple 80/20 train/test split for global evaluation
    test_size = int(0.2 * total_samples)
    train_size = total_samples - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Partition Training data for FL Clients
    client_datasets = partition_data(train_dataset, 
                                     num_clients=config['federated']['num_clients'],
                                     partition_type=config['federated']['partition'],
                                     dirichlet_alpha=config['federated']['dirichlet_alpha'])
    
    client_loaders = get_dataloaders(train_dataset, client_datasets, batch_size=batch_size)
    samples_per_client = int(train_size / config['federated']['num_clients'])
    
    # ==========================
    # Print Dataset Description
    # ==========================
    logger.info("Dataset Description:")
    logger.info(f"Dataset: {config['dataset']['name']}")
    logger.info(f"Total samples : {total_samples}")
    logger.info(f"Classes       : 2 (Benign, Malicious)")
    logger.info(f"Training samples : {train_size}")
    logger.info(f"Testing samples  : {test_size}")
    logger.info(f"Number of clients: {config['federated']['num_clients']}")
    logger.info(f"Data distribution: {config['federated']['partition'].upper()}")
    logger.info(f"Samples per client: ~{samples_per_client}")
    logger.info("---")
    
    # 2. Model Initialization & Pruning
    global_model = get_resnet50(num_classes=2)
    complexity_before = compute_model_complexity(global_model, device=device)
    
    masks = compute_structured_mask(global_model, config['pruning']['sparsity'])
    actual_sparsity = calculate_sparsity(global_model, masks)
    
    # Count remaining parameters roughly
    remaining_params = int(complexity_before['params'] * (1 - actual_sparsity))
    
    # ==========================
    # Print Experimental Setup
    # ==========================
    clients_per_round = max(1, int(config['federated']['fraction_fit'] * config['federated']['num_clients']))
    
    logger.info("Experimental Setup, model compression / pruning")
    logger.info(f"Framework : PyTorch")
    logger.info(f"Device    : {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs    : {config['federated']['local_epochs']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Optimizer : SGD")
    logger.info("-" * 15)
    logger.info("Federated settings:")
    logger.info(f"Rounds: {config['federated']['num_rounds']}")
    logger.info(f"Clients per round: {clients_per_round}")
    logger.info(f"Aggregation: FedAvg")
    logger.info("-" * 15)
    logger.info(f"Model: {config['training']['model_name'].capitalize()}")
    logger.info(f"Pruning sparsity: {config['pruning']['sparsity']:.2f} (Actual: {actual_sparsity:.2f})")
    logger.info(f"Remaining parameters: {remaining_params}")
    logger.info("---")
    
    # 3. Federated Learning Setup
    server = FLServer(global_model, config, device, masks)
    clients = [FLClient(i, cl, device) for i, cl in enumerate(client_loaders)]
    
    # 4. Training Loop
    logger.info("Starting Federated Training...")
    
    for round_num in range(1, config['federated']['num_rounds'] + 1):
        # Sample clients
        sampled_clients = random.sample(clients, clients_per_round)
        
        client_weights = []
        round_loss = 0.0
        
        # Local training
        for client in sampled_clients:
            weights, loss = client.train(global_model=server.get_model(),
                                         masks=masks,
                                         epochs=config['federated']['local_epochs'],
                                         lr=config['training']['learning_rate'],
                                         momentum=config['training']['momentum'],
                                         weight_decay=config['training']['weight_decay'])
            client_weights.append(weights)
            round_loss += loss
            
        round_loss /= len(sampled_clients)
        
        # Aggregation
        server.aggregate(client_weights)
        
        # Evaluation
        eval_metrics = evaluate_global_model(server.get_model(), test_loader, device)
        
        # Calculate Comm Cost based on Remaining Params
        comm_cost_mb = (remaining_params * 4) / (1024**2) # in MB, per client sent
        total_comm_cost = comm_cost_mb * 2 * clients_per_round # upload + download
        
        logger.log_round(round_num, {
            "loss": round_loss,
            "val_accuracy": eval_metrics['accuracy'],
            "val_f1": eval_metrics['f1'],
            "comm_cost_mb": round(total_comm_cost, 2),
            "client_stats": f"{clients_per_round} clients participated"
        })
        
    # Plot Training Curves
    plot_training_curves(logger.training_history, os.path.join("EXPERIMENT", "training_curve.png"))
    logger.save_history(os.path.join("EXPERIMENT", "history.json"))
    
    logger.info("Training Completed!")
    
    # End Report Model Complexity Metrics
    logger.info("\n# Final Model Complexity Metrics")
    logger.info(f"- number of parameters: {remaining_params}")
    logger.info(f"- FLOPs: {complexity_before['flops'] * (1 - actual_sparsity):.0f} (Estimated pruned FLOPs)")
    logger.info(f"- model size (MB): {(remaining_params * 4) / (1024**2):.2f}")
    logger.info(f"- inference latency: {complexity_before['latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
