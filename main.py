import argparse
import os
import yaml
import torch
import random
from data.data_loader import get_dataset, get_prepartitioned_client_datasets, get_dataloaders
from models.resnet50 import get_resnet50
from models.mlp import get_mlp
from models.model_utils import compute_model_complexity
from pruning.baseline import compute_structured_mask, calculate_sparsity
from training.trainer import FLServer, FLClient
from training.evaluator import evaluate_global_model
from utils.system import set_seed, get_device, get_optimal_batch_size
from utils.logger import FL_Logger
from utils.viz import plot_training_curves

def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning with Saliency Learning XAI Module")
    parser.add_argument('--use_xai', action='store_true', help="Enable Saliency Learning")
    args = parser.parse_args()

    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    config['training']['use_xai'] = args.use_xai
        
    set_seed(config['seed'])
    device = get_device()
    batch_size = get_optimal_batch_size(config['dataset']['batch_size'])
    
    logger = FL_Logger(log_dir="EXPERIMENT")
    
    # 1. Dataset Loading
    # Use prepartitioned train and test files
    train_path = os.path.join(config['dataset']['archive_path'], "train.parquet")
    test_path = os.path.join(config['dataset']['archive_path'], "test.parquet")
    
    concept_name = config['federated'].get('concept', 'concept_1')
    concept_path = os.path.join(config['dataset']['archive_path'], f"{concept_name}.parquet")
    
    logger.info(f"Loading global train dataset: {train_path}")
    train_dataset = get_dataset(train_path)
    logger.info(f"Loading global test dataset: {test_path}")
    test_dataset = get_dataset(test_path)
    
    total_samples = len(train_dataset) + len(test_dataset)
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load Concept 1 directly
    logger.info(f"Loading pre-partitioned client data: {concept_path}")
    client_datasets, num_clients_actual = get_prepartitioned_client_datasets(concept_path)
    
    config['federated']['num_clients'] = num_clients_actual
    
    client_loaders = get_dataloaders(client_datasets, batch_size=batch_size)
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
    if config['training']['model_name'].lower() == 'mlp':
        global_model = get_mlp(input_dim=config['dataset']['img_size'], num_classes=2)
        dummy_input_size = (1, config['dataset']['img_size'])
    else:
        global_model = get_resnet50(num_classes=2)
        dummy_input_size = (1, 3, config['dataset']['img_size'], config['dataset']['img_size'])
        
    complexity_before = compute_model_complexity(global_model, input_size=dummy_input_size, device=device)
    
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
    clients = [FLClient(i, cl, device, config=config) for i, cl in enumerate(client_loaders)]
    
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
