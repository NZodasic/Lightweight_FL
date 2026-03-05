import os
import logging
import json

class FL_Logger:
    def __init__(self, log_dir, log_file="experiment.log"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("FL_Logger")
        self.training_history = []
    
    def info(self, msg):
        self.logger.info(msg)
        
    def warn(self, msg):
        self.logger.warning(msg)
        
    def log_round(self, round_num, metrics):
        """
        Logs metrics at the end of an FL round.
        metrics: dict containing 'val_accuracy', 'loss', 'comm_cost_mb', etc.
        """
        msg = f"Round {round_num} | " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(msg)
        metrics['round'] = round_num
        self.training_history.append(metrics)
        
    def save_history(self, path):
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
