"""
trainer.py
==========
Federated Learning training loop implementing:
  - FL Server: global model initialization, one-time pruning, FedAvg aggregation
  - FL Client: local training with Adam optimizer, BCE loss, E epochs
  - Communication cost tracking

Paper Configuration (Section IV-B):
  Optimizer: Adam, lr=2e-4
  Batch size: 32
  Local epochs E: 10
  Rounds T: 5
  Clients N: 10 / 50 / 100
  Client fraction: 50%
  Aggregation: FedAvg
"""

import copy
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# FedAvg aggregation
# ──────────────────────────────────────────────────────────────────────────────

def fedavg(
    client_weights: List[dict],
    client_sizes:   List[int],
) -> dict:
    """
    Weighted FedAvg aggregation.

    θ_{t+1} = Σ (n_i / N) * θ_i
    """
    total = sum(client_sizes)
    avg_weights = copy.deepcopy(client_weights[0])

    for key in avg_weights:
        avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)

    for w, n in zip(client_weights, client_sizes):
        weight = n / total
        for key in avg_weights:
            avg_weights[key] += weight * w[key].float()

    return avg_weights


# ──────────────────────────────────────────────────────────────────────────────
# FL Client
# ──────────────────────────────────────────────────────────────────────────────

class FLClient:
    """
    Federated Learning client — performs local training for E epochs.
    Applies the pruned model structure from the server.

    Algorithm 1 (paper):
      1. Receive global model θ and pruning mask M
      2. Apply mask: θ ← θ ⊙ M  (already baked-in for structured pruning)
      3. Train E epochs with Adam on local dataset D_j
      4. Upload updated sparse model back to server
    """

    def __init__(
        self,
        client_id: int,
        local_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.local_loader = local_loader
        self.device = device
        self.n_samples = len(local_loader.dataset)

    def local_train(
        self,
        global_model: nn.Module,
        local_epochs: int = 10,
        lr: float = 2e-4,
    ) -> Tuple[dict, float]:
        """
        Train local model for `local_epochs` epochs.

        Returns:
            (state_dict, avg_train_loss)
        """
        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_batches = 0

        for epoch in range(local_epochs):
            for imgs, labels in self.local_loader:
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        logger.debug(
            f"  Client {self.client_id}: {local_epochs} epochs, "
            f"avg loss={avg_loss:.4f}, samples={self.n_samples}"
        )
        return model.state_dict(), avg_loss


# ──────────────────────────────────────────────────────────────────────────────
# FL Server
# ──────────────────────────────────────────────────────────────────────────────

class FLServer:
    """
    Federated Learning server.

    Algorithm 2 (paper):
      1. Initialize global model θ_0 and compute pruning mask M (once)
      2. Broadcast (θ_0, M) to all clients
      3. For each round t:
           - Select C × N clients
           - Collect local models from selected clients
           - FedAvg: θ_{t+1} ← (1/N) Σ θ_{t,j}
           - Broadcast θ_{t+1}
    """

    def __init__(
        self,
        model: nn.Module,
        client_loaders: Dict[int, DataLoader],
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        num_clients: int,
        client_fraction: float = 0.5,
        rounds: int = 5,
        local_epochs: int = 10,
        lr: float = 2e-4,
        experiment_dir: str = "EXPERIMENT",
        sparsity: float = 0.0,
    ) -> None:
        self.global_model    = model
        self.client_loaders  = client_loaders
        self.val_loader      = val_loader
        self.test_loader     = test_loader
        self.device          = device
        self.num_clients     = num_clients
        self.client_fraction = client_fraction
        self.rounds          = rounds
        self.local_epochs    = local_epochs
        self.lr              = lr
        self.experiment_dir  = experiment_dir
        self.sparsity        = sparsity
        self.round_logs: List[dict] = []

        # Compute model size once (for communication cost)
        from models.model_utils import model_size_mb
        self.model_size_mb = model_size_mb(self.global_model) * (1 - sparsity)

        # Build FL clients
        self.clients = {
            cid: FLClient(cid, loader, device)
            for cid, loader in client_loaders.items()
        }

    def select_clients(self, seed: Optional[int] = None) -> List[int]:
        """Select client_fraction × num_clients clients per round."""
        n_selected = max(1, int(self.num_clients * self.client_fraction))
        rng = np.random.default_rng(seed)
        return list(rng.choice(
            list(self.clients.keys()), size=n_selected, replace=False
        ))

    def communication_cost(self, round_num: int, n_selected: int) -> float:
        """
        CommCost formula (paper):
          2 × T × (N × fraction) × size(model) × (1 − sparsity)

        This returns the cumulative cost up to `round_num` in MB.
        """
        return 2 * round_num * n_selected * self.model_size_mb

    def run(self) -> dict:
        """
        Execute the full FL training loop for `self.rounds` rounds.
        Returns the final round logs.
        """
        from training.evaluator import evaluate

        logger.info("=" * 55)
        logger.info("Starting Federated Learning Training")
        logger.info(f"  Rounds: {self.rounds} | Clients: {self.num_clients}")
        logger.info(f"  Fraction/round: {self.client_fraction*100:.0f}%")
        logger.info(f"  Local epochs: {self.local_epochs} | LR: {self.lr}")
        logger.info(f"  Sparsity: {self.sparsity*100:.0f}%")
        logger.info("=" * 55)

        self.global_model.to(self.device)

        for rnd in range(1, self.rounds + 1):
            t_start = time.time()

            selected_ids = self.select_clients(seed=rnd)
            n_selected = len(selected_ids)

            # ── Local training ──
            client_weights = []
            client_sizes   = []
            round_losses   = []

            for cid in selected_ids:
                client = self.clients[cid]
                w, loss = client.local_train(
                    self.global_model,
                    local_epochs=self.local_epochs,
                    lr=self.lr,
                )
                client_weights.append(w)
                client_sizes.append(client.n_samples)
                round_losses.append(loss)

            # ── FedAvg aggregation ──
            agg_weights = fedavg(client_weights, client_sizes)
            self.global_model.load_state_dict(agg_weights, strict=False)

            # ── Evaluation ──
            val_metrics = evaluate(self.global_model, self.val_loader, self.device)
            t_elapsed = time.time() - t_start
            comm_cost = self.communication_cost(rnd, n_selected)

            log = {
                "round":       rnd,
                "train_loss":  float(np.mean(round_losses)),
                "val_acc":     val_metrics["accuracy"],
                "val_f1":      val_metrics["f1"],
                "comm_cost_mb": comm_cost,
                "round_time_s": t_elapsed,
                "n_selected":  n_selected,
            }
            self.round_logs.append(log)

            logger.info(
                f"[Round {rnd:2d}/{self.rounds}] "
                f"Loss={log['train_loss']:.4f} | "
                f"Val Acc={log['val_acc']*100:.2f}% | "
                f"Val F1={log['val_f1']*100:.2f}% | "
                f"CommCost={comm_cost:.2f} MB | "
                f"Time={t_elapsed:.1f}s"
            )

        return self.round_logs

    def test(self) -> dict:
        """Final evaluation on the held-out test set."""
        from training.evaluator import evaluate
        return evaluate(self.global_model, self.test_loader, self.device)


# ──────────────────────────────────────────────────────────────────────────────
# Communication cost utility
# ──────────────────────────────────────────────────────────────────────────────

def compute_total_comm_cost(
    rounds: int,
    num_clients: int,
    client_fraction: float,
    model_size_mb: float,
    sparsity: float,
) -> float:
    """
    Total communication cost (MB) for the entire FL process.
    Formula from the paper:
      CommCost = 2 × T × (N × fraction) × Size(model) × (1 − sparsity)
    """
    n_selected = int(num_clients * client_fraction)
    effective_size = model_size_mb * (1 - sparsity)
    return 2 * rounds * n_selected * effective_size
