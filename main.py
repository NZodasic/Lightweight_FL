"""
main.py
=======
Entry point for Lightweight-Fed-NIDS training and evaluation.

Usage:
  python main.py [options]

Examples:
  python main.py --sparsity 0.0 --num_clients 10 --rounds 5
  python main.py --sparsity 0.9 --num_clients 50 --partition non_iid
  python main.py --run_all_sparsities --num_clients 10
"""

import sys
import io

# ── Force UTF-8 on Windows to avoid CP1252 UnicodeEncodeError ─────────────────
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

# ── Reproducibility (must be before any torch/numpy imports) ──────────────────
from utils.device_utils import set_seed, get_device, suggest_batch_size
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lightweight-Fed-NIDS (Bouayad et al., 2024)"
    )
    # Data
    p.add_argument("--raw_data",    type=str, default="data/USTC-TFC2016",
                   help="Root dir of raw USTC-TFC2016 dataset")
    p.add_argument("--proc_data",   type=str, default="data/processed",
                   help="Where to store pre-processed images")
    p.add_argument("--image_size",  type=int, default=224)

    # FL
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--client_fraction", type=float, default=0.5)
    p.add_argument("--rounds",      type=int, default=5)
    p.add_argument("--local_epochs",type=int, default=10)
    p.add_argument("--partition",   type=str, default="iid",
                   choices=["iid", "non_iid"])
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)

    # Model / Pruning
    p.add_argument("--sparsity",    type=float, default=0.0,
                   help="Pruning sparsity [0.0, 0.5, 0.7, 0.9]")
    p.add_argument("--pretrained",  action="store_true", default=True)
    p.add_argument("--run_all_sparsities", action="store_true",
                   help="Run all sparsity levels: 0.0, 0.5, 0.7, 0.9")

    # Training
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--seed",        type=int,   default=42)

    # Output
    p.add_argument("--exp_dir",     type=str, default="EXPERIMENT")
    p.add_argument("--config",      type=str, default="configs/config.yaml",
                   help="Override defaults from YAML config")

    return p.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config (optional)."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def make_run_dir(exp_dir: str, sparsity: float, num_clients: int, partition: str) -> str:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"resnet50_sp{int(sparsity*100)}_n{num_clients}_{partition}_{timestamp}"
    run_dir   = os.path.join(exp_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"),   exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"),  exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    return run_dir


def run_experiment(
    sparsity:        float,
    num_clients:     int,
    args:            argparse.Namespace,
    client_loaders,
    val_loader,
    test_loader,
    device:          torch.device,
    run_dir:         str,
) -> dict:
    """Single experiment: build model → prune → FL train → evaluate."""
    import copy
    from models.resnet50     import build_resnet50
    from models.model_utils  import print_model_summary, model_size_mb
    from pruning.baseline    import prune_model, compute_pruning_stats
    from training.trainer    import FLServer, compute_total_comm_cost
    from training.evaluator  import (
        evaluate, print_metrics,
        plot_confusion_matrix, plot_roc_curve, plot_training_curves,
    )
    from utils.visualization import plot_fl_metrics_summary

    logger = __import__("logging").getLogger("lightweight_fl")

    # ── Build & prune model ──────────────────────────────────────────────────
    logger.info(f"\n{'='*55}")
    logger.info(f"Experiment: sparsity={sparsity*100:.0f}%, clients={num_clients}")
    logger.info(f"{'='*55}")

    model = build_resnet50(num_classes=2, pretrained=args.pretrained)
    orig_size = model_size_mb(model)

    dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
    pruned = prune_model(model, sparsity=sparsity, example_inputs=dummy_input)

    stats = compute_pruning_stats(model, pruned, device=str(device))

    # Print experimental setup
    print("\n" + "=" * 55)
    print("Experimental Setup")
    print("=" * 55)
    print(f"  Framework     : PyTorch")
    print(f"  Device        : {device}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Local epochs  : {args.local_epochs}")
    print(f"  Learning rate : {args.lr}")
    print(f"  Optimizer     : Adam")
    print()
    print("  Federated settings:")
    print(f"    Rounds            : {args.rounds}")
    print(f"    Clients per round : {int(num_clients * args.client_fraction)}")
    print(f"    Total clients     : {num_clients}")
    print(f"    Aggregation       : FedAvg")
    print()
    print(f"  Model: ResNet-50")
    print(f"  Pruning sparsity      : {sparsity*100:.0f}%")
    print(f"  Remaining parameters  : {stats['pruned_params']:,}")
    print(f"  Original model size   : {stats['orig_size_mb']:.2f} MB")
    print(f"  Pruned model size     : {stats['pruned_size_mb']:.2f} MB")
    print("=" * 55 + "\n")

    print_model_summary(pruned, sparsity=sparsity, device=device)

    # ── FL Training ──────────────────────────────────────────────────────────
    server = FLServer(
        model            = pruned,
        client_loaders   = client_loaders,
        val_loader       = val_loader,
        test_loader      = test_loader,
        device           = device,
        num_clients      = num_clients,
        client_fraction  = args.client_fraction,
        rounds           = args.rounds,
        local_epochs     = args.local_epochs,
        lr               = args.lr,
        experiment_dir   = run_dir,
        sparsity         = sparsity,
    )

    t0 = time.time()
    round_logs = server.run()
    train_time = time.time() - t0

    # ── Final evaluation ─────────────────────────────────────────────────────
    test_metrics = server.test()
    print_metrics(test_metrics, title=f"Test Results (Sparsity={sparsity*100:.0f}%)")

    total_comm = compute_total_comm_cost(
        args.rounds, num_clients, args.client_fraction,
        orig_size, sparsity
    )

    print(f"\n  Training time (total) : {train_time:.1f} s")
    print(f"  Total comm. cost      : {total_comm:.2f} MB\n")

    # ── Save plots ───────────────────────────────────────────────────────────
    plots_dir   = os.path.join(run_dir, "plots")
    suffix      = f" (sp={int(sparsity*100)}%, N={num_clients})"

    plot_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"],
        save_path=os.path.join(plots_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix{suffix}",
    )
    plot_roc_curve(
        test_metrics["y_true"], test_metrics["y_probs"],
        save_path=os.path.join(plots_dir, "roc_curve.png"),
        label=f"ResNet-50 (sp={int(sparsity*100)}%)",
    )
    plot_training_curves(
        round_logs,
        save_path=os.path.join(plots_dir, "training_curves.png"),
        title_suffix=suffix,
    )
    plot_fl_metrics_summary(
        round_logs,
        save_path=os.path.join(plots_dir, "fl_metrics_summary.png"),
        title_suffix=suffix,
    )

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = os.path.join(run_dir, "models", "global_model_final.pt")
    torch.save(pruned.state_dict(), model_path)
    logger.info(f"Final model saved -> {model_path}")

    return {
        "sparsity":    sparsity,
        "num_clients": num_clients,
        "accuracy":    test_metrics["accuracy"],
        "f1":          test_metrics["f1"],
        "auc":         test_metrics["auc"],
        "train_time":  train_time,
        "comm_cost_mb": total_comm,
        "run_dir":     run_dir,
        **stats,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # ── Override batch size based on device ──────────────────────────────────
    if args.batch_size == 32:           # if still at default
        args.batch_size = suggest_batch_size(device)

    # ── Temporary early run_dir for logger ───────────────────────────────────
    tmp_run_dir = make_run_dir(args.exp_dir, args.sparsity, args.num_clients, args.partition)
    logger = setup_logger(tmp_run_dir)

    logger.info("=" * 55)
    logger.info("Lightweight-Fed-NIDS (Bouayad et al., 2024)")
    logger.info(f"Dataset : USTC-TFC2016")
    logger.info(f"Model   : ResNet-50")
    logger.info("=" * 55)

    # ── Data loading (done once, shared across experiments) ──────────────────
    from data.data_loader import build_dataset
    from training.evaluator import log_client_stats

    client_loaders, val_loader, test_loader, full_dataset = build_dataset(
        raw_root         = args.raw_data,
        processed_root   = args.proc_data,
        image_size       = args.image_size,
        batch_size       = args.batch_size,
        num_clients      = args.num_clients,
        partition        = args.partition,
        dirichlet_alpha  = args.dirichlet_alpha,
        seed             = args.seed,
    )
    log_client_stats(client_loaders)

    # ── Run experiments ───────────────────────────────────────────────────────
    sparsity_list = [0.0, 0.5, 0.7, 0.9] if args.run_all_sparsities else [args.sparsity]

    all_results = []
    for sp in sparsity_list:
        run_dir = make_run_dir(args.exp_dir, sp, args.num_clients, args.partition)
        setup_logger(run_dir)   # re-point log file

        result = run_experiment(
            sparsity       = sp,
            num_clients    = args.num_clients,
            args           = args,
            client_loaders = client_loaders,
            val_loader     = val_loader,
            test_loader    = test_loader,
            device         = device,
            run_dir        = run_dir,
        )
        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Sparsity':>10} {'Accuracy':>10} {'F1':>10} {'AUC':>8} "
          f"{'CommCost(MB)':>14} {'Size(MB)':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"  {r['sparsity']*100:>6.0f}%  "
              f"  {r['accuracy']*100:>8.2f}%"
              f"  {r['f1']*100:>8.2f}%"
              f"  {r['auc']:>6.4f}"
              f"  {r['comm_cost_mb']:>12.2f}"
              f"  {r['pruned_size_mb']:>8.2f}")
    print("=" * 70 + "\n")

    # ── Sparsity comparison plot ──────────────────────────────────────────────
    if len(all_results) > 1:
        from utils.visualization import plot_sparsity_comparison
        sp_results = {
            f"{int(r['sparsity']*100)}%": r for r in all_results
        }
        plot_sparsity_comparison(
            sp_results,
            save_path=os.path.join(args.exp_dir, "sparsity_comparison.png"),
        )


if __name__ == "__main__":
    main()
