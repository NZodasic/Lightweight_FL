"""
logger.py
=========
Structured logging setup for Lightweight-Fed-NIDS experiments.
Creates per-run log files under EXPERIMENT/<run_name>/logs/.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    run_dir: str,
    name: str = "lightweight_fl",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure root logger to write to both console and file.

    Args:
        run_dir: Experiment run directory (logs saved here)
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"train_{timestamp}.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler — force UTF-8 to avoid Windows CP1252 UnicodeEncodeError
    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ch = logging.StreamHandler(utf8_stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Propagate to root so submodule loggers work
    logging.root.setLevel(level)
    for handler in [ch, fh]:
        if handler not in logging.root.handlers:
            logging.root.addHandler(handler)

    logger.info(f"Log file: {log_file}")
    return logger
