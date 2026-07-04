"""Logging utilities."""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_dir: str = "./logs", level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with file and console handlers.

    Args:
        name: Logger name.
        log_dir: Directory to store log files.
        level: Logging level.

    Returns:
        Configured logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)

    return logger
