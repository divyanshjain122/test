"""
Logging utilities for JSF-Core.

Provides a consistent logging interface across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
    detailed: bool = False,
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to log to console
        detailed: Whether to use detailed format with file/line info

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Choose format
    fmt = DETAILED_FORMAT if detailed else DEFAULT_FORMAT
    formatter = logging.Formatter(fmt)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default settings.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger


def set_global_log_level(level: int) -> None:
    """
    Set log level for all JSF loggers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    jsf_loggers = [
        name for name in logging.root.manager.loggerDict if name.startswith("jsf")
    ]
    for logger_name in jsf_loggers:
        logging.getLogger(logger_name).setLevel(level)


def create_run_logger(run_name: str, log_dir: Path) -> logging.Logger:
    """
    Create a logger for a specific experiment run.

    Args:
        run_name: Name of the experiment run
        log_dir: Directory to store log files

    Returns:
        Logger configured for this run
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{timestamp}.log"
    return setup_logger(
        f"jsf.run.{run_name}",
        level=logging.DEBUG,
        log_file=log_file,
        console=True,
        detailed=True,
    )


# Module-level logger
logger = get_logger(__name__)
