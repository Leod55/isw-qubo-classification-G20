# src/qubo_project/utils.py
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

def setup_logger(name: str, log_file: str = "logs/pipeline.log") -> logging.Logger:
    """Configure a logger with both console and file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers if called multiple times
        return logger

    logger.setLevel(logging.INFO)

    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)

    return logger

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Safely write a dictionary to a JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load a JSON file into a dict."""
    with open(filepath, 'r') as f:
        return json.load(f)