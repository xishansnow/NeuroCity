"""
Logging utilities for Grid-NeRF

Simple logging setup without circular dependencies.
"""

import logging
from pathlib import Path
from typing import Optional, Union


def setup_logging(log_file: Optional[str | Path] = None, level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ) 