"""
Centralized logging configuration for foundation layer

Provides standardized logging setup with:
- Structured logging format
- Performance monitoring
- Error tracking
- Debug capabilities
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get configured logger for foundation components

    Args:
        name: Logger name (typically __name__)
        level: Log level override (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Create handler
        handler = logging.StreamHandler(sys.stdout)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Set level
        log_level = level or logging.INFO
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())
        logger.setLevel(log_level)

    return logger