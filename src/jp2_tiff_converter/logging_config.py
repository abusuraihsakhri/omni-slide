"""Logging configuration."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from jp2_tiff_converter.config import Config


def setup_logging(config: Config) -> logging.Logger:
    """Configure application logging."""
    logger = logging.getLogger("jp2_tiff_converter")
    logger.setLevel(getattr(logging, config.log_level.upper()))

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if config.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.log_to_file:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = config.log_dir / "converter.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    if name:
        return logging.getLogger(f"jp2_tiff_converter.{name}")
    return logging.getLogger("jp2_tiff_converter")