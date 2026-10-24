"""Structured logging for RobotStudio.

All modules log through ``get_logger`` so output format stays uniform.
UI and CLI can attach extra handlers to the ``robotstudio`` root logger.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_ROOT_NAME = "robotstudio"
_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-28s | %(message)s"
_datefmt = "%H:%M:%S"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``robotstudio`` root, configuring it once."""
    global _configured
    root = logging.getLogger(_ROOT_NAME)
    if not _configured:
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_datefmt))
        root.addHandler(handler)
        root.propagate = False
        _configured = True
    if name and not name.startswith(_ROOT_NAME):
        name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(name)


def add_file_handler(log_dir: Path) -> Path:
    """Mirror all log records into ``log_dir`` with a timestamped filename."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"robotstudio_{stamp}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_datefmt))
    logging.getLogger(_ROOT_NAME).addHandler(handler)
    return path
