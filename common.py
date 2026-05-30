"""
src/utils/common.py
-------------------
Shared helpers: config loader, logger, DB engine.
"""

import io
import logging
import os
import sys
from pathlib import Path

import yaml
from sqlalchemy import create_engine


def load_config(path: str = "config/config.yaml") -> dict:
    root = Path(__file__).resolve().parents[2]
    with open(root / path) as f:
        return yaml.safe_load(f)


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    root     = Path(__file__).resolve().parents[2]
    log_path = root / log_dir
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — UTF-8 safe on Windows
    stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
        if hasattr(sys.stdout, "buffer") else sys.stdout
    ch = logging.StreamHandler(stream)
    ch.setFormatter(fmt)

    # File handler — always UTF-8
    fh = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_engine(cfg: dict):
    password = os.environ.get("DB_PASSWORD")
    if not password:
        raise EnvironmentError("DB_PASSWORD environment variable is not set.")

    db  = cfg["database"]
    url = (
        f"postgresql://{db['user']}:{password}"
        f"@{db['host']}:{db['port']}/{db['name']}"
    )
    return create_engine(url)
