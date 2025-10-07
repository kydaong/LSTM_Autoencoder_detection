# Created by aaronkueh on 9/19/2025
# aom/utils/logger.py
from __future__ import annotations
import logging
import logging.handlers as _handlers
import os
from pathlib import Path
from typing import Optional

# Defaults are environment-overridable
_DEFAULT_APP = os.getenv("APP_NAME", "agent-dev")
_DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_DEFAULT_DIR = os.getenv("LOG_DIR", "")  # if empty we'll compute project_root/logs
_DEFAULT_JSON = os.getenv("LOG_JSON", "false").lower() == "true"
_DEFAULT_ROTATION = os.getenv("LOG_ROTATION", "time")  # "time" or "size"
_DEFAULT_RETENTION = int(os.getenv("LOG_RETENTION_DAYS", "7"))
_DEFAULT_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10MB


def _project_root() -> Path:
    # aom/utils/logger.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _logs_dir() -> Path:
    base = Path(_DEFAULT_DIR) if _DEFAULT_DIR else _project_root() / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _make_formatter(json: bool) -> logging.Formatter:
    if not json:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        dft = "%Y-%m-%d %H:%M:%S%z"
        return logging.Formatter(fmt=fmt, datefmt=dft)
    # lightweight JSON without extra deps
    fmt = (
        '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
        '"msg":"%(message)s","module":"%(module)s","pid":%(process)d,"tid":"%(threadName)s"}'
    )
    dft = "%Y-%m-%dT%H:%M:%S%z"
    return logging.Formatter(fmt=fmt, datefmt=dft)


def init_logging(
    app_name: Optional[str] = 'aom',
    level: Optional[str] = None,
    json: Optional[bool] = None,
) -> None:
    """
    Initialize root logging once (idempotent). Call early in main entrypoint.
    """
    app = app_name or _DEFAULT_APP
    lvl = getattr(logging, (level or _DEFAULT_LEVEL).upper(), logging.INFO)
    use_json = _DEFAULT_JSON if json is None else json

    if getattr(init_logging, "_configured", False):
        return

    logs_dir = _logs_dir()
    log_file = logs_dir / f"{app}.log"
    formatter = _make_formatter(use_json)

    # Console
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.setFormatter(formatter)

    # File (rotate by day or size)
    if _DEFAULT_ROTATION == "size":
        file_handler = _handlers.RotatingFileHandler(
            log_file, maxBytes=_DEFAULT_MAX_BYTES, backupCount=5, encoding="utf-8"
        )
    else:
        file_handler = _handlers.TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=_DEFAULT_RETENTION, encoding="utf-8", utc=False
        )
    file_handler.setLevel(lvl)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(lvl)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Reduce noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    init_logging._configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """
    Module-level getter. Use in every module: log = get_logger(__name__)
    """
    if not getattr(init_logging, "_configured", False):
        init_logging()  # safe default init if caller forgot
    return logging.getLogger(name)
