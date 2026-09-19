"""Shared logging setup: one timestamped format for console + LOG_FILE.

Called at import time by every module that logs, since each
ProcessPoolExecutor worker re-imports runner.py fresh and would otherwise
have no handler configured; logging.basicConfig is a no-op once handlers
exist, so this is safe regardless of import order.

LOG_FILE is truncated once per whole process tree, not once per call:
_CLEAR_ENV_FLAG (inherited by child processes via os.environ) ensures only
the first call clears it, so spawned workers don't wipe out lines other
processes already wrote."""

import logging
import os

LOG_FILE = "log.txt"
_CLEAR_ENV_FLAG = "SEMANTICAMQ_LOG_CLEARED"


def configure_logging(level: int = logging.INFO) -> None:
    if _CLEAR_ENV_FLAG not in os.environ:
        open(LOG_FILE, "w").close()
        os.environ[_CLEAR_ENV_FLAG] = "1"

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        ],
    )
