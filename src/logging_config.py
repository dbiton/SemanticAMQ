"""Shared logging setup: one timestamped format for every module's
progress/status output, replacing ad hoc print() calls. Logs go to both the
console and LOG_FILE.

configure_logging() is called at import time by every module that logs
(main.py, runner.py) rather than left to whichever happens to import first,
because runner.py's worker functions run inside separate processes under
ProcessPoolExecutor (Runner._run_parallel): each spawned process re-imports
runner.py fresh and would otherwise have no handler configured (Python's
"handler of last resort" only surfaces WARNING+ through an unformatted,
un-timestamped fallback). logging.basicConfig is a no-op once handlers
exist, so calling this from multiple modules is safe regardless of import
order.

LOG_FILE is cleared once per run, not once per call: every spawned worker
process re-runs this function too, and truncating on each of those would
wipe out lines other processes already wrote. _CLEAR_ENV_FLAG marks the
clear as done in os.environ, which child processes inherit, so only the
first call in the whole process tree truncates; every call (including the
first) opens the file in append mode.
"""

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
