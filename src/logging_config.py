"""Shared logging setup: one timestamped format for every module's
progress/status output, replacing ad hoc print() calls.

configure_logging() is called at import time by every module that logs
(main.py, runner.py) rather than left to whichever happens to import first,
because runner.py's worker functions run inside separate processes under
ProcessPoolExecutor (Runner._run_parallel): each spawned process re-imports
runner.py fresh and would otherwise have no handler configured (Python's
"handler of last resort" only surfaces WARNING+ through an unformatted,
un-timestamped fallback). logging.basicConfig is a no-op once handlers
exist, so calling this from multiple modules is safe regardless of import
order.
"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
