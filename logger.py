"""Step 0: shared logger used by every other step.

Every module in the app asks this file for a logger instead of
setting up its own. That way all logs (from document loading to the
final answer) end up in one place, in one consistent format. Use the same setup in every app 
for Robust Observability and Debugging.
"""

import logging
from datetime import datetime
from pathlib import Path

# Ensure LOGS_DIR is a Path object so .mkdir() works
LOGS_DIR = Path("server_logs")
LOGS_DIR.mkdir(exist_ok=True)

# One log file per run, named with the time the run started.
_run_started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_FILE = LOGS_DIR / f"server_{_run_started_at}.log"


def _configure_logging() -> None:
    """Configure logging once so repeated imports do not duplicate handlers."""
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(RUN_LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


_configure_logging()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)