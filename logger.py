"""Shared, idempotent logging for the complete analysis run."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Iterator
from uuid import uuid4


LOGS_DIR = Path(os.getenv("LOGS_DIR", "app_logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_run_started_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
RUN_LOG_FILE = LOGS_DIR / f"logs_{_run_started_at}.log"
_HANDLER_MARKER = "log_analysis_multi_agent_rag_handler"
_LOG_FORMAT = "%(asctime)s | %(levelname)s | run_id=%(run_id)s | %(name)s | %(message)s"
_run_id: ContextVar[str] = ContextVar("run_id", default="startup")


class _RunContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        return True


def _configure_logging() -> None:
    """Install this app's handlers without disturbing host application handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    if not any(getattr(handler, _HANDLER_MARKER, None) == "file" for handler in root_logger.handlers):
        file_handler = logging.FileHandler(RUN_LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        file_handler.addFilter(_RunContextFilter())
        setattr(file_handler, _HANDLER_MARKER, "file")
        root_logger.addHandler(file_handler)

    if not any(
        getattr(handler, _HANDLER_MARKER, None) == "stream"
        for handler in root_logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        stream_handler.addFilter(_RunContextFilter())
        setattr(stream_handler, _HANDLER_MARKER, "stream")
        root_logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for a module or application component."""
    _configure_logging()
    return logging.getLogger(name)


def fingerprint(value: object) -> str:
    """Return a short stable identifier without logging the original value."""
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()[:12]


def loggable_text(value: object, setting: str = "LOG_QUERY_CONTENT") -> str:
    """Return text only when explicitly enabled; otherwise return its fingerprint."""
    if os.getenv(setting, "false").lower() in {"1", "true", "yes"}:
        return str(value)
    return f"<redacted:{fingerprint(value)}>"


def invoke_with_logging(logger: logging.Logger, chain: object, operation: str, payload: object) -> object:
    """Invoke a chain once and log its input/output at a bounded detail level."""
    started_at = perf_counter()
    logger.info(
        "stage=llm event=invoke_start operation=%s input=%r",
        operation,
        _summarize_payload(payload),
    )
    try:
        result = chain.invoke(payload)
    except Exception:
        logger.exception(
            "stage=llm event=invoke_failed operation=%s elapsed_ms=%.1f",
            operation,
            (perf_counter() - started_at) * 1000,
        )
        raise
    logger.info(
        "stage=llm event=invoke_complete operation=%s elapsed_ms=%.1f output=%r",
        operation,
        (perf_counter() - started_at) * 1000,
        _summarize_value(result),
    )
    return result


def _summarize_payload(payload: object) -> object:
    if not isinstance(payload, dict):
        return _summarize_value(payload)
    return {key: _summarize_value(value, key) for key, value in payload.items()}


def _summarize_value(value: object, field_name: str = "") -> object:
    if isinstance(value, (list, tuple)):
        return [_summarize_value(item, field_name) for item in value]
    if isinstance(value, dict):
        return {key: _summarize_value(item, key) for key, item in value.items()}
    if hasattr(value, "page_content"):
        content = str(getattr(value, "page_content", ""))
        return {
            "chunk_id": fingerprint(content),
            "characters": len(content),
            "content": _content_preview(content),
        }
    if isinstance(value, str):
        return {
            "characters": len(value),
            "sha256": fingerprint(value),
            "content": _content_preview(value, field_name),
        }
    return value


def _content_preview(value: str, field_name: str = "") -> str:
    show_full = os.getenv("LOG_LLM_PAYLOADS", "false").lower() in {"1", "true", "yes"}
    if not show_full:
        return f"<redacted:{fingerprint(value)}>"
    limit = int(os.getenv("LOG_LLM_PREVIEW_CHARS", "1000"))
    return value[:limit].replace("\n", "\\n")


@contextmanager
def log_run(logger: logging.Logger, **details: object) -> Iterator[str]:
    """Correlate and time all log records emitted during one analysis run."""
    run_id = uuid4().hex[:12]
    token = _run_id.set(run_id)
    started_at = perf_counter()
    logger.info("stage=analysis event=run_started %s", _format_details(details))
    try:
        yield run_id
    except Exception:
        logger.exception("stage=analysis event=run_failed elapsed_ms=%.1f", (perf_counter() - started_at) * 1000)
        raise
    else:
        logger.info("stage=analysis event=run_completed elapsed_ms=%.1f", (perf_counter() - started_at) * 1000)
    finally:
        _run_id.reset(token)


def _format_details(details: dict[str, object]) -> str:
    return " ".join(f"{key}={value!r}" for key, value in details.items())


@contextmanager
def log_stage(logger: logging.Logger, stage: str, **details: object) -> Iterator[None]:
    """Log stage start, completion and failures with elapsed time."""
    started_at = perf_counter()
    suffix = " " + _format_details(details) if details else ""
    logger.info("stage=%s event=start%s", stage, suffix)
    try:
        yield
    except Exception:
        logger.exception("stage=%s event=failed elapsed_ms=%.1f", stage, (perf_counter() - started_at) * 1000)
        raise
    else:
        logger.info("stage=%s event=complete elapsed_ms=%.1f", stage, (perf_counter() - started_at) * 1000)


_configure_logging()