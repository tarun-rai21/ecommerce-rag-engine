"""Log format for the API process.

text (default): human-readable lines for local work.
json: one JSON object per line, for a log platform to index (level, message, time,
      module, and `request_id` when the line was written while handling a request).
"""
import sys

from loguru import logger

TEXT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "{name}:{function}:{line} | rid={extra[request_id]} - <level>{message}</level>"
)


def configure_logging(fmt: str = "text", level: str = "INFO") -> None:
    logger.remove()
    logger.configure(extra={"request_id": "-"})  # lines outside a request still format cleanly
    if fmt == "json":
        logger.add(sys.stderr, level=level, serialize=True, backtrace=False, diagnose=False)
    else:
        # diagnose=False: tracebacks don't print local variables, which can hold keys or queries.
        logger.add(sys.stderr, level=level, format=TEXT_FORMAT, diagnose=False)
