"""The utrcalling logger: messages to the console and, optionally, to a log file.

In a terminal, the console shows the messages with rich: their time, their
level in colour, and the message. Elsewhere, such as in a workflow's log or a
pipe, it writes plain lines that do not wrap. The log file has plain lines,
with every message down to debug.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# The lines of the log file, and of the console outside a terminal.
FILE_FORMAT = "%(asctime)s %(name)s » %(levelname)s - %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def module_logger(
    disable: bool = False,
    store_log: bool = False,
    log_dir: str | None = None,
    level: str = "info",
) -> logging.Logger:
    """The logger "utrcalling", set up the first time.

    Later calls return it as it is: joblib's worker processes call this too.

    Args:
        disable: Return the logger without setting it up.
        store_log: Also write every message, down to debug, to a log file
            named after the time, in *log_dir*.
        log_dir: Directory of the log file; by default the working directory.
        level: The lowest level the console shows: debug, info, warning or error.

    Raises:
        ValueError: If *level* is not a level.
    """
    logger = logging.getLogger("utrcalling")
    logger.propagate = False
    if disable or logger.handlers:
        return logger

    console_level = logging.getLevelName(level.upper())
    if not isinstance(console_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logger.addHandler(_console_handler(console_level))
    logger.setLevel(console_level)
    if store_log:
        # The file has every message; the console filters at its own level.
        logger.setLevel(logging.DEBUG)
        directory = Path(log_dir) if log_dir else Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%Hh%Mm")
        file_handler = logging.FileHandler(directory / f"{timestamp}-utrcalling.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATEFMT))
        logger.addHandler(file_handler)
    return logger


def _console_handler(level: int) -> logging.Handler:
    """Rich lines in a terminal, plain ones elsewhere; both on standard error."""
    handler: logging.Handler
    if sys.stderr.isatty():
        handler = RichHandler(
            console=Console(stderr=True), show_path=False, markup=False, log_time_format="[%X]"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATEFMT))
    handler.setLevel(level)
    return handler


def log_end_run(message: str) -> None:
    """Log that a run ended."""
    module_logger().info(f"Job ended: {message}")
