import os
from datetime import datetime
import logging
import coloredlogs
from colorama import Fore, Style
import verboselogs

verboselogs.install()

CONSOLE_FORMAT = (
    Fore.YELLOW
    + "║"
    + Style.RESET_ALL
    + " %(asctime)s %(name)s %(programname)s » %(levelname)s - %(message)s"
)

FILE_FORMAT = "%(asctime)s %(name)s » %(levelname)s - %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def module_logger(
    disable=False,
    store_log=False,
    log_dir=None,
    level="info",
    reinstall=False,
) -> logging.Logger:

    logger = logging.getLogger("utrcalling")
    logger.propagate = False

    if disable:
        return logger

    if not logger.handlers:
        numeric_level = getattr(logging, level.upper(), None)
        if numeric_level is None:
            raise ValueError(f"Invalid log level: {level}")

        # Let coloredlogs be the sole console handler (no explicit StreamHandler).
        # Its handler filters at the requested level independently.
        coloredlogs.install(
            level=numeric_level,
            logger=logger,
            fmt=CONSOLE_FORMAT,
            field_styles=field_styles,
            level_styles=level_styles,
        )

        if store_log:
            # Override the logger level to DEBUG *after* coloredlogs.install()
            # (which sets logger.level = numeric_level internally).
            # The console handler still filters at numeric_level via its own level.
            logger.setLevel(logging.DEBUG)
            resolved_dir = log_dir if log_dir else os.getcwd()
            os.makedirs(resolved_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%Hh%Mm")
            log_path = os.path.join(resolved_dir, f"{timestamp}-utrcalling.log")
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATEFMT))
            logger.addHandler(fh)

    if reinstall:
        coloredlogs.install(
            level=logger.getEffectiveLevel(),
            logger=logger,
            fmt=CONSOLE_FORMAT,
            field_styles=field_styles,
            level_styles=level_styles,
        )

    return logger


field_styles = {
    "asctime": {"color": "yellow"},
    "hostname": {"color": "cyan"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "magenta"},
    "programname": {"color": "blue"},
    "username": {"color": "green"},
}

level_styles = {
    "critical": {"bold": True, "color": "red"},
    "debug": {"color": "green"},
    "error": {"color": "red"},
    "info": {},
    "notice": {"color": "magenta"},
    "spam": {"color": "green", "faint": True},
    "success": {"bold": True, "color": "green"},
    "verbose": {"color": "blue"},
    "warning": {"color": "yellow"},
}


def log_end_run(message: str) -> None:
    logger = module_logger(reinstall=True)
    logger.success(f"Job ended: {message}")
