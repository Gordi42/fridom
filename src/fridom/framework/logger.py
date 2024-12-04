"""logger.py - Logging module for the fridom framework."""

import logging
import os
import sys

import coloredlogs
from IPython import get_ipython

log = logging.getLogger("fridom")

# We add three new log levels
logging.addLevelName(15, "VERBOSE")
logging.addLevelName(25, "NOTICE")
logging.addLevelName(30, "SILENT")

console_handler = logging.StreamHandler(stream=sys.stdout)

# check if the output should be colorized
def should_colorize() -> bool:
    """Check if the output should be colorized."""
    colored_output = os.isatty(sys.stdout.fileno())
    if get_ipython() is not None:
        colored_output = True  # colors in ipython
    return colored_output

if should_colorize():
    formatter = coloredlogs.ColoredFormatter(
        "%(asctime)s: %(message)s", datefmt="%H:%M:%S")
else:
    formatter = logging.Formatter(
        "%(asctime)s: %(levelname)s: %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

log.setLevel("NOTICE")

# add verbose and notice methods
# I haven't found any other way than accessing the private _log method
# here, which is not recommended. Nevertheless, we disable the lint
# warning here.
def verbose(message: str, *args: any, **kwargs: dict) -> None:
    """Log a message with level VERBOSE."""
    if log.isEnabledFor(15):
        log._log(15, message, args, **kwargs)  # noqa: SLF001

def notice(message: str, *args: any, **kwargs: dict) -> None:
    """Log a message with level NOTICE."""
    if log.isEnabledFor(25):
        log._log(25, message, args, **kwargs)  # noqa: SLF001

log.verbose = verbose
log.notice = notice
