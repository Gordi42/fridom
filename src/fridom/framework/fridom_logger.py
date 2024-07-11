from enum import Enum
import logging
import sys
import os
from IPython import get_ipython
import coloredlogs


class LogLevel(Enum):
    """
    Logging Levels:
    ---------------
    10 - DEBUG:
        The lowest logging level. It is only used for debugging purposes.
        Computational expensive debugging information should be logged at this level.
        For debugging purposes: everything is logged.
    15 - VERBOSE:
        The lowest logging level that is not used for debugging. This should
        be used for detailed documentation of what is happening in the code.
    20 - INFO:
        This should be used for general information.
    25 - NOTICE:
        This should be used for the very essential information.
    30 - SILENT:
        No information is logged (but warnings and errors are still logged).
    40 - ERROR:
        No warnings are logged, only errors.
    50 - CRITICAL:
        Only critical errors are logged.
    """
    DEBUG = 10
    VERBOSE = 15
    INFO = 20
    NOTICE = 25
    SILENT = 30
    ERROR = 40
    CRITICAL = 50


class FridomLogger:
    """
    Logging Levels:
    ---------------

    Attributes
    ----------
    `level` : `LogLevel`
        The logging level.
    `rank` : `list[int] | None` (default: `[0]`)
        A list of MPI ranks which should log the information. If None, all ranks log.
    """

    def __init__(self):
        logger = logging.Logger("Fridom Logger")
        # add logging level names
        logging.addLevelName(LogLevel.VERBOSE.value, "VERBOSE")
        logging.addLevelName(LogLevel.NOTICE.value, "NOTICE")

        # logger = logging.getLogger("Fridom Logger")
        console_handler = logging.StreamHandler(stream=sys.stdout)

        # check if the output should be colorized
        if os.isatty(sys.stdout.fileno()):
            colored_output = True  # colors in terminal
        else:
            colored_output = False  # no colors in file
        if get_ipython() is not None:
            colored_output = True  # colors in ipython

        if colored_output:
            formatter = coloredlogs.ColoredFormatter(
                '%(asctime)s: %(message)s', datefmt='%H:%M:%S')
        else:
            formatter = logging.Formatter(
                '%(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self._logger = logger
        self.level = LogLevel.SILENT
        self._is_active: bool
        self.ranks = [0]
        

    def debug(self, message: str, *args, **kwargs) -> None:
        if self._is_active:
            self._logger.debug(message, *args, **kwargs)

    def verbose(self, message: str, *args, **kwargs) -> None:
        if self._is_active:
            self._logger.log(LogLevel.VERBOSE.value, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        if self._is_active:
            self._logger.info(message, *args, **kwargs)

    def notice(self, message: str, *args, **kwargs) -> None:
        if self._is_active:
            self._logger.log(LogLevel.NOTICE.value, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        if self._is_active:
            self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        if self._is_active:
            self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        if self._is_active:
            self._logger.critical(message, *args, **kwargs)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def ranks(self) -> list[int] | None:
        """A list of MPI ranks which should log the information. If None, all ranks log."""
        return self._ranks

    @ranks.setter
    def ranks(self, ranks: list[int] | None) -> None:
        if not isinstance(ranks, list or None):
            raise TypeError("The rank must be a list of integers.")
        try:
            from mpi4py import MPI
            my_rank = MPI.COMM_WORLD.Get_rank()
            if ranks is None:
                self._is_active = True
            else:
                self._is_active = False
                self._is_active = my_rank in ranks
        except ImportError:
            self._is_active = True
        self._ranks = ranks
        return

    @property
    def level(self) -> LogLevel:
        """The logging level."""
        return self._level
    
    @level.setter
    def level(self, level: LogLevel) -> None:
        if not isinstance(level, LogLevel):
            raise TypeError("The logging level must be of type LogLevel.")
        self._logger.setLevel(level.value)
        self._level = level
