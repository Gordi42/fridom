"""Utility functions for file system operations."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from IPython import get_ipython

import fridom.framework as fr


def chdir_to_submit_dir() -> None:
    """Change the working directory to the job submission directory."""
    fr.log.info("Changing working directory")
    fr.log.info("Old working directory: %s", Path.cwd())
    submit_dir = os.getenv("SLURM_SUBMIT_DIR")
    os.chdir(submit_dir)
    fr.log.info("New working directory: %s", Path.cwd())

def stdout_is_file() -> bool:
    """Check if the standard output is a file."""
    # the output is a file if it is not a terminal
    res = not os.isatty(sys.stdout.fileno())

    # check if the output is ipython
    if get_ipython() is not None:
        res = False  # output is ipython
    return res
