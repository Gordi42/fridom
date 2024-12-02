"""filesystem.py: Utility functions for file system operations."""
import os
import sys
from IPython import get_ipython
import fridom.framework as fr

def chdir_to_submit_dir():
    """
    Change the current working directory to the directory where the job was submitted.
    """
    fr.log.info("Changing working directory")
    fr.log.info("Old working directory: %s", os.getcwd())
    submit_dir = os.getenv('SLURM_SUBMIT_DIR')
    os.chdir(submit_dir)
    fr.log.info("New working directory: %s", os.getcwd())

def stdout_is_file():
    """Check if the standard output is a file."""
    # check if the output is not a file
    if os.isatty(sys.stdout.fileno()):
        res = False  # output is a terminal
    else:
        res = True   # output is a file

    # check if the output is ipython
    if get_ipython() is not None:
        res = False  # output is ipython
    return res
