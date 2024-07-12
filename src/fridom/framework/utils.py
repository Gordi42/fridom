"""
utils
===
Contains utility functions.

Functions
---------
`print_bar(char='=')`
    Print a bar to the stdout.
`print_job_init_info()`
    Print the job starting time and the number of MPI processes.
"""
from . import config
from .config import logger
from mpi4py import MPI

def print_bar(char='='):
    """
    Print a bar to the log file.

    Parameters
    char: str
        Character to use for the bar.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(char*80, flush=True)

def print_job_init_info():
    """
    Print the job starting time and the number of MPI processes.
    """
    print_bar("#")
    logger.info("FRIDOM: Framework for Idealized Ocean Models")
    # get system time
    from datetime import datetime

    # Get the current system time
    current_time = datetime.now()

    # Format the time according to the given format
    formatted_time = current_time.strftime(" > Job starting on %Y.%m.%d at %I:%M:%S %p")

    logger.info(formatted_time)

    # get the number of MPI processes
    size = MPI.COMM_WORLD.Get_size()
    logger.info(f" > Running on {size} MPI processes.")
    logger.info(f" > Backend: {config.backend}")
    print_bar("#")
    [print_bar(" ") for _ in range(3)]

def humanize_number(value, unit):
    if unit == "meters":
        if value < 1e-2:
            return f"{value*1e3:.2f} mm"
        elif value < 1:
            return f"{value*1e2:.2f} cm"
        elif value < 1e3:
            return f"{value:.2f} m"
        else:
            return f"{value/1e3:.2f} km"
    else:
        raise NotImplementedError(f"Unit '{unit}' not implemented.")

def chdir_to_submit_dir():
    """
    Change the current working directory to the directory where the job was submitted.
    """
    import os
    logger.info("Changing working directory")
    logger.info(f"Old working directory: {os.getcwd()}")
    submit_dir = os.getenv('SLURM_SUBMIT_DIR')
    os.chdir(submit_dir)
    logger.info(f"New working directory: {os.getcwd()}")
    return