"""Utilities for printing to the console."""
from __future__ import annotations

import datetime

import jax

import fridom.framework as fr


def print_bar(char: str = "=") -> None:
    """
    Print a bar to the log file.

    Parameters
    ----------
    char: str
        Character to use for the bar.
    """
    if fr.utils.I_AM_MAIN_RANK:
        fr.log.info(char*80)

def print_job_init_info() -> None:
    """Print the job starting time and the number of MPI processes."""
    print_bar("#")
    fr.log.info("FRIDOM: Framework for Idealized Ocean Models")
    # Get the current system time (in the local timezone)
    current_time = datetime.datetime.now(tz=datetime.UTC).astimezone()

    # Format the time according to the given format
    formatted_time = current_time.strftime(
        " > Job starting on %Y.%m.%d at %I:%M:%S %p")

    fr.log.info(formatted_time)

    # get the number of MPI processes
    size = fr.utils.get_mpi_size()
    fr.log.info(" > Running on %d processes.", size)
    fr.log.info(" > Platform: %s (%d device(s))",
                jax.default_backend(), jax.device_count())
    print_bar("#")
    _ = [print_bar(" ") for _ in range(3)]

