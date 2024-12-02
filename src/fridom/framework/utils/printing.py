"""printing.py: Utilities for printing to the console."""
import datetime
import fridom.framework as fr

def print_bar(char='='):
    """
    Print a bar to the log file.

    Parameters
    ----------
    `char`: `str`
        Character to use for the bar.
    """
    if fr.utils.I_AM_MAIN_RANK:
        print(char*80, flush=True)

def print_job_init_info():
    """
    Print the job starting time and the number of MPI processes.
    """
    print_bar("#")
    fr.log.info("FRIDOM: Framework for Idealized Ocean Models")
    # Get the current system time
    current_time = datetime.datetime.now()

    # Format the time according to the given format
    formatted_time = current_time.strftime(" > Job starting on %Y.%m.%d at %I:%M:%S %p")

    fr.log.info(formatted_time)

    # get the number of MPI processes
    if fr.utils.MPI_AVAILABLE:
        size = fr.utils.get_mpi_size()
        fr.log.info(" > Running on %d MPI processes.", size)
    fr.log.info(" > Backend: %s", fr.config.backend)
    print_bar("#")
    _ = [print_bar(" ") for _ in range(3)]
