"""
utils
===
Contains utility functions.

Functions
---------
`print_main(msg, flush=False)`
    Print message only from rank 0.
`print_job_init_info()`
    Print the job starting time and the number of MPI processes.
"""
from mpi4py import MPI

def print_main(msg, flush=True):
    """
    Print message only from rank 0.
    
    Parameters
    ----------
    `msg` : `str`
        The message to print.
    `flush` : `bool`
        Flush the output
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(msg, flush=flush)
    return

def print_job_init_info():
    """
    Print the job starting time and the number of MPI processes.
    """
    print_main("=========================================================")
    print_main("FRIDOM: Framework for Idealized Ocean Models")
    # get system time
    from datetime import datetime

    # Get the current system time
    current_time = datetime.now()

    # Format the time according to the given format
    formatted_time = current_time.strftime(" > Job starting on %Y.%m.%d at %I:%M:%S %p")

    print_main(formatted_time, flush=True)

    # get the number of MPI processes
    size = MPI.COMM_WORLD.Get_size()
    print_main(f" > Running on {size} MPI processes.", flush=True)
    print_main("=========================================================")

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