"""mpi.py - MPI utilities for Fridom framework."""
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# check if MPI is available
MPI_AVAILABLE = MPI is not None

# Check if the current rank is the main rank
def am_i_main_rank():
    """
    Check if the current rank is the main rank.
    
    Returns
    -------
    `bool`
        True if the current rank is the main rank, False otherwise.
    """
    i_am_main_rank = False
    if MPI_AVAILABLE:
        i_am_main_rank = MPI.COMM_WORLD
    else:
        # if no MPI is available, assume that the current rank is the main rank
        i_am_main_rank = True
    return i_am_main_rank

I_AM_MAIN_RANK = am_i_main_rank()

def mpi_barrier():
    """
    Barrier synchronization for MPI.
    """
    if MPI_AVAILABLE:
        MPI.COMM_WORLD.Barrier()

def get_mpi_size():
    """
    Get the number of MPI processes.
    
    Returns
    -------
    `int`
        The number of MPI processes.
    """
    if MPI_AVAILABLE:
        return MPI.COMM_WORLD.Get_size()
    return 1
