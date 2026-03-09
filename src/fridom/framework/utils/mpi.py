"""mpi.py - MPI utilities for Fridom framework."""
import fridom.framework as fr
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
    if MPI_AVAILABLE:
        return MPI.COMM_WORLD.Get_rank() == 0
    if fr.config.backend_is_jax:
        import jax
        return jax.process_index() == 0

    # if no MPI is available, assume that the current rank is the main rank
    return True

I_AM_MAIN_RANK = am_i_main_rank()

def mpi_barrier():
    """
    Barrier synchronization for MPI.
    """
    if MPI_AVAILABLE:
        MPI.COMM_WORLD.Barrier()
    if fr.config.backend_is_jax:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("mpi_barrier")

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
    if fr.config.backend_is_jax:
        import jax
        return jax.process_count()
    return 1

def get_my_rank() -> int:
    """
    Get the rank of the current MPI process.

    Returns
    -------
    int
        The rank of the current MPI process.

    """
    if MPI_AVAILABLE:
        return MPI.COMM_WORLD.Get_rank()
    if fr.config.backend_is_jax:
        import jax
        return jax.process_index()
    return 0
