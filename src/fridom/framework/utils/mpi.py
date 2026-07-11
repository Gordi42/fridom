"""MPI utilities for Fridom framework."""
from __future__ import annotations

import jax
from jax.experimental import multihost_utils

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# check if MPI is available
MPI_AVAILABLE = MPI is not None

# Check if the current rank is the main rank
def am_i_main_rank() -> bool:
    """
    Check if the current rank is the main rank.

    Returns
    -------
    `bool`
        True if the current rank is the main rank, False otherwise.
    """
    if MPI_AVAILABLE:
        return MPI.COMM_WORLD.Get_rank() == 0
    return jax.process_index() == 0

I_AM_MAIN_RANK = am_i_main_rank()

def mpi_barrier() -> None:
    """Barrier synchronization for MPI."""
    if MPI_AVAILABLE:
        MPI.COMM_WORLD.Barrier()
    multihost_utils.sync_global_devices("mpi_barrier")

def get_mpi_size() -> int:
    """
    Get the number of MPI processes.

    Returns
    -------
    `int`
        The number of MPI processes.
    """
    if MPI_AVAILABLE:
        return MPI.COMM_WORLD.Get_size()
    return jax.process_count()

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
    return jax.process_index()
