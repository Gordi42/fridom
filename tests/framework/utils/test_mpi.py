"""Tests for the MPI utilities."""
from unittest.mock import MagicMock

import fridom.framework as fr
import fridom.framework.utils.mpi as mpi_module


# ================================================================
#  Without MPI (jax fallbacks)
# ================================================================
def test_without_mpi(monkeypatch):
    monkeypatch.setattr(mpi_module, "MPI_AVAILABLE", False)

    assert mpi_module.am_i_main_rank()
    assert mpi_module.get_mpi_size() == 1
    assert mpi_module.get_my_rank() == 0
    # the barrier is a no-op on a single process
    fr.utils.mpi_barrier()


# ================================================================
#  With a mocked MPI
# ================================================================
def test_with_mocked_mpi(monkeypatch):
    mpi = MagicMock()
    mpi.COMM_WORLD.Get_rank.return_value = 3
    mpi.COMM_WORLD.Get_size.return_value = 8
    monkeypatch.setattr(mpi_module, "MPI", mpi)
    monkeypatch.setattr(mpi_module, "MPI_AVAILABLE", True)

    assert not mpi_module.am_i_main_rank()
    assert mpi_module.get_my_rank() == 3
    assert mpi_module.get_mpi_size() == 8

    mpi_module.mpi_barrier()
    assert mpi.COMM_WORLD.Barrier.called


def test_with_mocked_main_rank(monkeypatch):
    mpi = MagicMock()
    mpi.COMM_WORLD.Get_rank.return_value = 0
    monkeypatch.setattr(mpi_module, "MPI", mpi)
    monkeypatch.setattr(mpi_module, "MPI_AVAILABLE", True)

    assert mpi_module.am_i_main_rank()
