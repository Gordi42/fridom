"""
Kernel-level halo-validity claims (task 1.8, stage B).

Description
-----------
The construction seams stamp the result's ghost validity where the
knowledge lives: the staggered/aligned window tails claim
``valid_in - reach`` on the applied axis (they compute every output
ghost slot their window reaches), the storage-frame elementwise ops
claim the pointwise minimum of their operands, and everything
rebuilt through ``store`` claims zero. These tests exercise the
kernel seams directly (``_apply``): while the iteration-1
post-application sync is still live, the public surface overwrites
the stamps with the negotiated widths — the flip is stage C
(``design/plans/done/sync_redo_plan.md``).
"""
import pytest

from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.operators.products import (
    CollocationProduct,
    Divide,
)


@pytest.fixture
def grid():
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    my = IntervalMesh(16, (0.0, 2.0), name="y")
    return Grid((mx, my))


@pytest.fixture
def synced(grid):
    # full validity: the negotiated widths on both names
    return grid.sync(grid.create_field(init=lambda x, y: x + y))


@pytest.fixture
def fresh(grid):
    # zero validity: store-constructed
    return grid.create_field(init=lambda x, y: x * y)


# ================================================================
#  Windowed stencil kernels consume their reach
# ================================================================
def test_stencil_kernel_consumes_reach_on_its_axis(grid, synced):
    w = grid.decomposition.halo["x"]
    d = FiniteDifference(order=2)["x"]._apply(synced)
    # 2-point staggered window: per-side maximum reach 1
    assert d.halo_valid["x"] == w - 1
    assert d.halo_valid["y"] == grid.decomposition.halo["y"]


def test_stencil_kernel_floors_at_zero(fresh):
    d = FiniteDifference(order=2)["x"]._apply(fresh)
    assert d.halo_valid == HaloSpec.zero(("x", "y"))


def test_interp_consumes_like_a_stencil(grid, synced):
    w = grid.decomposition.halo["y"]
    i = LinearInterp()["y"]._apply(synced)
    assert i.halo_valid["y"] == w - 1
    assert i.halo_valid["x"] == grid.decomposition.halo["x"]


def test_chained_kernels_consume_stepwise(grid, synced):
    # validity threads through raw kernel chains without any sync
    w = grid.decomposition.halo["x"]
    op = FiniteDifference(order=2)["x"]
    once = op._apply(synced)
    twice = op._apply(once)
    assert once.halo_valid["x"] == w - 1
    assert twice.halo_valid["x"] == max(w - 2, 0)


# ================================================================
#  Storage-frame elementwise ops carry the operand minimum
# ================================================================
def test_product_claims_the_operand_minimum(synced, fresh):
    p = CollocationProduct()._apply(synced, fresh)
    assert p.halo_valid == HaloSpec.zero(("x", "y"))
    q = CollocationProduct()._apply(synced, synced)
    assert q.halo_valid == synced.halo_valid


def test_divide_claims_the_operand_minimum(grid, synced):
    denominator = grid.sync(
        grid.create_field(init=lambda x, y: 1.0 + x + y))
    q = Divide()._apply(synced, denominator)
    assert q.halo_valid == synced.halo_valid


# ================================================================
#  The public surface returns the kernel claim (stage C: the
#  pre-kernel sync fills the operand, the kernel consumes its reach)
# ================================================================
def test_public_application_returns_the_kernel_claim(grid, fresh):
    d = fresh.diff("x")
    w = grid.decomposition.halo
    assert d.halo_valid["x"] == w["x"] - 1
    assert d.halo_valid["y"] == w["y"]
