"""
Exchange-count gate of the consumption-side sync contract (task 1.8).

Description
-----------
The point of the redo (``notes/framework2/sync_redo_plan.md``): the
composed step pays roughly one exchange per state component per
step, not one per operator application and field ``+``/``-``. Every
``grid.sync`` call is one (potential) communication round, so the
counts below are the contract: chains elide via the kernels'
validity claims, repeated consumers share the memoized sync,
pointwise work never exchanges.
"""
import jax
import pytest

from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)


@pytest.fixture
def sync_log(monkeypatch):
    calls = []
    original = Grid.sync

    def counting(self, field, boundary_data=None):
        calls.append(field)
        return original(self, field, boundary_data)

    monkeypatch.setattr(Grid, "sync", counting)
    return calls


@pytest.fixture
def grid():
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    my = IntervalMesh(16, (0.0, 2.0), name="y")
    return Grid((mx, my))


@pytest.fixture
def f(grid):
    return grid.create_field(init=lambda x, y: x + y, name="f")


@pytest.fixture
def g(grid):
    return grid.create_field(init=lambda x, y: x * y, name="g")


def test_field_arithmetic_never_syncs(sync_log, f, g):
    _ = 2.0 * f + g - 0.5 * (f + g)
    assert sync_log == []


def test_pointwise_products_never_sync(sync_log, f, g):
    _ = f * g
    _ = f / (1.0 + g)
    _ = f ** 2
    assert sync_log == []


def test_a_stencil_chain_pays_one_exchange(sync_log, grid, f):
    # entry sync fills the negotiated width; the kernel claims keep
    # the intermediate valid, so the second diff elides
    assert grid.decomposition.halo["x"] >= 2
    _ = f.diff("x").diff("x")
    assert len(sync_log) == 1
    assert sync_log[0] is f


def test_repeated_consumers_share_one_exchange(sync_log, f):
    # n tendency modules reading one state component: the first
    # consumer's sync is memoized onto the field object
    _ = f.diff("x")
    _ = f.diff("y")
    _ = f.diff("x").to(f.function_space.bare)
    assert len(sync_log) == 1


def test_an_operator_sum_pays_one_exchange(sync_log, f):
    fd = FiniteDifference(order=2)
    _ = (fd["x"] + 2.0 * fd["x"])(f)
    assert len(sync_log) == 1


def test_a_representative_tendency_pays_one_exchange_per_component(
        sync_log, f, g):
    # advection + diffusion on two components: one exchange each
    center = f.function_space.bare

    def tendency(u, q):
        du = (-1.0 * u.diff("x").to(center)
              + 0.02 * (u.diff("x").diff("x")
                        + u.diff("y").diff("y")))
        dq = -1.0 * (u * q).diff("x").to(center)
        return du, dq

    _ = tendency(f, g)
    # u synced once (memoized across its four consumers); the
    # product u*q is pointwise (no sync) but its *derivative*
    # consumes ghosts of the store-constructed product: one more
    assert len(sync_log) == 2
    assert sync_log[0] is f


def test_traced_negotiation_buys_chain_elision(sync_log):
    # width-independence (task 1.8): under the registry width (2) a
    # triple chain pays a mid-chain re-sync; negotiating against the
    # traced step widens to its sync-free demand (3) and the same
    # chain pays one exchange — both are correct, width only tunes
    # the exchange count
    def chain(u):
        return u.diff("x").diff("x").diff("x")

    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    my = IntervalMesh(16, (0.0, 2.0), name="y")
    narrow = Grid((mx, my))
    assert narrow.decomposition.halo["x"] == 2
    _ = chain(narrow.create_field(init=lambda x, y: x + y))
    assert len(sync_log) == 2

    sync_log.clear()
    mx2 = IntervalMesh(16, (0.0, 1.0), name="x")
    my2 = IntervalMesh(16, (0.0, 2.0), name="y")
    wide = Grid((mx2, my2))
    space = wide.create_field().function_space
    wide.negotiate(state_spaces=(space,), tendency=chain)
    assert wide.decomposition.halo["x"] == 3
    _ = chain(wide.create_field(init=lambda x, y: x + y))
    assert len(sync_log) == 1


def test_the_memoized_sync_is_ghost_only(grid, f):
    # the write-back must never touch true-shape data
    before = f.data
    _ = f.diff("x")
    assert (f.data == before).all()
    assert f.halo_valid == grid.decomposition.halo.over(("x", "y"))


def test_closure_captured_fields_do_not_swallow_tracers(f):
    # a concrete field consumed inside someone else's trace: the
    # write-back is skipped, the field stays concrete

    @jax.jit
    def consume():
        return f.diff("x").data

    consume()
    assert not isinstance(f._data, jax.core.Tracer)
    assert f.halo_valid["x"] == 0


def test_scan_carries_stay_at_the_zero_validity_fixed_point(grid):
    # the persistent seam: state updates are store-constructed
    # (validity zero), so scan carries have a stable treedef.
    # (unnamed carry: arithmetic resets metadata, the known
    # phase-1 finding — orthogonal to validity)
    f = grid.create_field(init=lambda x, y: x + y)
    center = f.function_space.bare
    dt = 0.01

    def body(u, _):
        du = -1.0 * u.diff("x").to(center)
        return u + dt * du, None

    @jax.jit
    def run(u):
        final, _ = jax.lax.scan(body, u, None, length=3)
        return final

    out = run(f)
    assert out.halo_valid == f.halo_valid
