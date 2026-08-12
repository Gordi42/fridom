r"""Flat-axis halo elision adds no compilations and no sync nodes.

The elision keeps ``halo_valid`` honest: a flat axis stores no ghost
slots, so a stencil application along it consumes no validity and the
operator base inserts no extra ``Sync``. Were the claim consumed
instead, ``_ensure_valid`` would rebuild a sync node at every
application -- and that sync refills the *other* axes for real, which
is both a step cost and a recompile.

Kept in its own file rather than in ``test_thin_axis.py``: the
compile-count fixture measures a whole warm re-run, so it wants a
module boundary of its own (``conftest.py`` evicts jax's in-process
cache at every test-file boundary).
"""
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
DT = 5e-3
TWO_PI = 2.0 * np.pi


def make_model(nz, order=5):
    """Return a tiny periodic ``N x N x nz`` WENO model."""
    grid = Grid((
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(nz, (0.0, TWO_PI), periodic=True, name="z"),
    ))
    model = nh.Model(
        grid=grid,
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=1.0),
        advection=nh.modules.WENOAdvection(order=order))
    ax = (np.arange(N) + 0.5) * (TWO_PI / N)
    x, y = np.meshgrid(ax, ax, indexing="ij")
    model.set_fields(u=np.repeat((0.2 * np.sin(y))[:, :, None], nz,
                                 axis=2),
                     v=np.repeat((0.2 * np.cos(x))[:, :, None], nz,
                                 axis=2))
    return model


@pytest.mark.parametrize("nz", [1, 2])
def test_a_warm_re_run_adds_no_compilations(nz, compile_counter):
    # nz = 1 is the elided flat axis, nz = 2 the un-elided control:
    # both must reach the steady state where stepping recompiles
    # nothing. A flat axis whose validity claim were consumed would
    # churn a fresh Sync node into the chunk on every application
    model = make_model(nz)
    model.advance(3)
    warm = compile_counter.count
    compile_counter.reset()
    model.advance(3)
    assert compile_counter.count == 0
    # the counter was live: the cold build did trace (a listener that
    # never fires would make the assertion above vacuous)
    assert warm > 0
