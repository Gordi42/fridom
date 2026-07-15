"""The distributed-solve fast path of the production projection.

Perf-guard tests (merge plan stage 2.4): every *fallback* of the
pressure solve is covered elsewhere, but nothing asserted that the
production projection actually lands on the fast path. A regression
that pushes the real solve off the distributed plan — an intern-key
change, a transform-plan decline, a Symbol-form elliptic — leaves
every correctness test green while multiplying the multi-device step
cost. These tests spy on the resolution seam and run a real step.
"""
import numpy as np
import pytest

import fridom.nonhydro2 as nh
import fridom.spatial.operators.spectral_solve as spectral_solve_mod
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.distributed_solve import SlabSolve

# a distinctive domain length so the interned spaces (and with them
# the chunk executable) cannot be shared with another test's model —
# a cache hit would skip the trace and the spy would see nothing
LENGTH = 5.0
N = 16


def _make_model(*, periodic_z):
    grid = Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=periodic, name=name)
        for name, periodic in (("x", True), ("y", True),
                               ("z", periodic_z))))
    model = nh.Model(grid=grid, dt=0.02, advection=False,
                     coriolis=FPlaneCoriolis(f0=1.0))
    model.set_fields(u=np.ones((N, N, N)))
    return model


@pytest.fixture
def resolutions(monkeypatch):
    """Spy on ``resolve_distributed_solve``, collecting its results."""
    seen = []
    real = spectral_solve_mod.resolve_distributed_solve

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append(out)
        return out

    monkeypatch.setattr(
        spectral_solve_mod, "resolve_distributed_solve", spy)
    return seen


@pytest.mark.multi_device
def test_periodic_projection_resolves_the_distributed_solve(
        resolutions):
    # the production step itself: the projection stage constructs its
    # SpectralSolve at trace time, so one advance drives the seam
    model = _make_model(periodic_z=True)
    model.advance(1)
    assert resolutions, (
        "the projection never consulted the distributed resolution")
    assert any(s is not None for s in resolutions), (
        "the periodic multi-device pressure solve fell back to the "
        "replicated composite — the distributed fast path regressed")


@pytest.mark.multi_device
def test_walled_projection_resolves_the_distributed_solve(
        resolutions):
    # the gap is closed: the walled column now distributes through the
    # joint ComposedTransform plan (the trig axis is the transpose
    # partner, so a Fourier axis stays local for the rfft half
    # spectrum), so the production projection lands on the distributed
    # fast path instead of the replicated composite it used to keep.
    model = _make_model(periodic_z=False)
    model.advance(1)
    assert resolutions, (
        "the projection never consulted the distributed resolution")
    assert any(s is not None for s in resolutions), (
        "the walled multi-device pressure solve fell back to the "
        "replicated composite — the mixed distributed path regressed")
    assert all(isinstance(s, SlabSolve)
               for s in resolutions if s is not None)
