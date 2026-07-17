"""The distributed-solve fast path of the production projection.

Perf-guard tests (merge plan stage 2.4): every *fallback* of the
pressure solve is covered elsewhere, but nothing asserted that the
production projection actually lands on the fast path. A regression
that pushes the real solve off the distributed plan — an intern-key
change, a transform-plan decline, a Symbol-form elliptic — leaves
every correctness test green while multiplying the multi-device step
cost. These tests spy on the resolution seam and run a real step.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
import fridom.spatial.operators.spectral_solve as spectral_solve_mod
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.distributed_solve import SlabSolve

# a distinctive domain length so the interned spaces (and with them
# the chunk executable) cannot be shared with another test's model —
# a cache hit would skip the trace and the spy would see nothing
LENGTH = 5.0
N = 16


def _make_model(*, periodic_z, family=None):
    grid = Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=periodic, name=name)
        for name, periodic in (("x", True), ("y", True),
                               ("z", periodic_z))))
    model = nh.Model(grid=grid, dt=0.02, advection=False,
                     coriolis=FPlaneCoriolis(f0=1.0), family=family)
    model.set_fields(u=np.ones(model.state["u"].data.shape))
    return model


# an indivisible (prime) triple-periodic domain: shards over no device
# count, so it exercises the Phase 2 padded balanced all-to-all (19 is
# the smallest prime whose last ceil-block shard clears the model's
# width-2 halo over four devices)
N_PRIME = 19


def _make_prime_model(*, device_ids=None):
    grid = Grid(tuple(
        IntervalMesh(N_PRIME, (0.0, LENGTH), periodic=True, name=name)
        for name in ("x", "y", "z")), device_ids=device_ids)
    return nh.Model(grid=grid, dt=0.02, advection=False,
                    coriolis=FPlaneCoriolis(f0=1.0))


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
    # family="nodal" is explicit: since the 2026-07-16 ruling a walled
    # grid auto-flips to FV (that path has its own twin below), so this
    # nodal walled distributed gate is pinned by an explicit family.
    model = _make_model(periodic_z=False, family="nodal")
    model.advance(1)
    assert resolutions, (
        "the projection never consulted the distributed resolution")
    assert any(s is not None for s in resolutions), (
        "the walled multi-device pressure solve fell back to the "
        "replicated composite — the mixed distributed path regressed")
    assert all(isinstance(s, SlabSolve)
               for s in resolutions if s is not None)


@pytest.mark.multi_device
def test_walled_fv_projection_resolves_the_distributed_solve(
        resolutions):
    # F4: the walled FV pressure solve (DCT-II on the Neumann CellAvg
    # origin) distributes through the same joint ComposedTransform plan
    # as the nodal walled solve -- the trig axis is the transpose
    # partner, a Fourier axis stays local -- so an explicit family="fv"
    # walled model lands on the distributed fast path too
    model = _make_model(periodic_z=False, family="fv")
    model.advance(1)
    assert resolutions, (
        "the FV projection never consulted the distributed resolution")
    slabs = [s for s in resolutions if s is not None]
    assert slabs, (
        "the walled FV multi-device pressure solve fell back to the "
        "replicated composite -- the mixed distributed path regressed")
    assert all(isinstance(s, SlabSolve) for s in slabs)


@pytest.mark.multi_device
def test_walled_x_fv_projection_dodges_the_wall_and_distributes(
        resolutions):
    # the walled-x FV twin: the staggering-aware default shards the
    # periodic y and keeps the walled x local, so the FV projection
    # stays on the distributed fast path (a Fourier axis sharded, the
    # trig axis local)
    grid = Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=periodic, name=name)
        for name, periodic in (("x", False), ("y", True), ("z", True))))
    model = nh.Model(grid=grid, dt=0.02, advection=False,
                     coriolis=FPlaneCoriolis(f0=1.0), family="fv")
    default = grid.decomposition.default_layout
    assert default.is_local("x")       # the walled axis is dodged
    assert not default.is_local("y")   # a periodic axis is sharded
    model.set_fields(u=np.ones(model.state["u"].data.shape))
    model.advance(1)
    slabs = [s for s in resolutions if s is not None]
    assert slabs, (
        "the walled-x FV multi-device pressure solve fell back to the "
        "replicated composite -- the FV distributed path regressed")
    assert all(isinstance(s, SlabSolve) for s in slabs)


def test_walled_x_fv_step_is_device_count_invariant():
    # the FV twin of the walled-x parity gate: the 4-device FV step
    # (sharding periodic y, x trig local) vs the 1-device replicated FV
    # step agree within the step gate over 20 steps
    def build(device_ids):
        grid = Grid(tuple(
            IntervalMesh(N, (0.0, LENGTH), periodic=periodic, name=name)
            for name, periodic in (("x", False), ("y", True),
                                   ("z", True))),
            device_ids=device_ids)
        return nh.Model(grid=grid, dt=0.02, advection=False,
                        coriolis=FPlaneCoriolis(f0=1.0), family="fv")

    one = build((0,))
    rng = np.random.default_rng(0)
    ic = {name: rng.standard_normal(one.state[name].data.shape)
          for name in ("u", "v", "w", "b")}

    def run(model):
        model.set_fields(**ic)
        model.advance(20)
        return {name: np.asarray(model.state[name].data)
                for name in ("u", "v", "w", "b")}

    ref = run(one)
    many = run(build(None))
    for name, r in ref.items():
        assert np.allclose(many[name], r, rtol=1e-10, atol=1e-11), (
            name, np.abs(many[name] - r).max())


@pytest.mark.multi_device
def test_walled_x_projection_dodges_the_wall_and_distributes(
        resolutions):
    # x is walled but y, z are periodic and divisible: the staggering-
    # aware default (spatial.decomposition ordering) shards the periodic
    # y, never the walled x, so the deficit face leg stays off the
    # storage-shard axis. The production projection still lands on the
    # distributed fast path (a periodic Fourier axis is sharded, the
    # trig axis is kept local). family="nodal" is explicit (the FV twin
    # is above): a walled grid auto-flips to FV since the 2026-07-16
    # ruling, so this nodal walled-x gate is pinned by family.
    grid = Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=periodic, name=name)
        for name, periodic in (("x", False), ("y", True), ("z", True))))
    model = nh.Model(grid=grid, dt=0.02, advection=False,
                     coriolis=FPlaneCoriolis(f0=1.0), family="nodal")
    default = grid.decomposition.default_layout
    assert default.is_local("x")       # the walled axis is dodged
    assert not default.is_local("y")   # a periodic axis is sharded
    model.set_fields(u=np.ones(model.state["u"].data.shape))
    model.advance(1)
    assert resolutions
    slabs = [s for s in resolutions if s is not None]
    assert slabs, (
        "the walled-x multi-device pressure solve fell back to the "
        "replicated composite -- the staggering-aware default regressed")
    assert all(isinstance(s, SlabSolve) for s in slabs)


def test_walled_x_step_is_device_count_invariant():
    # parity gate for the reordered walled-x geometry: the 4-device step
    # (sharding periodic y, x trig local) vs the 1-device replicated step
    # drift within the step gate over 20 steps -- the consistency check
    # that reordering the default off the walled axis is exact.
    # family="nodal" is explicit (the FV twin is above): a walled grid
    # auto-flips to FV since the 2026-07-16 ruling.
    def build(device_ids):
        grid = Grid(tuple(
            IntervalMesh(N, (0.0, LENGTH), periodic=periodic, name=name)
            for name, periodic in (("x", False), ("y", True),
                                   ("z", True))),
            device_ids=device_ids)
        return nh.Model(grid=grid, dt=0.02, advection=False,
                        coriolis=FPlaneCoriolis(f0=1.0), family="nodal")

    one = build((0,))
    rng = np.random.default_rng(0)
    ic = {name: rng.standard_normal(one.state[name].data.shape)
          for name in ("u", "v", "w", "b")}

    def run(model):
        model.set_fields(**ic)
        model.advance(20)
        return {name: np.asarray(model.state[name].data)
                for name in ("u", "v", "w", "b")}

    ref = run(one)
    many = run(build(None))
    for name, r in ref.items():
        assert np.allclose(many[name], r, rtol=1e-10, atol=1e-11), (
            name, np.abs(many[name] - r).max())


@pytest.mark.multi_device
def test_prime_projection_resolves_the_padded_distributed_solve(
        resolutions):
    # the Phase 2 gate: an indivisible (prime) triple-periodic domain
    # keeps the distributed projection through the padded balanced
    # all-to-all instead of replicating the spectral cube. The spy sees
    # a padded SlabSolve, not a None fallback.
    model = _make_prime_model()
    model.set_fields(u=np.ones((N_PRIME, N_PRIME, N_PRIME)))
    model.advance(1)
    assert resolutions
    slabs = [s for s in resolutions if s is not None]
    assert slabs, (
        "the prime multi-device pressure solve fell back to the "
        "replicated composite -- the indivisible fast path regressed")
    assert all(isinstance(s, SlabSolve) for s in slabs)
    assert all(s.plan.padded for s in slabs)


def test_prime_step_is_device_count_invariant():
    # the production step at a prime N stays device-count invariant:
    # the 4-device padded distributed solve vs the 1-device replicated
    # solve drift within the usual step gate over 20 steps
    rng = np.random.default_rng(0)
    ic = {name: rng.standard_normal((N_PRIME,) * 3)
          for name in ("u", "v", "w", "b")}

    def run(device_ids):
        model = _make_prime_model(device_ids=device_ids)
        model.set_fields(**ic)
        model.advance(20)
        return {name: np.asarray(model.state[name].data)
                for name in ("u", "v", "w", "b")}

    many = run(None)
    one = run((0,))
    for name, ref in one.items():
        assert np.allclose(many[name], ref, rtol=1e-10, atol=1e-11), (
            name, np.abs(many[name] - ref).max())


def _make_mapped_fv_model(*, device_ids=None):
    # a terrain-following FV column (zp = z H(x), walled z, periodic x/y):
    # the mapped PCG projection derives every metric through halo-clean
    # field ops and preconditions with the mixed Fourier x Cosine
    # spectral inverse, so it must ride the same decomposition as the
    # walled FV solve
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    grid = Grid((
        IntervalMesh(N, (0.0, LENGTH), periodic=True, name="x"),
        IntervalMesh(N, (0.0, LENGTH), periodic=True, name="y"),
        IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")),
        mapping=mapping, device_ids=device_ids)
    return nh.Model(grid=grid, dt=0.02, advection=True,
                    coriolis=FPlaneCoriolis(f0=1.0),
                    pressure_iterations=16, family="fv")


def test_mapped_fv_step_is_device_count_invariant():
    # gate 7: the mapped FV projection smoke under forced-4 matches the
    # 1-device replicated result. Sharding the periodic x/y while the
    # walled z stays local must not perturb the PCG metric chains, the
    # corner cross interpolations, or the conservative buoyancy flux
    rng = np.random.default_rng(0)
    one = _make_mapped_fv_model(device_ids=(0,))
    ic = {name: rng.standard_normal(one.state[name].data.shape)
          for name in ("u", "v", "w", "b")}

    def run(model):
        model.set_fields(**ic)
        model.advance(6)
        return {name: np.asarray(model.state[name].data)
                for name in ("u", "v", "w", "b")}

    ref = run(one)
    many = run(_make_mapped_fv_model(device_ids=None))
    for name, r in ref.items():
        assert np.allclose(many[name], r, rtol=1e-9, atol=1e-10), (
            name, np.abs(many[name] - r).max())


def _make_immersed_model(*, device_ids=None):
    # a face-aligned {0, 1} immersed box (periodic x/y, walled z): the
    # masked cut-cell PCG derives the open-area fractions through the
    # ordinary store + sync path and preconditions with the wet-masked
    # spectral inverse, so sharding the periodic x/y must not perturb
    # the projection, the MaskState, or the fraction-weighted advection
    from fridom.spatial.immersed_domain import ImmersedDomain  # noqa: PLC0415

    def box(x, y, z):
        return ((x > 1.0) & (x < 4.0) & (y > 1.0) & (y < 4.0)
                & (z > 0.2) & (z < 0.8)).astype(float)

    grid = Grid((
        IntervalMesh(N, (0.0, LENGTH), periodic=True, name="x"),
        IntervalMesh(N, (0.0, LENGTH), periodic=True, name="y"),
        IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(box), device_ids=device_ids)
    return nh.Model(grid=grid, dt=0.02, advection=True,
                    coriolis=FPlaneCoriolis(f0=1.0),
                    pressure_iterations=25)


def test_immersed_step_is_device_count_invariant():
    # I2 gate: the masked cut-cell projection smoke under forced-4
    # matches the 1-device replicated result. The immersed fractions,
    # the wet-mean projection reductions, and the boolean masking all
    # ride the ordinary decomposition, so the sharded step reproduces
    # the replicated one
    rng = np.random.default_rng(0)
    one = _make_immersed_model(device_ids=(0,))
    ic = {name: rng.standard_normal(one.state[name].data.shape)
          for name in ("u", "v", "w", "b")}

    def run(model):
        model.set_fields(**ic)
        model.advance(6)
        return {name: np.asarray(model.state[name].data)
                for name in ("u", "v", "w", "b")}

    ref = run(one)
    many = run(_make_immersed_model(device_ids=None))
    for name, r in ref.items():
        assert np.allclose(many[name], r, rtol=1e-9, atol=1e-10), (
            name, np.abs(many[name] - r).max())
