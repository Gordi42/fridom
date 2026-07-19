r"""The split-explicit free surface on a terrain-following (sigma) grid.

Retires the H3 chart taught error for the split-explicit variant: the
barotropic subcycle steps the volume-exact terrain transport form
(GM-D1 option 1), consistent with the shipped implicit variant. With
``J`` the column Jacobian, ``H(x, y) = int J dz`` the physical column
depth and ``H_ref`` the constant vertical mesh extent, the ps substep is
``ps <- ps - dtau (c^2/H_ref) div(H_a ubar)`` (a CONSTANT gravity
coefficient, no ``1/H(x, y)`` division), the transports are ``U = H_a
ubar``, and the S4 depth-mean correction targets ``U/H(x, y)`` with the
variable per-column depth. The SM2005 filter is unchanged (frozen §5.4).
Terrain + immersed stays a narrowed taught error. Self-contained builders
(AGENTS oversized-module rule).
"""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.io.streams import SnapshotMismatchError
from fridom.model.model import _chunk_body
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.immersed_domain import ImmersedDomain

IM = fr.spatial.meshes.IntervalMesh
SEFS = hy.SplitExplicitFreeSurface
N2, CSQR = 2.0, 1.0


# ================================================================
#  Builders (duplicated per the self-contained-shard rule)
# ================================================================
def _wavy(x, y):
    """Return a genuinely horizontally-varying sigma column depth H(x, y)."""
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _mapping(depth=_wavy):
    return CoordinateMapping(maps={"zp": lambda z, H: z * H},
                             params={"H": depth})


def _terrain_grid(n, nz, *, depth=_wavy, zext=1.0):
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-zext, 0.0), periodic=False, name="z")),
        mapping=_mapping(depth))


def _flat_grid(n, nz, *, zext=1.0):
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-zext, 0.0), periodic=False, name="z")))


def _model(grid, *, free_surface=None, substeps=16, csqr=CSQR, f0=0.5,
           n2=1.0, dt=1e-2, stepper=None):
    return hy.Model(
        grid=grid, dt=dt, csqr=csqr, advection=False,
        stratification=hy.ConstantStratification(n2=n2),
        coriolis=hy.FPlaneCoriolis(f0=f0) if f0 else None,
        free_surface=free_surface or SEFS(substeps=substeps),
        time_stepper=stepper
        or fr.model.time_steppers.AdamBashforth(dt, order=3))


def _random_ic(model, *, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        k: scale * rng.standard_normal(model.state[k].shape)
        for k in ("u", "v", "b", "ps")})


def _fs(model):
    return model.module(SEFS)


# ================================================================
#  The taught error is gone: the terrain subcycle engages
# ================================================================
def test_terrain_split_engages_and_runs_finite():
    grid = _terrain_grid(16, 4)
    model = _model(grid, f0=0.5, n2=1.0)
    fs = _fs(model)
    assert fs._column == ("zp", "z")
    assert fs._immersed is None
    assert {"ps", "U", "V"} <= set(model.state.component_names)
    _random_ic(model, seed=1, scale=0.5)
    model.advance(20)
    assert not model.panicked
    for k in ("u", "v", "b", "ps", "U", "V"):
        data = np.asarray(model.state[k].data)
        assert bool(np.isfinite(data).all()), k
    # a barotropic gravity wave oscillates, does not blow up
    assert float(np.abs(np.asarray(model.state["ps"].data)).max()) < 50.0


# ================================================================
#  Flat-limit: a J == 1 chart reproduces the unmapped subcycle
# ================================================================
def _state_bytes_sha(model, *, normalize_signed_zero):
    h = hashlib.sha256()
    for name in sorted(model.state.component_names):
        arr = np.ascontiguousarray(np.asarray(model.state[name].data))
        if normalize_signed_zero:
            arr = arr + 0.0  # -0.0 -> +0.0, every nonzero value unchanged
        h.update(name.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def test_flat_chart_matches_the_unmapped_subcycle():
    r"""A J == 1 sigma chart is bitwise the unmapped split-explicit.

    With H(x, y) == 1 (identity map, J == 1) and a power-of-two vertical
    extent / spacing, the terrain subcycle's ``c^2/H_ref`` and ``H_a``
    factors are exact powers of two, so the divergence, the transports and
    the S4 correction agree with the flat scalar ``1/H_ref`` path to the
    last bit. Every component is *numerically* bitwise identical
    (``array_equal``, so ``-0.0 == 0.0``); the split-explicit prognostics
    ``ps, U, V, u, v`` and the AUX buffers are byte-for-byte identical; the
    only raw-byte difference across the full state is signed-zero elements
    in the diagnosed ``w`` (a pre-existing core terrain w-diagnosis
    artifact, ``u Z_x + v Z_y`` on a J == 1 chart — present for the
    explicit variant too, independent of the subcycle), so a
    signed-zero-normalized state hash matches exactly.
    """
    one = lambda x, y: 1.0 + 0.0 * x + 0.0 * y  # noqa: E731 — tiny local
    chart = _model(_terrain_grid(12, 4, depth=one, zext=2.0), csqr=4.0)
    flat = _model(_flat_grid(12, 4, zext=2.0), csqr=4.0)
    assert _fs(chart)._column == ("zp", "z")
    assert _fs(flat)._column is None
    # the J == 1 physical column depth is exactly the (power-of-two) extent
    depth = _fs(chart)._physical_depth(chart.state["u"])
    assert float(depth.data.min()) == 2.0
    assert float(depth.data.max()) == 2.0
    _random_ic(chart, seed=5)
    _random_ic(flat, seed=5)
    chart.advance(15)
    flat.advance(15)
    # every component numerically bitwise (array_equal treats -0.0 == 0.0)
    for name in sorted(chart.state.component_names):
        a = np.asarray(chart.state[name].data)
        b = np.asarray(flat.state[name].data)
        assert np.array_equal(a, b), name
    # the split-explicit prognostics / AUX are byte-for-byte identical
    for name in ("ps", "U", "V", "u", "v", "ubar_prev", "vbar_prev"):
        a = np.ascontiguousarray(np.asarray(chart.state[name].data))
        b = np.ascontiguousarray(np.asarray(flat.state[name].data))
        assert a.tobytes() == b.tobytes(), name
    # signed-zero-normalized full-state hash is identical
    assert (_state_bytes_sha(chart, normalize_signed_zero=True)
            == _state_bytes_sha(flat, normalize_signed_zero=True))


# ================================================================
#  Volume-exact signature: int(ps) is a machine-precision invariant
# ================================================================
def test_terrain_ps_volume_conserved_to_machine_precision():
    # d/dt int(ps) = -(c^2/H_ref) int(div(H_a ubar)) = 0 on a periodic
    # domain, so the physical (J-weighted) ps volume drifts only at
    # round-off — the volume-exact (no 1/H) signature. Measured ~4e-17.
    model = _model(_terrain_grid(16, 4), csqr=2.0, f0=0.5, n2=1.0)
    _random_ic(model, seed=2)

    def volume():
        return float(jnp.sum(model.state["ps"].integrate().data))

    before = volume()
    model.advance(20)
    after = volume()
    assert not model.panicked
    assert abs(after - before) < 1e-13 * max(abs(before), 1.0)


# ================================================================
#  Rest state over topography stays at rest
# ================================================================
def _nodes(grid, space, name):
    return grid.evaluation_nodes(space.bare, name).data


def _rest_run(grid, free_surface, *, steps=10, dt=2e-3):
    model = _model(grid, free_surface=free_surface, f0=0.0, n2=N2, dt=dt)
    coll = model.state["b"].function_space
    zp = (_nodes(grid, coll, "z")
          * _wavy(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
    model.set_fields(
        u=np.zeros(model.state["u"].shape),
        v=np.zeros(model.state["v"].shape),
        b=np.asarray(-N2 * zp),
        ps=np.zeros(model.state["ps"].shape))
    model.run(steps, progress=False)
    return model


def test_terrain_rest_state_matches_the_explicit_oracle():
    # a stratified fluid at rest over topography: the residual current is
    # the truncation-order (slope-corrected) baroclinic PG error, and the
    # barotropic subcycle injects NO spurious barotropic mode on top — the
    # split rest residual equals the explicit oracle's to a fraction of a
    # percent (measured ratio 1.000), and both stay small.
    grid = _terrain_grid(16, 8)
    split = _rest_run(grid, SEFS(substeps=16))
    oracle = _rest_run(grid, hy.ExplicitFreeSurface())
    assert not split.panicked
    su = float(jnp.abs(split.state["u"].data).max())
    eu = float(jnp.abs(oracle.state["u"].data).max())
    assert su < 1e-2
    assert su == pytest.approx(eu, rel=0.05)


# ================================================================
#  Seeding: U = ubar_physical * H_col with the VARIABLE per-column H
# ================================================================
def _z_uniform(model, name, seed):
    shape = model.state[name].shape
    plane = np.random.default_rng(seed).standard_normal((*shape[:2], 1))
    return np.broadcast_to(plane, shape).copy()


def test_set_fields_seeds_the_variable_depth_transport():
    grid = _terrain_grid(16, 6)
    model = _model(grid, substeps=16)
    model.set_fields(u=_z_uniform(model, "u", 0),
                     v=_z_uniform(model, "v", 1))
    fs = _fs(model)
    # H_a on the u/v faces is the physical column depth int J dz, NOT the
    # constant extent, so the transport U = ubar * H(x, y) varies with x, y
    h_u = np.asarray(fs._physical_depth(model.state["u"]).data)
    h_v = np.asarray(fs._physical_depth(model.state["v"]).data)
    ubar = np.asarray(model.state["u"].mean("z").data)
    vbar = np.asarray(model.state["v"].mean("z").data)
    assert np.abs(np.asarray(model.state["U"].data)
                  - ubar * h_u).max() < 1e-13
    assert np.abs(np.asarray(model.state["V"].data)
                  - vbar * h_v).max() < 1e-13
    # the seeded depth is genuinely variable (not the flat extent)
    assert float(h_u.max() - h_u.min()) > 0.1
    # the barotropic velocity survives a few steps (no collapse)
    umax0 = float(np.abs(np.asarray(model.state["u"].data)).max())
    model.advance(3)
    assert not model.panicked
    assert float(np.abs(np.asarray(model.state["u"].data)).max()) > 0.5 * umax0


# ================================================================
#  Cross-variant sanity: split vs the volume-exact implicit variant
# ================================================================
def test_cross_variant_closeness_to_the_implicit_variant():
    # both variants are now volume-exact with the same gH(x, y) barotropic
    # physics; from a nearly-balanced barotropic state a short small-dt run
    # spins down closely (the split FILTERS the fast barotropic mode, the
    # implicit PROJECTS it, so they agree on the slow adjusted state to a
    # few percent, not bitwise).
    grid = _terrain_grid(16, 4)
    dt = 1e-3
    ms = _model(grid, free_surface=SEFS(substeps=32), csqr=CSQR, f0=0.5,
                n2=1.0, dt=dt)
    mi = _model(grid, free_surface=hy.ImplicitFreeSurface(), csqr=CSQR,
                f0=0.5, n2=1.0, dt=dt)
    for m in (ms, mi):
        rng = np.random.default_rng(7)
        m.set_fields(
            u=0.1 * _z_uniform(m, "u", 7),
            v=0.1 * _z_uniform(m, "v", 8),
            b=0.1 * rng.standard_normal(m.state["b"].shape))
    ms.run(steps=30)
    mi.run(steps=30)
    us = np.asarray(ms.state["u"].data)
    ui = np.asarray(mi.state["u"].data)
    rel = np.linalg.norm(us - ui) / max(np.linalg.norm(ui), 1e-30)
    assert not ms.panicked
    assert not mi.panicked
    assert rel < 0.1, rel


# ================================================================
#  Autodiff regression (differentiability policy): no 1/H hazard
# ================================================================
def test_grad_through_terrain_split_run_matches_finite_difference():
    r"""``jax.grad`` w.r.t. the initial ``ps`` on a periodic terrain run.

    Differentiate the pure kernel ``_chunk_body`` w.r.t. the initial
    surface pressure through the volume-exact terrain subcycle (no
    ``1/H(x, y)`` division in the substep path, so no masked-singularity
    VJP hazard) and check a random directional projection against a central
    finite difference.
    """
    model = _model(_terrain_grid(8, 4), substeps=16, f0=0.5, n2=0.0,
                   dt=2e-3)
    rng = np.random.default_rng(1)
    model.set_fields(ps=0.1 * rng.standard_normal(model.state["ps"].shape))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper

    leaf = carry.state["ps"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 8, spliced, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = np.asarray(jax.grad(loss)(leaf))
    assert bool(np.all(np.isfinite(grad)))

    direction = jnp.asarray(
        np.random.default_rng(2).standard_normal(leaf.shape),
        dtype=leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(leaf + eps * direction))
          - float(loss(leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  Restart: terrain fingerprint distinct + snapshot round-trip bitwise
# ================================================================
def test_terrain_snapshot_roundtrip_is_bitwise(tmp_path):
    grid = _terrain_grid(12, 4)
    model = _model(grid, substeps=8)
    _random_ic(model, seed=3, scale=0.5)
    model.advance(5)
    model.snapshot(tmp_path / "snap")
    ref = {k: np.asarray(model.state[k].data).copy()
           for k in ("ps", "U", "V", "ubar_prev", "vbar_prev", "u", "v")}
    resumed = _model(_terrain_grid(12, 4), substeps=8)
    resumed.load_snapshot(tmp_path / "snap")
    for k, arr in ref.items():
        assert np.array_equal(np.asarray(resumed.state[k].data), arr), k


def test_terrain_snapshot_refuses_a_different_substep_count(tmp_path):
    # the integrator statics (N / filter / forcing) ride the ADVANCE stage
    # attribution into the restart fingerprint, so a terrain snapshot
    # written with one subcycle length refuses to resume a structurally
    # different one (the grid geometry is checked separately at load).
    model = _model(_terrain_grid(12, 4), substeps=8)
    _random_ic(model, seed=4, scale=0.5)
    model.advance(3)
    model.snapshot(tmp_path / "snap")
    variant = _model(_terrain_grid(12, 4), substeps=16)
    with pytest.raises(SnapshotMismatchError):
        variant.load_snapshot(tmp_path / "snap")


# ================================================================
#  Forced-4 multi-device: the terrain subcycle runs (H6 gate extended)
# ================================================================
@pytest.mark.multi_device
def test_terrain_subcycle_runs_under_forced_devices():
    grid = _terrain_grid(16, 4)
    model = _model(grid, substeps=16, csqr=9.0, f0=0.5, n2=1.0)
    _random_ic(model, seed=7)
    model.advance(20)
    umax = float(np.abs(np.asarray(model.state["u"].data)).max())
    assert not model.panicked
    assert np.isfinite(umax)
    assert umax > 0.0


# ================================================================
#  Terrain + immersed stays a narrowed taught error
# ================================================================
def _cut(x, y, zp):
    # a partial-bottom cut that leaves the upper column wet
    bed = -0.5 - 0.1 * jnp.cos(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    return zp > bed


def test_terrain_plus_immersed_is_a_narrowed_taught_error():
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=True, name="x"),
        IM(8, (0.0, 1.0), periodic=True, name="y"),
        IM(8, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mapping(),
        immersed=ImmersedDomain(_cut, order=4, min_fraction=0.1))
    with pytest.raises(NotImplementedError, match="immersed"):
        _model(grid, substeps=4,
               stepper=fr.model.time_steppers.AdamBashforth(2e-3, order=2))
