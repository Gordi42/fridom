"""End-to-end gates for the immersed hydrostatic model (IP-D9).

Column equivalence (a flat immersed bottom reproduces the shallower
unimmersed domain, dispersion parity through the ``ps`` mode), dry-DOF
hygiene over a run, lateral-partial mass conservation, and the taught
rejections (eigenmodes / transforms / biased advection). Self-contained
per the AGENTS oversized-module rule (builders duplicated).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

IM = IntervalMesh


# ================================================================
#  Shared builders (duplicated, self-contained)
# ================================================================
def _immersed_grid(init, *, n=8, nz=8, order=None, min_fraction=0.1):
    """Doubly-periodic horizontal, bounded z immersed grid on [0, 1]."""
    return Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(init, order=order,
                                min_fraction=min_fraction))


def _short_grid(*, n=8, nz=4, depth=0.5):
    """Build the unimmersed twin on the shallower z in [0, depth]."""
    return Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (0.0, depth), periodic=False, name="z")))


def _flat_bottom(x, y, z):  # noqa: ARG001
    """Face-aligned flat bottom: wet in the top four cells (z > 0.5)."""
    return (z > 0.5).astype(float)


def _prognostic_names(model):
    """Return the model's PROGNOSTIC component names."""
    return set(model._artifacts.field_table.prognostic)


# ================================================================
#  Gate b: column equivalence + dispersion parity (ps mode)
# ================================================================
@pytest.mark.parametrize(
    "make_fs",
    [pytest.param(hy.ExplicitFreeSurface, id="explicit"),
     # fixed-iteration mode: pinned for determinism
     pytest.param(lambda: hy.ImplicitFreeSurface(
         pressure_iterations=25, pressure_tolerance=None),
                  id="implicit")],
)
def test_column_equivalence_flat_bottom(make_fs):
    # a flat immersed bottom (wet = top 4 of 8 cells on z in [0, 1])
    # must reproduce the unimmersed model on the shallower z in [0, 0.5]
    # (nz = 4, same dz).  csqr uses the SAME physical gravity g = 4.0:
    # the immersed extent Z = 1 vs the unimmersed wet depth H' = 0.5, so
    # the immersed model recovers the wet-column wave speed internally.
    # A fresh free-surface instance per model (bind runs exactly once).
    g = 4.0
    mi = hy.Model(
        grid=_immersed_grid(_flat_bottom), dt=0.01, csqr=g * 1.0,
        coriolis=hy.FPlaneCoriolis(f0=0.8),
        stratification=hy.ConstantStratification(n2=2.0),
        advection=False, free_surface=make_fs())
    mu = hy.Model(
        grid=_short_grid(), dt=0.01, csqr=g * 0.5,
        coriolis=hy.FPlaneCoriolis(f0=0.8),
        stratification=hy.ConstantStratification(n2=2.0),
        advection=False, free_surface=make_fs())

    rng = np.random.default_rng(0)
    icu = {k: 0.2 * rng.standard_normal(mu.state[k].data.shape)
           for k in ("u", "v", "b")}
    # a single horizontal cos mode in ps excites the barotropic wave
    # (the dispersion-parity carrier); shared verbatim across the twins
    if "ps" in _prognostic_names(mu):
        icu["ps"] = 0.2 * rng.standard_normal(mu.state["ps"].data.shape)
    mu.set_fields(**icu)

    # embed the SAME wet-region IC into the immersed top four z-cells
    ici = {}
    for k in ("u", "v", "b"):
        arr = np.zeros(mi.state[k].data.shape)
        arr[:, :, 4:8] = icu[k][:, :, 0:4]
        ici[k] = arr
    if "ps" in icu:
        ici["ps"] = icu["ps"]  # 2D (ConstantSpace z): copy directly
    mi.set_fields(**ici)

    mi.advance(30)
    mu.advance(30)
    assert not mi.panicked
    assert not mu.panicked

    diffs = {}
    for k in ("u", "v", "b"):
        di = np.asarray(mi.state[k].data)[:, :, 4:8]
        du = np.asarray(mu.state[k].data)[:, :, 0:4]
        diffs[k] = float(np.abs(di - du).max())
    dw = np.asarray(mi.state["w"].data)[:, :, 4:9]
    wu = np.asarray(mu.state["w"].data)[:, :, 0:5]
    diffs["w"] = float(np.abs(dw - wu).max())
    if "ps" in icu:
        dps = np.asarray(mi.state["ps"].data)
        pu = np.asarray(mu.state["ps"].data)
        diffs["ps"] = float(np.abs(dps - pu).max())
    assert max(diffs.values()) < 1e-13, diffs


# ================================================================
#  Gate e: dry-DOF hygiene over a multi-step run
# ================================================================
@pytest.mark.parametrize(
    "free_surface",
    [pytest.param(hy.ImplicitFreeSurface(), id="implicit"),
     pytest.param(hy.SplitExplicitFreeSurface(substeps=8), id="split")],
)
def test_dry_dof_hygiene_over_a_run(free_surface):
    grid = _immersed_grid(_flat_bottom)
    model = hy.Model(
        grid=grid, dt=0.004, csqr=2.0,
        coriolis=hy.FPlaneCoriolis(f0=0.6),
        stratification=hy.ConstantStratification(n2=1.0),
        advection=True, free_surface=free_surface)
    rng = np.random.default_rng(3)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "b")})
    model.advance(12)
    assert not model.panicked

    imm = grid.immersed
    # the 3D prognostics are exactly zero on their dry (staggered) DOFs
    for name in ("u", "v", "b"):
        field = model.state[name]
        mask = np.asarray(imm.mask(field.function_space).data)
        dry = np.asarray(field.data) * (1 - mask)
        assert np.abs(dry).max() == 0.0, name
    # w: closed faces are alpha_z == 0 (NOT the mask -- the valid
    # surface face is mask-dry under the dry-exterior rule)
    core = model.module(hy.HydrostaticCore)
    az = np.asarray(core._masked_w_faces(imm, model.state).data)
    w_closed = np.asarray(model.state["w"].data) * (az == 0.0)
    assert np.abs(w_closed).max() == 0.0


# ================================================================
#  Gate f: lateral partials (sloping side wall) mass conservation
# ================================================================
def _sidewall(x, y, z):  # noqa: ARG001
    """Return a genuine x-partial side wall B(y), one cell wide."""
    b = 0.35 + 0.1 * jnp.sin(2 * jnp.pi * y)
    return jnp.clip((x - b) / (1.0 / 8) + 0.5, 0.0, 1.0)


def test_lateral_partials_conserve_mass():
    grid = _immersed_grid(_sidewall, nz=4, order=4, min_fraction=0.1)
    # surface_flux=False (legacy fixed-domain closure): the default
    # constancy-preserving closure advects through the top face and so
    # exchanges tracer content with the moving surface, which breaks the
    # exact theta-mass conservation this test characterizes.
    m = hy.Model(
        grid=grid, dt=0.002, csqr=1.0,
        free_surface=hy.ImplicitFreeSurface(pressure_iterations=40),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=0.0),
        advection=fr.model.modules.CenteredAdvection(surface_flux=False))
    rng = np.random.default_rng(11)
    m.set_fields(**{
        k: 0.1 * rng.standard_normal(m.state[k].data.shape)
        for k in ("u", "v", "b")})

    theta = grid.immersed.fraction(m.state["b"].function_space)
    # genuine partial cells exist (0 < theta < 1 somewhere)
    th = np.asarray(theta.data)
    assert ((th > 0) & (th < 1)).any()

    mass0 = float(jnp.sum((theta * m.state["b"]).integrate().data))
    m.advance(30)
    assert not m.panicked
    theta1 = grid.immersed.fraction(m.state["b"].function_space)
    mass1 = float(jnp.sum((theta1 * m.state["b"]).integrate().data))
    drift = abs(mass1 - mass0) / max(abs(mass0), 1e-30)
    assert drift < 1e-12, drift


# ================================================================
#  Family / taught gates
# ================================================================
def test_immersed_model_installs_maskstate():
    model = hy.Model(grid=_immersed_grid(_flat_bottom), dt=0.01,
                     advection=False)
    assert any(type(m).__name__ == "MaskState" for m in model.modules)


def test_unimmersed_model_has_no_maskstate():
    model = hy.Model(grid=_short_grid(), dt=0.01, advection=False)
    assert not any(
        type(m).__name__ == "MaskState" for m in model.modules)


def test_eigenmodes_reject_immersed():
    model = hy.Model(
        grid=_immersed_grid(_flat_bottom), dt=0.01, advection=False,
        free_surface=hy.ExplicitFreeSurface())
    with pytest.raises(NotImplementedError, match="immersed"):
        hy.eigenmodes.from_model(model)
    with pytest.raises(NotImplementedError, match="immersed"):
        hy.eigenmodes.HydrostaticEigenmodes(model)


def test_transforms_reject_immersed():
    model = hy.Model(
        grid=_immersed_grid(_flat_bottom), dt=0.01, advection=False,
        free_surface=hy.ExplicitFreeSurface())
    with pytest.raises(NotImplementedError, match="immersed"):
        hy.transforms.VorticalProjection.from_model(model)


def test_biased_advection_rejected_on_immersed():
    with pytest.raises(NotImplementedError, match="immersed"):
        hy.Model(
            grid=_immersed_grid(_flat_bottom), dt=0.01,
            advection=fr.model.modules.UpwindAdvection(3))
