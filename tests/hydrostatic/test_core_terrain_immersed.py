r"""Masked contravariant continuity on a terrain + immersed grid (M5).

Prefix-mirrored shard of ``hy.modules.core`` covering the composed
``_diagnose_w`` path: on a grid carrying **both** a terrain-following
(sigma) column and an immersed (cut-cell) domain the diagnosed ``w`` is
the masked contravariant volume flux ``J\omega`` — the min-rule face
fraction weights the ``J``-weighted horizontal transport, the running
integral yields ``alpha_z J\omega`` and ``w`` is the guarded division.
The single-descriptor cores live in ``test_core.py`` (flat),
``test_core_terrain.py`` (terrain) and ``test_core_immersed.py`` (flat
immersed). Self-contained per the AGENTS oversized-module rule.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.context import StepContext
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

IM = IntervalMesh
JN = "dzp_dz"
CTX = StepContext(params={}, clock=jnp.asarray(0.0), dt=jnp.asarray(0.01),
                  stage_dt=jnp.asarray(0.01))


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _mapping(a):
    """Sigma map ``zp = z H(x, y)`` with terrain amplitude ``a``."""
    def depth(x, y):
        return 1.0 + a * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    return CoordinateMapping(maps={"zp": lambda z, H: z * H},
                             params={"H": depth})


def _cut(x, y, z):  # noqa: ARG001
    """Return a genuine sloped partial bottom in computational z."""
    return jnp.clip((z - (-0.5 + 0.1 * jnp.sin(2 * jnp.pi * x))) / (1.0 / 8)
                    + 0.5, 0.0, 1.0)


def _allwet(x, y, z):  # noqa: ARG001
    return x * 0.0 + 1.0


def _terrain_immersed_grid(*, n=8, nz=8, a=0.4, order=4, init=_cut,
                           min_fraction=0.1):
    """Doubly-periodic horizontal sigma grid with an immersed cut."""
    return Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mapping(a),
        immersed=ImmersedDomain(init, order=order,
                                min_fraction=min_fraction))


def _pure_terrain_grid(*, n=8, nz=8, a=0.4):
    """Return the unimmersed terrain twin (same sigma map, no mask)."""
    return Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mapping(a))


def _model(grid, *, dt=0.01):
    return hy.Model(
        grid=grid, dt=dt, csqr=3.0,
        stratification=hy.ConstantStratification(n2=1.0),
        coriolis=hy.FPlaneCoriolis(f0=0.5), advection=False,
        free_surface=hy.ImplicitFreeSurface(pressure_iterations=30),
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=2))


def _masked_residual(model, u0, v0):
    r"""Return (masked contravariant continuity residual, w on closed faces).

    Diagnose ``w`` from a mask-consistent velocity and evaluate the full
    masked contravariant 3-D divergence ``(alpha_x Ju).diff(x) +
    (alpha_y Jv).diff(y) + (alpha_z w).diff(z)`` on the wet cells (``w``
    is the contravariant flux ``J\omega`` here). A jitted closure so the
    ``order>=2`` quadrature fractions concretize.
    """
    imm = model.grid.immersed
    core = model.module(hy.HydrostaticCore)

    @jax.jit
    def run(ud, vd):
        st = model.state
        mu = imm.mask(st["u"].function_space)
        mv = imm.mask(st["v"].function_space)
        st = st.replace(u=st["u"].with_data(ud * mu.data),
                        v=st["v"].with_data(vd * mv.data))
        w = core._diagnose_w(st, CTX)["w"]
        st = st.replace(w=w)
        ax = imm.fraction(st["u"].function_space)
        ay = imm.fraction(st["v"].function_space)
        az = core._masked_w_faces(imm, st)
        grid = st["u"].grid
        ju = st["u"] * grid.metric(st["u"].function_space.bare, JN)
        jv = st["v"] * grid.metric(st["v"].function_space.bare, JN)
        div = ((ax * ju).diff("x") + (ay * jv).diff("y")
               + (az * w).diff("z"))
        theta = imm.fraction(st["b"].function_space)
        return div.data, w.data, az.data, theta.data

    div, wdat, az, theta = (np.asarray(a) for a in run(
        jnp.asarray(u0), jnp.asarray(v0)))
    wet = theta > 0.0
    return (float(np.abs(div[wet]).max()),
            float(np.abs(wdat[az == 0.0]).max()))


# ================================================================
#  Gate: masked contravariant continuity is machine zero on wet cells
# ================================================================
@pytest.mark.parametrize(
    "a", [pytest.param(0.4, id="mild"), pytest.param(0.8, id="steep")])
def test_masked_contravariant_continuity_is_machine_zero(a):
    model = _model(_terrain_immersed_grid(a=a))
    rng = np.random.default_rng(0)
    shape = model.state["u"].data.shape
    residual, w_closed = _masked_residual(
        model, 0.3 * rng.standard_normal(shape),
        0.3 * rng.standard_normal(shape))
    # the face-form cumulative integral makes the masked contravariant
    # divergence telescope to machine zero on every wet cell
    assert residual < 1e-13, (a, residual)
    assert w_closed == 0.0            # Jomega == 0 on every closed face


# ================================================================
#  All-wet chart reproduces the pure terrain contravariant w (bitwise)
# ================================================================
def test_all_wet_chart_matches_pure_terrain_bytewise():
    im = _model(_terrain_immersed_grid(init=_allwet, min_fraction=0.0),
                dt=0.02)
    un = _model(_pure_terrain_grid(), dt=0.02)
    rng = np.random.default_rng(7)
    ic = {k: 0.3 * rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v", "b")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    im.advance(1)
    un.advance(1)
    diff = np.abs(np.asarray(im.state["w"].data)
                  - np.asarray(un.state["w"].data)).max()
    assert diff == 0.0               # alpha == 1 -> byte-identical


# ================================================================
#  Identity chart (a=0, J==1) + mask == flat immersed diagnosis
# ================================================================
def test_identity_chart_mask_matches_flat_immersed():
    def flat_immersed():
        return Grid(
            (IM(8, (0.0, 1.0), periodic=True, name="x"),
             IM(8, (0.0, 1.0), periodic=True, name="y"),
             IM(8, (-1.0, 0.0), periodic=False, name="z")),
            immersed=ImmersedDomain(_cut, order=4, min_fraction=0.1))
    ch = _model(_terrain_immersed_grid(a=0.0), dt=0.02)
    fl = _model(flat_immersed(), dt=0.02)
    rng = np.random.default_rng(3)
    ic = {k: 0.2 * rng.standard_normal(ch.state[k].data.shape)
          for k in ("u", "v", "b")}
    ch.set_fields(**ic)
    fl.set_fields(**ic)
    ch.advance(1)
    fl.advance(1)
    # J == 1: the contravariant flux equals the flat masked continuity
    diff = np.abs(np.asarray(ch.state["w"].data)
                  - np.asarray(fl.state["w"].data)).max()
    assert diff < 1e-13


# ================================================================
#  extra_halo: terrain slope + masked continuity compose (z=2)
# ================================================================
def test_core_extra_halo_terrain_immersed():
    model = _model(_terrain_immersed_grid(a=0.4))
    core = model.module(hy.HydrostaticCore)
    # the terrain slope gradient carries the vertical reach (z=2), the
    # masked continuity is a vertical reduction (reach 0); horizontal 1
    assert dict(core.extra_halo.widths) == {"x": 1, "y": 1, "z": 2}


# ================================================================
#  Taught error: a collocation-order mask on a chart (order=None)
# ================================================================
def test_collocation_mask_on_a_chart_is_a_taught_error():
    grid = _terrain_immersed_grid(order=None, min_fraction=0.0)
    with pytest.raises(NotImplementedError,
                       match="genuine per-cell quadrature"):
        _model(grid)
