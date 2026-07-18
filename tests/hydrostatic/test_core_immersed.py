"""Masked continuity (w diagnosis) on immersed grids (IP-D9, gate a).

Prefix-mirrored shard of ``hy.modules.core`` covering the cut-cell
``_diagnose_w`` path (the unimmersed core lives in ``test_core.py``);
self-contained per the AGENTS oversized-module rule.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.hydrostatic as hy
from fridom.model.context import StepContext
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

IM = IntervalMesh
CTX = StepContext(params={}, clock=jnp.asarray(0.0), dt=jnp.asarray(0.01),
                  stage_dt=jnp.asarray(0.01))


def grid(init, *, n=8, nz=8, order=None, min_fraction=0.0):
    """Doubly-periodic horizontal, bounded vertical immersed grid."""
    return Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(init, order=order,
                                min_fraction=min_fraction))


def plain_grid(n=8, nz=8):
    """Return the unimmersed twin (same meshes, no immersed domain)."""
    return Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (0.0, 1.0), periodic=False, name="z")))


def _masked_residual(model, u0, v0):
    """Return (residual on wet cells, w on closed faces) for given u, v.

    Diagnose ``w`` from the (mask-consistent) velocity and evaluate the
    full masked 3D divergence — a jitted closure so ``order>=2``
    quadrature fractions concretize.
    """
    imm = model.grid.immersed
    core = model.module(hy.HydrostaticCore)

    @jax.jit
    def run(ud, vd):
        st = model.state
        st = st.replace(u=st["u"].with_data(ud), v=st["v"].with_data(vd))
        # zero the dry velocity DOFs (the MaskState-consistent state)
        mu = imm.mask(st["u"].function_space)
        mv = imm.mask(st["v"].function_space)
        st = st.replace(u=st["u"].with_data(st["u"].data * mu.data),
                        v=st["v"].with_data(st["v"].data * mv.data))
        w = core._diagnose_w(st, CTX)["w"]
        st = st.replace(w=w)
        ax = imm.fraction(st["u"].function_space)
        ay = imm.fraction(st["v"].function_space)
        az = core._masked_w_faces(imm, st)
        div = ((ax * st["u"]).diff("x") + (ay * st["v"]).diff("y")
               + (az * w).diff("z"))
        theta = imm.fraction(st["b"].function_space)
        return div.data, w.data, az.data, theta.data

    div, wdat, az, theta = (np.asarray(a) for a in run(
        jnp.asarray(u0), jnp.asarray(v0)))
    wet = theta > 0.0
    return (float(np.abs(div[wet]).max()),
            float(np.abs(wdat[az == 0.0]).max()))


# ================================================================
#  Gate a: masked continuity residual is machine zero on wet cells
# ================================================================
@pytest.mark.parametrize(
    ("init", "order", "min_fraction", "label"),
    [pytest.param(lambda x, y, z: (z > 0.5).astype(float), None, 0.0,  # noqa: ARG005
                  "face-aligned", id="face-aligned"),
     pytest.param(lambda x, y, z: (z > 0.4).astype(float), None, 0.0,  # noqa: ARG005
                  "collocation", id="collocation-staircase"),
     pytest.param(
         lambda x, y, z: jnp.clip(  # noqa: ARG005
             (z - (0.3 + 0.1 * jnp.sin(2 * jnp.pi * x))) / (1.0 / 8)
             + 0.5, 0.0, 1.0),
         4, 0.0, "z-partial", id="z-partial-quadrature")],
)
def test_masked_continuity_residual_is_machine_zero(
        init, order, min_fraction, label):
    model = hy.Model(
        grid=grid(init, order=order, min_fraction=min_fraction),
        dt=0.01, advection=False)
    rng = np.random.default_rng(0)
    shape = model.state["u"].data.shape
    residual, w_closed = _masked_residual(
        model, 0.3 * rng.standard_normal(shape),
        0.3 * rng.standard_normal(shape))
    assert residual < 1e-13, (label, residual)   # masked continuity
    assert w_closed == 0.0                        # w == 0 on dry faces


def _old_masked_w_faces(core, immersed, state):
    """Frozen pre-machinery raw-``.data`` surgery (the bitwise anchor).

    A verbatim copy of the original ``_masked_w_faces`` moveaxis +
    row-set surgery, kept here as the reference the machinery rewrite
    must reproduce bit-for-bit (``modify_array`` inlined as ``.at``).
    """
    vertical = core._vertical
    w_space = state["w"].function_space
    alpha_z = immersed.fraction(w_space)
    theta_cell = immersed.fraction(state["p_hyd"].function_space)
    z_axis = next(
        i for i, f in enumerate(w_space.bare.factors)
        if vertical in f.names)
    az = jnp.moveaxis(alpha_z.data, z_axis, 0)
    tc = jnp.moveaxis(theta_cell.data, z_axis, 0)
    az = az.at[-1].set(tc[-1])  # surface face = surface cell
    return alpha_z.with_data(jnp.moveaxis(az, 0, z_axis))


# ================================================================
#  Machinery rewrite reproduces the raw-.data surgery bit-for-bit
# ================================================================
# NB: a stretched-*vertical* immersed grid is unreachable here — a
# mapped / terrain column on top of an immersed mask is a taught error
# (``HydrostaticCore.bind``) — so representativeness comes from the
# fraction pattern (walled step, quadrature partial, sloping genuine
# partials incl. a partial *surface* cell) and the column count.
@pytest.mark.parametrize(
    ("init", "n", "nz", "order", "min_fraction", "id_"),
    [pytest.param(lambda x, y, z: (z > 0.5).astype(float),  # noqa: ARG005
                  8, 8, None, 0.0, "walled-step", id="walled-step"),
     pytest.param(lambda x, y, z: (z > 0.4).astype(float),  # noqa: ARG005
                  6, 6, None, 0.0, "staircase", id="staircase"),
     pytest.param(
         lambda x, y, z: jnp.clip(  # noqa: ARG005
             (z - (0.3 + 0.1 * jnp.sin(2 * jnp.pi * x))) / (1.0 / 8)
             + 0.5, 0.0, 1.0),
         8, 8, 4, 0.0, "z-partial", id="z-partial"),
     pytest.param(
         lambda x, y, z: jnp.clip(  # noqa: ARG005
             (x - (0.35 + 0.1 * jnp.sin(2 * jnp.pi * y))) / (1.0 / 8)
             + 0.5, 0.0, 1.0),
         8, 4, 4, 0.1, "sidewall-partial", id="sidewall-partial"),
     pytest.param(
         lambda x, y, z: jnp.clip(  # noqa: ARG005
             (z - 0.85) / (1.0 / 4) + 0.5, 0.0, 1.0),
         4, 4, 4, 0.1, "partial-surface", id="partial-surface")],
)
def test_masked_w_faces_matches_raw_data_surgery_bitwise(
        init, n, nz, order, min_fraction, id_):
    """The machinery path reproduces the raw ``.data`` surgery exactly."""
    model = hy.Model(
        grid=grid(init, n=n, nz=nz, order=order,
                  min_fraction=min_fraction),
        dt=0.01, advection=False)
    imm = model.grid.immersed
    core = model.module(hy.HydrostaticCore)
    old = np.asarray(_old_masked_w_faces(core, imm, model.state).data)
    new = np.asarray(core._masked_w_faces(imm, model.state).data)
    assert np.array_equal(old, new), id_          # bitwise equivalence


def test_partial_surface_cell_is_a_true_partial():
    """The ``partial-surface`` case copies a genuine partial (not 0/1)."""
    model = hy.Model(
        grid=grid(lambda x, y, z: jnp.clip(  # noqa: ARG005
            (z - 0.85) / (1.0 / 4) + 0.5, 0.0, 1.0),
            n=4, nz=4, order=4, min_fraction=0.1),
        dt=0.01, advection=False)
    core = model.module(hy.HydrostaticCore)
    az = np.asarray(core._masked_w_faces(
        model.grid.immersed, model.state).data)
    surface = az[..., -1]
    assert ((surface > 1e-6) & (surface < 1.0 - 1e-6)).any()


def test_surface_face_is_a_genuine_dof_not_dry():
    # the physical surface face keeps alpha_z = surface-cell fraction
    # (the barotropic column-divergence carrier), NOT the dry-exterior 0
    model = hy.Model(
        grid=grid(lambda x, y, z: (z > 0.5).astype(float)),  # noqa: ARG005
        dt=0.01, advection=False)
    imm = model.grid.immersed
    core = model.module(hy.HydrostaticCore)
    az = np.asarray(core._masked_w_faces(
        imm, model.state).data)
    # bottom-most and interior dry faces are closed; the surface (last
    # z face) of a wet column is open
    assert az[0, 0, -1] == 1.0     # wet surface column: open surface
    assert az[0, 0, 4] == 0.0      # the immersed bottom face is closed


# ================================================================
#  All-wet immersed reproduces the unimmersed diagnosis (byte-exact)
# ================================================================
def test_all_wet_matches_unimmersed_bytewise():
    im = hy.Model(grid=grid(lambda x, y, z: x * 0.0 + 1.0),  # noqa: ARG005
                  dt=0.02, advection=False)
    un = hy.Model(grid=plain_grid(), dt=0.02, advection=False)
    rng = np.random.default_rng(7)
    ic = {k: 0.3 * rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v", "b")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    im.advance(1)
    un.advance(1)
    diff = np.abs(np.asarray(im.state["w"].data)
                  - np.asarray(un.state["w"].data)).max()
    assert diff == 0.0


# ================================================================
#  extra_halo: exempt only on an immersed grid
# ================================================================
def test_core_extra_halo_only_when_immersed():
    imm_model = hy.Model(
        grid=grid(lambda x, y, z: (z > 0.5).astype(float)),  # noqa: ARG005
        dt=0.01, advection=False)
    plain = hy.Model(grid=plain_grid(), dt=0.01, advection=False)
    assert isinstance(
        imm_model.module(hy.HydrostaticCore).extra_halo, HaloSpec)
    assert plain.module(hy.HydrostaticCore).extra_halo is None
