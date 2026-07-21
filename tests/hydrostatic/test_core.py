"""The hydrostatic dynamical core: the two DIAGNOSE stages + balance."""
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.operators.cumulative import CumulativeIntegral
from fridom.spatial.spaces.constant import ConstantSpace

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=6, depth=2.0):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def make_model(grid=None, *, n2=2.0, csqr=3.0, f0=1.3, dt=1e-3):
    """Return a minimal linear hydrostatic model (advection=False)."""
    if grid is None:
        grid = make_grid()
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=csqr),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=hy.FPlaneCoriolis(f0=f0),
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)


# ================================================================
#  Manufactured DIAGNOSE: the diagnosed w (continuity, face form)
# ================================================================
def test_diagnosed_w_matches_the_cumulative_integral():
    depth, nz = 2.0, 6
    model = make_model(make_grid(nz=nz, depth=depth))
    zc = 0.5 * depth
    su = model.state["u"].function_space
    sv = model.state["v"].function_space
    sb = model.state["b"].function_space
    u = model.grid.create_field(
        su, init=lambda x, y, z: jnp.sin(2 * jnp.pi * x) + 0 * y + 0 * z)
    v = model.grid.create_field(
        sv, init=lambda x, y, z: jnp.cos(2 * jnp.pi * y) + 0 * x + 0 * z)
    b = model.grid.create_field(
        sb, init=lambda x, y, z: jnp.sin(2 * jnp.pi * x) * (z - zc)
        + 0 * y)

    core = model.module(hy.Core)
    state = model.state.replace(u=u, v=v, b=b)
    w = core._diagnose_w(state, None)["w"]

    div_h = u.diff("x") + v.diff("y")
    w_exact = -CumulativeIntegral(
        direction="up", target="face")["z"](div_h)
    assert np.allclose(np.asarray(w.data), np.asarray(w_exact.data),
                       atol=1e-13, rtol=0.0)
    # the fundamental theorem: d_z w == -(d_x u + d_y v)
    ft = np.asarray(w.diff("z").data) + np.asarray(div_h.data)
    assert np.abs(ft).max() < 1e-12


# ================================================================
#  Manufactured DIAGNOSE: the diagnosed p_hyd (half-cell center)
# ================================================================
def test_diagnosed_p_hyd_matches_the_cumulative_integral():
    depth, nz = 2.0, 6
    model = make_model(make_grid(nz=nz, depth=depth))
    zc = 0.5 * depth
    sb = model.state["b"].function_space
    b = model.grid.create_field(
        sb, init=lambda x, y, z: jnp.sin(2 * jnp.pi * x) * (z - zc)
        + 0 * y)

    core = model.module(hy.Core)
    state = model.state.replace(b=b)
    p_hyd = core._diagnose_p_hyd(state, None)["p_hyd"]

    p_exact = -CumulativeIntegral(
        direction="down", target="center")["z"](b)
    assert np.allclose(np.asarray(p_hyd.data), np.asarray(p_exact.data),
                       atol=1e-13, rtol=0.0)
    # the top cell carries exactly the half-cell hydrostatic pressure
    dz = depth / nz
    top = np.asarray(p_hyd.data)[..., -1]
    b_top = np.asarray(b.data)[..., -1]
    assert np.abs(top + 0.5 * dz * b_top).max() < 1e-13


# ================================================================
#  First-step (S1') placement: fresh w feeds the buoyancy tendency
# ================================================================
def test_first_step_buoyancy_reads_the_freshly_diagnosed_w():
    # the DIAGNOSE stages run before the stratification term consumes
    # w, so db/dt equals -N^2 * w_exact interpolated onto the b cell
    n2 = 2.0
    model = make_model(n2=n2)
    rng = np.random.default_rng(4)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=rng.standard_normal(model.state["b"].shape),
        ps=np.zeros(model.state["ps"].shape))
    state = model.state
    div_h = state["u"].diff("x") + state["v"].diff("y")
    w_exact = -CumulativeIntegral(
        direction="up", target="face")["z"](div_h)
    expected = -(n2 * w_exact.to(state["b"]))

    dX = model.tendency(state)
    assert np.allclose(np.asarray(dX["b"].data),
                       np.asarray(expected.data), atol=1e-12, rtol=0.0)


# ================================================================
#  Geostrophic steady state (barotropic null eigenvector)
# ================================================================
def _mode(model, name, kx, ky):
    """Single (kx, ky) horizontal Fourier mode at name's nodes."""
    fs = model.state[name].function_space
    var = tuple(n for f in fs.bare.factors
                if not isinstance(f, ConstantSpace) for n in f.names)

    def make(phase):
        trig = np.cos if phase == "cos" else np.sin

        def init(**c):
            return trig(2 * np.pi * (kx * c["x"] + ky * c["y"])) \
                + 0.0 * sum(c.values())
        init.__signature__ = inspect.Signature(
            [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
             for n in var])
        return model.grid.create_field(fs, init=init)

    return make


def _reduced_operator(model, fields, kx, ky):
    """Return the linear operator restricted to one horizontal mode."""
    basis = [(f, p) for f in fields for p in ("cos", "sin")]
    modes = {(f, p): _mode(model, f, kx, ky)(p) for f, p in basis}
    norms = {k: float((v.data * v.data).sum()) for k, v in modes.items()}
    n = len(basis)
    mat = np.zeros((n, n))
    for i, (f, p) in enumerate(basis):
        st = model.state
        for g in fields:
            zero = model.grid.create_field(
                model.state[g].function_space,
                data=0.0 * model.state[g].data)
            st = st.replace(**{g: zero})
        st = st.replace(**{f: modes[(f, p)]})
        dX = model.tendency(st)
        for j, (g, p2) in enumerate(basis):
            proj = float((np.asarray(dX[g].data)
                          * np.asarray(modes[(g, p2)].data)).sum())
            mat[j, i] = proj / norms[(g, p2)]
    return mat, basis, modes


def test_geostrophic_null_eigenvector_is_steady():
    grid = fr.spatial.Grid((
        IM(16, (0.0, 1.0), periodic=True, name="x"),
        IM(16, (0.0, 1.0), periodic=True, name="y"),
        IM(4, (0.0, 1.0), periodic=False, name="z")))
    model = make_model(grid, n2=0.0, csqr=4.0, f0=0.8, dt=2e-3)
    fields = ["u", "v", "ps"]
    mat, basis, modes = _reduced_operator(model, fields, 1, 1)

    evals, evecs = np.linalg.eig(mat)
    i0 = int(np.argmin(np.abs(evals)))
    assert abs(evals[i0]) < 1e-12  # a genuine geostrophic null mode

    # reconstruct the geostrophic eigenvector state
    st = model.state
    for g in fields:
        st = st.replace(**{g: model.grid.create_field(
            model.state[g].function_space,
            data=0.0 * model.state[g].data)})
    for coeff, (f, p) in zip(evecs[:, i0].real, basis, strict=True):
        st = st.replace(**{f: st[f] + coeff * modes[(f, p)]})

    amp = max(float(np.abs(st[f].data).max()) for f in fields)
    dX = model.tendency(st)
    tend = max(float(np.abs(dX[f].data).max()) for f in fields)
    assert tend / amp < 1e-10

    # and it stays put over a short integration
    model.set_state(st)
    u0 = np.asarray(model.state["u"].data).copy()
    model.advance(20)
    u1 = np.asarray(model.state["u"].data)
    assert np.abs(u1 - u0).max() / amp < 1e-10


# ================================================================
#  Constructor validation (the horizontal geometry names)
# ================================================================
@pytest.mark.parametrize(
    "horizontal",
    [
        pytest.param(("x",), id="one-name"),
        pytest.param(("x", "x"), id="duplicate"),
        pytest.param(("x", "y", "z"), id="three-names"),
    ],
)
def test_core_rejects_a_bad_horizontal(horizontal):
    with pytest.raises(TypeError, match="horizontal"):
        hy.Core(horizontal=horizontal)
