"""The implicit free surface: CONSTRAINT-stage 2D projection (H3).

Prefix-mirrored shard of ``hy.modules.free_surface`` covering the
``ImplicitFreeSurface`` variant (the explicit variant lives in
``test_free_surface.py``); self-contained per the AGENTS oversized-module
rule (small builders duplicated).
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.params import GRAVITY
from fridom.model.context import StepContext
from fridom.model.errors import LinearOperatorGapError
from fridom.model.implicit import VerticalDiffusion
from fridom.model.model import _chunk_body
from fridom.model.module import Module
from fridom.model.terms import Treatment
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.spaces.constant import ConstantSpace

IM = fr.spatial.meshes.IntervalMesh


# ================================================================
#  Builders (duplicated per the self-contained-shard rule)
# ================================================================
def make_grid(nx, nz, depth=1.0):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def _zextent(grid):
    """Physical vertical extent of the grid (the flat column depth)."""
    lo, hi = next(m.extent for m in grid.factors if "z" in m.names)
    return float(hi - lo)


def make_model(grid, free_surface, *, n2=0.0, csqr=4.0, f0=0.0,
               dt=1e-2, stepper=None):
    """Return a linear hydrostatic model on the given free surface.

    ``csqr`` keeps the legacy barotropic parameterization c^2 = g*H:
    the gravity handed to the core is csqr / H(grid).
    """
    if stepper is None:
        stepper = AdamBashforth(dt, order=3)
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=csqr / _zextent(grid)),
        time_stepper=stepper,
        coriolis=hy.FPlaneCoriolis(f0=f0),
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=free_surface,
        advection=False)


def k_disc_sq(n_mode, n_cells, length=1.0):
    """Return the squared discrete wavenumber of the C-grid difference."""
    dx = length / n_cells
    k = 2.0 * np.pi * n_mode / length
    return (2.0 * np.sin(k * dx / 2.0) / dx) ** 2


def mode_field(model, name, kx, ky, phase):
    """cos/sin(kx x + ky y), depth-uniform, at name's nodes."""
    fs = model.state[name].function_space
    var = tuple(n for f in fs.bare.factors
                if not isinstance(f, ConstantSpace) for n in f.names)
    trig = np.cos if phase == "cos" else np.sin

    def init(**c):
        return trig(2 * np.pi * (kx * c["x"] + ky * c["y"])) \
            + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return model.grid.create_field(fs, init=init)


def zeroed(model, fields):
    """Return the model's state with the named fields zeroed."""
    st = model.state
    for g in fields:
        st = st.replace(**{g: model.grid.create_field(
            model.state[g].function_space,
            data=0.0 * model.state[g].data)})
    return st


# ================================================================
#  Constructor / assembly
# ================================================================
def test_factory_assembles_both_variants():
    grid = make_grid(8, 4)
    free = make_model(grid, hy.ImplicitFreeSurface())
    rigid = make_model(grid, hy.ImplicitFreeSurface(epsilon=0.0))
    # ps present in both; PROGNOSTIC (free) vs DIAGNOSTIC (rigid lid)
    assert "ps" in free.state.component_names
    assert "ps" in rigid.state.component_names


def test_epsilon_property_and_default():
    assert hy.ImplicitFreeSurface().epsilon == 1.0
    assert hy.ImplicitFreeSurface(epsilon=0.0).epsilon == 0.0
    assert hy.ImplicitFreeSurface(epsilon=0.3).epsilon == 0.3


@pytest.mark.parametrize(
    "epsilon",
    [pytest.param(-1.0, id="negative"),
     pytest.param(-1e-9, id="tiny-negative"),
     pytest.param("1", id="string"),
     pytest.param(True, id="bool")],
)
def test_epsilon_validation_rejects(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        hy.ImplicitFreeSurface(epsilon=epsilon)


@pytest.mark.parametrize(
    "horizontal",
    [pytest.param(("x",), id="one-name"),
     pytest.param(("x", "x"), id="duplicate"),
     pytest.param(("x", "y", "z"), id="three-names")],
)
def test_implicit_rejects_a_bad_horizontal(horizontal):
    with pytest.raises(TypeError, match="horizontal"):
        hy.ImplicitFreeSurface(horizontal=horizontal)


# ================================================================
#  Phase C: the terrain multigrid knobs (validation + defaults)
# ================================================================
def test_pressure_preconditioner_default_and_property():
    assert hy.ImplicitFreeSurface().pressure_preconditioner == "spectral"
    assert hy.ImplicitFreeSurface(
        pressure_preconditioner="multigrid").pressure_preconditioner \
        == "multigrid"


def test_multigrid_levels_default_and_property():
    assert hy.ImplicitFreeSurface().multigrid_levels is None
    assert hy.ImplicitFreeSurface(multigrid_levels=3).multigrid_levels == 3


@pytest.mark.parametrize(
    "preconditioner",
    [pytest.param("jacobi", id="unknown"),
     pytest.param("Spectral", id="miscased"),
     pytest.param(None, id="none")],
)
def test_pressure_preconditioner_validation_rejects(preconditioner):
    with pytest.raises(ValueError, match="pressure_preconditioner"):
        hy.ImplicitFreeSurface(pressure_preconditioner=preconditioner)


@pytest.mark.parametrize(
    "levels",
    [pytest.param(0, id="zero"),
     pytest.param(-2, id="negative"),
     pytest.param(2.0, id="float"),
     pytest.param(True, id="bool")],
)
def test_multigrid_levels_validation_rejects(levels):
    with pytest.raises(ValueError, match="multigrid_levels"):
        hy.ImplicitFreeSurface(multigrid_levels=levels)


def test_ps_lifecycle_depends_on_epsilon():
    prog = hy.ImplicitFreeSurface(epsilon=1.0).field_declarations[0]
    diag = hy.ImplicitFreeSurface(epsilon=0.0).field_declarations[0]
    assert prog.lifecycle is fr.model.Lifecycle.PROGNOSTIC
    assert diag.lifecycle is fr.model.Lifecycle.DIAGNOSTIC


# ================================================================
#  The refactor: each variant owns BOTH sides of its coupling
# ================================================================
def test_core_pressure_gradient_no_longer_reads_ps():
    # HY-D3 refactor: the core reads only the baroclinic p_hyd
    refs = {r.name for r in hy.Core().field_references}
    assert "ps" not in refs
    assert "b" in refs


def test_explicit_free_surface_owns_the_momentum_term():
    # ExplicitFreeSurface now carries BOTH -c^2 div(ubar) (advances ps)
    # AND -grad(ps) (advances u, v)
    grid = make_grid(8, 4)
    model = make_model(grid, hy.ExplicitFreeSurface(), n2=0.0, f0=0.0)
    # -grad(ps) reaches the momentum: seed ps, read the u/v tendency
    st = zeroed(model, ["u", "v", "b", "ps"])
    st = st.replace(ps=mode_field(model, "ps", 1, 0, "cos"))
    dX = model.tendency(st)
    assert float(np.abs(dX["u"].data).max()) > 1e-6   # -d_x ps drives u
    assert float(np.abs(dX["ps"].data).max()) < 1e-12  # div(ubar)=0 here


# ================================================================
#  The backward-Euler amplification factor (analytic)
# ================================================================
def _constraint_map(model, kx, ky, fields):
    """Build the linear CONSTRAINT map on one horizontal mode."""
    basis = [(f, p) for f in fields for p in ("cos", "sin")]
    modes = {(f, p): mode_field(model, f, kx, ky, p) for f, p in basis}
    norms = {k: float((v.data * v.data).sum()) for k, v in modes.items()}
    n = len(basis)
    mat = np.zeros((n, n))
    for i, (f, p) in enumerate(basis):
        st = zeroed(model, ["u", "v", "b", "ps"])
        st = st.replace(**{f: modes[(f, p)]})
        out = model.constrain(st)
        for j, (g, p2) in enumerate(basis):
            proj = float((np.asarray(out[g].data)
                          * np.asarray(modes[(g, p2)].data)).sum())
            mat[j, i] = proj / norms[(g, p2)]
    return mat


def test_backward_euler_amplification_factor():
    nx, csqr, dt = 16, 4.0, 0.05
    grid = make_grid(nx, 4)
    model = make_model(grid, hy.ImplicitFreeSurface(epsilon=1.0),
                       n2=0.0, csqr=csqr, f0=0.0, dt=dt)
    kx, ky = 2, 0
    mat = _constraint_map(model, kx, ky, ["u", "ps"])
    eig = np.linalg.eigvals(mat)

    # omega_disc from the discrete C-grid symbol
    kd2 = k_disc_sq(kx, nx) + k_disc_sq(ky, nx)
    omega = np.sqrt(csqr * kd2)
    gain = 1.0 / np.sqrt(1.0 + omega**2 * dt**2)   # backward-Euler |G|
    phase = np.arctan(omega * dt)                  # and its phase

    comp = eig[np.abs(eig.imag) > 1e-9]
    assert comp.size >= 2  # the +/- omega gravity pair
    assert np.allclose(np.abs(comp), gain, rtol=1e-9)
    assert np.allclose(np.abs(np.angle(comp)), phase, rtol=1e-9)


# ================================================================
#  Small-dt convergence: implicit (BE) vs explicit oracle -> slope 1
# ================================================================
def test_small_dt_first_order_convergence_to_the_oracle():
    nx, csqr, T = 16, 1.0, 0.5
    grid = make_grid(nx, 4)
    kx = 1
    errs = []
    for n_steps in (20, 40, 80):
        dt = T / n_steps
        exp = make_model(grid, hy.ExplicitFreeSurface(),
                         n2=0.0, csqr=csqr, dt=dt)
        imp = make_model(grid, hy.ImplicitFreeSurface(epsilon=1.0),
                         n2=0.0, csqr=csqr, dt=dt)
        for m in (exp, imp):
            m.set_fields(ps=mode_field(m, "ps", kx, 0, "cos").data)
            m.run(steps=n_steps)
        d = (np.asarray(exp.state["ps"].data)
             - np.asarray(imp.state["ps"].data))
        errs.append(float(np.sqrt((d * d).sum())))
    errs = np.asarray(errs)
    slopes = np.log2(errs[:-1] / errs[1:])
    # backward Euler is first order in the surface mode: slope -> 1
    assert errs[-1] < errs[0]
    assert slopes[-1] > 0.9


# ================================================================
#  Unconditional stability at sqrt(csqr)-CFL >> 1
# ================================================================
def test_stable_far_beyond_the_explicit_cfl():
    nx, csqr = 16, 100.0
    dx = 1.0 / nx
    dt = 50.0 * dx / np.sqrt(csqr)      # sqrt(csqr) * dt / dx = 50
    assert np.sqrt(csqr) * dt / dx > 50.0 - 1e-9
    grid = make_grid(nx, 4)
    model = make_model(grid, hy.ImplicitFreeSurface(epsilon=1.0),
                       n2=0.0, csqr=csqr, f0=0.5, dt=dt)
    rng = np.random.default_rng(2)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        ps=rng.standard_normal(model.state["ps"].shape))

    def energy(m):
        s = m.state
        return float(sum((np.asarray(s[g].data) ** 2).sum()
                         for g in ("u", "v")))

    e0 = energy(model)
    model.advance(200)
    e1 = energy(model)
    # backward Euler damps but never amplifies: bounded and smooth
    assert np.isfinite(e1)
    assert e1 <= e0 * (1.0 + 1e-9)
    assert e1 > 0.0


# ================================================================
#  Rigid lid (epsilon = 0): divergence-free depth mean + Poisson ps
# ================================================================
def _depth_mean_div(state, depth):
    """Return the depth-mean horizontal divergence on the ps cell."""
    div_h = state["u"].diff("x") + state["v"].diff("y")
    return Integral()["z"](div_h) * (1.0 / depth)


def test_rigid_lid_projects_the_depth_mean_divergence_free():
    depth = 2.0
    grid = make_grid(16, 4, depth=depth)
    model = make_model(grid, hy.ImplicitFreeSurface(epsilon=0.0),
                       n2=0.0, csqr=3.0, f0=0.5, dt=0.05)
    rng = np.random.default_rng(1)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))

    # a single constraint removes a large divergence to machine zero
    fs = model.module(hy.ImplicitFreeSurface)
    ctx = StepContext(params={GRAVITY: jnp.asarray(3.0 / depth)},
                      clock=jnp.asarray(0.0), dt=jnp.asarray(0.05),
                      stage_dt=jnp.asarray(0.05))
    out = fs._barotropic_solve(model.state, ctx)
    pre = float(np.max(np.abs(
        np.asarray(_depth_mean_div(model.state, depth).data))))
    post_state = model.state.replace(u=out["u"], v=out["v"])
    post = float(np.max(np.abs(
        np.asarray(_depth_mean_div(post_state, depth).data))))
    uscale = float(np.max(np.abs(np.asarray(out["u"].data))))
    assert pre > 1.0                       # a genuine divergence to remove
    assert post / uscale < 1e-11           # projected to machine zero

    # and it stays non-divergent every step over a short bounded run
    for _ in range(5):
        model.advance(1)
        s = model.state
        rel = float(np.max(np.abs(
            np.asarray(_depth_mean_div(s, depth).data)))) / max(
            float(np.max(np.abs(np.asarray(s["u"].data)))), 1e-30)
        assert rel < 1e-11


def test_rigid_lid_ps_is_the_surface_pressure_poisson_solution():
    nx, depth, csqr, dt = 16, 2.0, 3.0, 0.1
    grid = make_grid(nx, 4, depth=depth)
    model = make_model(grid, hy.ImplicitFreeSurface(epsilon=0.0),
                       n2=0.0, csqr=csqr, f0=0.4, dt=dt)
    rng = np.random.default_rng(5)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))

    div2d = np.asarray(
        _depth_mean_div(model.state, depth).data).reshape(nx, nx)
    fs = model.module(hy.ImplicitFreeSurface)
    ctx = StepContext(params={GRAVITY: jnp.asarray(csqr / depth)},
                      clock=jnp.asarray(0.0), dt=jnp.asarray(dt),
                      stage_dt=jnp.asarray(dt))
    ps2d = np.asarray(
        fs._barotropic_solve(model.state, ctx)["ps"].data
        ).reshape(nx, nx)

    # independent numpy C-grid FFT Poisson: nabla^2 ps = div_bar / dt
    dx = 1.0 / nx
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    kdx = (2.0 * np.sin(kx * dx / 2.0) / dx) ** 2
    lap = kdx[:, None] + kdx[None, :]              # discrete -laplacian
    rhs_hat = np.fft.fft2(div2d / dt)
    ps_hat = np.zeros_like(rhs_hat)
    mask = lap > 1e-12
    ps_hat[mask] = -rhs_hat[mask] / lap[mask]
    ps_oracle = np.real(np.fft.ifft2(ps_hat))

    ps2d = ps2d - ps2d.mean()                      # both mean-free (gauge)
    ps_oracle = ps_oracle - ps_oracle.mean()
    err = np.max(np.abs(ps2d - ps_oracle))
    assert err / np.max(np.abs(ps_oracle)) < 1e-11


# ================================================================
#  Geostrophic steady state under BOTH variants
# ================================================================
def _reduced_tendency(model, fields, kx, ky):
    """Build the reduced tendency operator on one horizontal mode."""
    basis = [(f, p) for f in fields for p in ("cos", "sin")]
    modes = {(f, p): mode_field(model, f, kx, ky, p) for f, p in basis}
    norms = {k: float((v.data * v.data).sum()) for k, v in modes.items()}
    n = len(basis)
    mat = np.zeros((n, n))
    for i, (f, p) in enumerate(basis):
        st = zeroed(model, ["u", "v", "b", "ps"])
        st = st.replace(**{f: modes[(f, p)]})
        dX = model.tendency(st)
        for j, (g, p2) in enumerate(basis):
            proj = float((np.asarray(dX[g].data)
                          * np.asarray(modes[(g, p2)].data)).sum())
            mat[j, i] = proj / norms[(g, p2)]
    return mat, basis, modes


def test_geostrophic_steady_state_exact_under_both_variants():
    grid = make_grid(16, 4)
    csqr, f0 = 4.0, 0.8
    fields = ["u", "v", "ps"]
    # build the geostrophic null eigenvector from the EXPLICIT operator
    exp = make_model(grid, hy.ExplicitFreeSurface(), csqr=csqr, f0=f0,
                     dt=2e-3)
    mat, basis, modes = _reduced_tendency(exp, fields, 1, 1)
    evals, evecs = np.linalg.eig(mat)
    i0 = int(np.argmin(np.abs(evals)))
    assert abs(evals[i0]) < 1e-12

    for free_surface in (hy.ExplicitFreeSurface(),
                         hy.ImplicitFreeSurface(epsilon=1.0)):
        m = make_model(grid, free_surface, csqr=csqr, f0=f0, dt=2e-3)
        st = zeroed(m, fields)
        for coeff, (f, p) in zip(evecs[:, i0].real, basis, strict=True):
            st = st.replace(**{f: st[f] + coeff * modes[(f, p)]})
        amp = max(float(np.abs(st[f].data).max()) for f in fields)
        m.set_state(st)
        u0 = np.asarray(m.state["u"].data).copy()
        p0 = np.asarray(m.state["ps"].data).copy()
        m.advance(20)
        du = np.abs(np.asarray(m.state["u"].data) - u0).max() / amp
        dp = np.abs(np.asarray(m.state["ps"].data) - p0).max() / amp
        assert du < 1e-10
        assert dp < 1e-10


# ================================================================
#  The linear-operator honesty gate (HY-D7)
# ================================================================
def test_implicit_declares_the_linear_operator_gap():
    grid = make_grid(8, 4)
    implicit = make_model(grid, hy.ImplicitFreeSurface(epsilon=1.0),
                          f0=0.5)
    explicit = make_model(grid, hy.ExplicitFreeSurface(), f0=0.5)
    # the implicit variant is refused (surface pressure is NOT in L)
    with pytest.raises(LinearOperatorGapError, match="surface-pressure"):
        fr.model.require_linear_operator(implicit, consumer="test")
    # the explicit variant passes (its barotropic terms ARE in L)
    fr.model.require_linear_operator(explicit, consumer="test")


def test_rigid_lid_also_declares_the_gap():
    grid = make_grid(8, 4)
    rigid = make_model(grid, hy.ImplicitFreeSurface(epsilon=0.0))
    with pytest.raises(LinearOperatorGapError):
        fr.model.require_linear_operator(rigid, consumer="test")


# ================================================================
#  Stepper composition smokes
# ================================================================
@pytest.mark.parametrize(
    "stepper_factory",
    [pytest.param(
        lambda dt: AdamBashforth(
            dt, order=2, eps=0.1), id="ab2-eps0.1-pyom"),
     pytest.param(
        lambda dt: AdamBashforth(dt, order=3),
        id="ab3"),
     pytest.param(
        fr.model.time_steppers.LowStorageRK3, id="rk3")],
)
@pytest.mark.parametrize("epsilon", [1.0, 0.0])
def test_stepper_composition_smokes(stepper_factory, epsilon):
    grid = make_grid(16, 4)
    dt = 1e-2
    model = make_model(
        grid, hy.ImplicitFreeSurface(epsilon=epsilon),
        n2=1.0, csqr=4.0, f0=0.5, dt=dt, stepper=stepper_factory(dt))
    rng = np.random.default_rng(0)
    fields = {
        "u": rng.standard_normal(model.state["u"].shape),
        "v": rng.standard_normal(model.state["v"].shape),
        "b": rng.standard_normal(model.state["b"].shape)}
    if epsilon > 0:
        fields["ps"] = rng.standard_normal(model.state["ps"].shape)
    model.set_fields(**fields)
    model.advance(10)
    umax = float(np.max(np.abs(np.asarray(model.state["u"].data))))
    assert np.isfinite(umax)


# ================================================================
#  CNAB2 mixing-solve then surface constraint (the mixed IMEX path)
# ================================================================
def _test_kappa(_module, _state, _ctx, _name):
    return 0.05


class _VertMix(Module):

    """Minimal in-test implicit vertical mixing on u, v.

    The 2.5 reference vertical-diffusion consumer is test-only, so
    this module exercises the solve-then-surface-constraint
    composition directly.
    """

    field_references = (
        fr.model.FieldReference("u", hint="core"),
        fr.model.FieldReference("v", hint="core"),
    )

    @fr.model.term(name="mix", treatment=Treatment.IMPLICIT,
                   implicit=VerticalDiffusion("z", ("u", "v"),
                                              _test_kappa))
    def mix(self, _state, _ctx):
        return {}


def test_cnab2_vertical_diffusion_then_surface_constraint():
    # CNAB2 solves the vertical mixing (S3) then the ImplicitFreeSurface
    # runs its CONSTRAINT (S4): the mixed "solve then surface" path
    grid = make_grid(16, 6)
    model = hy.Model(
        grid=grid,
        core=hy.Core(gravity=4.0),
        time_stepper=fr.model.time_steppers.CNAB2(1e-2),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ImplicitFreeSurface(epsilon=1.0),
        advection=False,
        modules_extra=(_VertMix(),))
    rng = np.random.default_rng(3)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=rng.standard_normal(model.state["b"].shape),
        ps=rng.standard_normal(model.state["ps"].shape))
    import jax  # noqa: PLC0415
    td0 = jax.tree_util.tree_structure(model.carry)
    model.advance(20)
    td1 = jax.tree_util.tree_structure(model.carry)
    umax = float(np.max(np.abs(np.asarray(model.state["u"].data))))
    assert td0 == td1                 # composition treedef stable
    assert np.isfinite(umax)


# ================================================================
#  Walled horizontal grids (walled-horizontal gap, fix 2 of 3):
#  the CONSTRAINT-stage spectral / CG solve on a bounded axis
# ================================================================
# (periodic-x, periodic-y) for each walled-horizontal configuration
WALLS = {
    "x": (False, True),
    "y": (True, False),
    "xy": (False, False),
}


def walled_grid(periodic, nx=8, ny=8, nz=4, depth=1.0):
    """Return a horizontally (partly) walled, bounded-z grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=periodic[0], name="x"),
        IM(ny, (0.0, 1.0), periodic=periodic[1], name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def walled_model(grid, *, epsilon=1.0, csqr=1.0, f0=0.5, n2=0.0, dt=1e-3,
                 pressure_iterations=30):
    """Return a linear implicit-free-surface hydrostatic model."""
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=csqr),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=hy.FPlaneCoriolis(f0=f0),
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=hy.ImplicitFreeSurface(
            epsilon=epsilon,
            pressure_iterations=pressure_iterations),
        advection=False)


@pytest.mark.parametrize("wall", list(WALLS), ids=list(WALLS))
@pytest.mark.parametrize("epsilon", [1.0, 0.0])
def test_implicit_assembles_and_runs_finite_on_walls(wall, epsilon):
    # the barotropic Div @ Diag @ Grad solve expands on the Neumann-tagged
    # wall sibling (before the fix it keyed the absent bare Inner(x) diff
    # row and failed to assemble); both variants now run finite
    model = walled_model(walled_grid(WALLS[wall]), epsilon=epsilon, f0=1.0)
    rng = np.random.default_rng(0)
    keys = ("u", "v", "b", "ps") if epsilon > 0 else ("u", "v", "b")
    model.set_fields(**{k: 0.1 * rng.standard_normal(model.state[k].shape)
                        for k in keys})
    model.advance(10)
    assert not model.panicked
    for k in ("u", "v", "b"):
        data = np.asarray(model.state[k].data)
        assert bool(np.isfinite(data).all()), (wall, epsilon, k)


@pytest.mark.parametrize("wall", list(WALLS), ids=list(WALLS))
def test_rigid_lid_gauge_lands_on_the_constant_mode_on_walls(wall):
    r"""``eps == 0`` zeros the DC (cosine) mode on a walled trig axis.

    The ``where_zero=0.0`` gauge removes the ``k = 0`` spectral
    coefficient; on a walled (Neumann / Cosine-II) axis that mode is the
    constant, so the rigid-lid surface pressure comes out plain-mean-free
    exactly as on a periodic axis. Measured ``|mean| / scale``: ``0``
    (single wall) / ``~1.3e-17`` (xy); pinned well above.
    """
    model = walled_model(walled_grid(WALLS[wall]), epsilon=0.0, csqr=3.0,
                         f0=0.5, dt=0.05)
    rng = np.random.default_rng(1)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape),
                     v=rng.standard_normal(model.state["v"].shape))
    fs = model.module(hy.ImplicitFreeSurface)
    # unit-depth walled grid: gravity = csqr / H = 3.0
    ctx = StepContext(params={GRAVITY: jnp.asarray(3.0)},
                      clock=jnp.asarray(0.0), dt=jnp.asarray(0.05),
                      stage_dt=jnp.asarray(0.05))
    ps = fs._barotropic_solve(model.state, ctx)["ps"]
    data = np.asarray(ps.data)
    assert bool(np.isfinite(data).all())
    scale = float(np.max(np.abs(data)))
    assert scale > 0.1                              # a genuine solution
    assert abs(float(np.mean(data))) / scale < 1e-14


def test_implicit_ps_volume_conserved_on_walls():
    r"""``int(ps)`` drifts only at round-off on a walled implicit run.

    The barotropic operator conserves the measure-weighted surface-
    pressure integral: ``A ps = eps ps - dt'^2 div(c^2 grad ps)``, whose
    divergence leg telescopes to zero on no-flux walls, so
    ``int(A ps) = eps int(ps)`` and the constant mode carries through; the
    right-hand-side divergence integrates to zero on the walls, so
    ``int(ps)`` is invariant. Measured drift ``~8.7e-19``; pinned above.
    """
    model = walled_model(walled_grid(WALLS["xy"]), epsilon=1.0, csqr=2.0,
                         f0=0.5)
    rng = np.random.default_rng(0)
    model.set_fields(**{k: 0.1 * rng.standard_normal(model.state[k].shape)
                        for k in ("u", "v", "b", "ps")})

    def volume():
        return float(jnp.sum(model.state["ps"].integrate().data))

    before = volume()
    model.advance(20)
    after = volume()
    assert not model.panicked
    assert abs(after - before) < 1e-14 * max(abs(before), 1.0)


# ================================================================
#  Mirror-symmetry physics gate (the channel image trick)
# ================================================================
def _mirror_grids(nx, ny=3, nz=4):
    """Return (walled-x channel, doubled periodic-x) grid pair.

    The walled channel stores ``nx`` cells over ``[0, 1]``; the doubled
    domain stores ``2*nx`` cells over ``[0, 2]`` at the identical
    ``dx = 1/nx`` so the two C-grids collocate on ``[0, 1]``.
    """
    walled = fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=False, name="x"),
        IM(ny, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))
    doubled = fr.spatial.Grid((
        IM(2 * nx, (0.0, 2.0), periodic=True, name="x"),
        IM(ny, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))
    return walled, doubled


def test_implicit_channel_matches_the_mirror_image_run():
    r"""A walled-x implicit channel equals its doubled periodic mirror.

    Solid walls at ``x = 0, 1`` are the reflection symmetry of a
    doubly-long periodic domain with a mirror-symmetric state: cell
    scalars (``ps``) even-extended, the wall-normal velocity (``u``)
    odd-extended (the two wall faces forced to zero). The CONSTRAINT-stage
    projection (``f = 0``, ``N^2 = 0``, no advection) preserves the
    symmetry exactly, so the walled run reproduces the periodic run
    restricted to ``[0, 1]`` to round-off — the same C-grid correspondence
    proven for the explicit variant (``test_free_surface_walled.py``).
    Measured drift: ``ps ~3.2e-15``, ``u ~4.7e-16``; pinned honestly above.
    """
    nx, ny, nz, steps, dt = 6, 3, 4, 12, 2e-3
    walled, doubled = _mirror_grids(nx, ny, nz)
    mw = walled_model(walled, csqr=1.0, dt=dt)
    mp = walled_model(doubled, csqr=1.0, dt=dt)

    # walled initial condition (uniform in y, z), a genuine wave state
    ps_cells = np.cos(np.pi * (np.arange(nx) + 0.5) / nx) + 0.3
    u_faces = 0.2 * np.sin(np.pi * np.arange(1, nx) / nx)  # nx-1 faces
    wps = np.zeros((nx, ny, 1))
    wps[:, :, 0] = ps_cells[:, None]
    wu = np.zeros((nx - 1, ny, nz))
    wu[:] = u_faces[:, None, None]
    mw.set_fields(ps=wps, u=wu)

    # even extension of ps, odd extension of u (wall faces zeroed)
    ps_ext = np.concatenate([ps_cells, ps_cells[::-1]])
    u_ext = np.concatenate([u_faces, [0.0], -u_faces[::-1], [0.0]])
    pps = np.zeros((2 * nx, ny, 1))
    pps[:, :, 0] = ps_ext[:, None]
    pu = np.zeros((2 * nx, ny, nz))
    pu[:] = u_ext[:, None, None]
    mp.set_fields(ps=pps, u=pu)

    mw.advance(steps)
    mp.advance(steps)
    assert not mw.panicked
    assert not mp.panicked

    wps_f = np.asarray(mw.state["ps"].data)
    pps_f = np.asarray(mp.state["ps"].data)
    wu_f = np.asarray(mw.state["u"].data)
    pu_f = np.asarray(mp.state["u"].data)
    # ps cells and u interior faces restricted to the channel [0, 1]
    ps_drift = np.abs(wps_f - pps_f[:nx]).max()
    u_drift = np.abs(wu_f - pu_f[:nx - 1]).max()
    assert ps_drift < 1e-12
    assert u_drift < 1e-12


# ================================================================
#  Immersed (cut-cell) mask on a walled horizontal grid
# ================================================================
def _coast(x, y, z):  # noqa: ARG001
    """Return a partial-coastline mask (open west, shelf elsewhere)."""
    return ((z > 0.25) | (x < 0.6)).astype(float)


def test_immersed_walled_rigid_lid_projects_divergence_free():
    r"""The masked barotropic CG converges on a walled immersed grid.

    The rigid-lid (``eps = 0``) constraint removes a large masked
    depth-mean divergence to near machine zero on a fully
    walled-horizontal cut-cell grid — the wall closure (the Dirichlet-
    tagged flux legs) and the wet-column mask compose. Measured residual:
    ``pre ~13``, ``post/uscale ~3.2e-9``; pinned at ``1e-7``.
    """
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=False, name="x"),
        IM(8, (0.0, 1.0), periodic=False, name="y"),
        IM(4, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(_coast))
    model = walled_model(grid, epsilon=0.0, csqr=2.0, f0=0.5, dt=0.05,
                         pressure_iterations=40)
    rng = np.random.default_rng(1)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape),
                     v=rng.standard_normal(model.state["v"].shape))
    fs = model.module(hy.ImplicitFreeSurface)
    # unit-depth immersed grid: gravity = csqr / H = 2.0
    ctx = StepContext(params={GRAVITY: jnp.asarray(2.0)},
                      clock=jnp.asarray(0.0), dt=jnp.asarray(0.05),
                      stage_dt=jnp.asarray(0.05))
    pre = float(np.max(np.abs(np.asarray(
        fs._depth_mean_div(model.state).data))))
    out = fs._barotropic_solve(model.state, ctx)
    post_state = model.state.replace(u=out["u"], v=out["v"])
    post = float(np.max(np.abs(np.asarray(
        fs._depth_mean_div(post_state).data))))
    uscale = float(np.max(np.abs(np.asarray(out["u"].data))))
    assert pre > 1.0                                # a genuine divergence
    assert post / uscale < 1e-7                     # projected to ~zero
    assert bool(np.isfinite(np.asarray(out["ps"].data)).all())


def test_immersed_walled_free_surface_runs_finite():
    """A walled immersed ``eps > 0`` implicit free surface stays finite."""
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=False, name="x"),
        IM(8, (0.0, 1.0), periodic=False, name="y"),
        IM(4, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(_coast))
    model = walled_model(grid, epsilon=1.0, csqr=1.0, f0=0.5,
                         pressure_iterations=40)
    rng = np.random.default_rng(0)
    model.set_fields(**{k: 0.1 * rng.standard_normal(model.state[k].shape)
                        for k in ("u", "v", "b", "ps")})
    model.advance(10)
    assert not model.panicked
    assert all(bool(np.isfinite(np.asarray(model.state[k].data)).all())
               for k in ("u", "v", "b"))


# ================================================================
#  Autodiff regression (differentiability policy)
# ================================================================
def test_grad_through_walled_implicit_run_matches_finite_difference():
    r"""``jax.grad`` w.r.t. the initial ``ps`` on a walled-x implicit run.

    The CONSTRAINT-stage spectral solve stays differentiable through the
    wall retag seam. Differentiate the pure kernel ``_chunk_body`` w.r.t.
    the initial surface pressure and match a random directional projection
    to a central finite difference (measured relerr ``~5e-12``).
    """
    model = walled_model(walled_grid(WALLS["x"]), epsilon=1.0, csqr=1.0,
                         f0=0.5, dt=2e-3)
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
        final = _chunk_body(record, 10, spliced, stepper)
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
#  Periodic fast-path regression pin (the wall arm must not perturb it)
# ================================================================
def test_periodic_flat_spectral_takes_the_no_retag_fast_path():
    r"""On a doubly-periodic grid the flat solve is the bare SpectralSolve.

    The Neumann-sibling of a doubly-periodic solve space is the space
    itself, so ``_flat_spectral`` short-circuits to ``SpectralSolve.solve``
    directly (a bound method) with no retag wrapper — the byte-identical
    fast path. A walled grid instead returns the plain retag closure.
    """
    grid = make_grid(8, 4)
    model = make_model(grid, hy.ImplicitFreeSurface(epsilon=1.0))
    fs = model.module(hy.ImplicitFreeSurface)
    space = model.state["ps"].function_space.bare
    solver = fs._flat_spectral(space, grid, column_csqr=4.0,
                               dt=1e-2)
    # the fast path returns the bound SpectralSolve.solve, not a closure
    assert isinstance(getattr(solver, "__self__", None), SpectralSolve)

    wgrid = walled_grid(WALLS["xy"])
    wmodel = walled_model(wgrid, epsilon=1.0)
    wfs = wmodel.module(hy.ImplicitFreeSurface)
    wsolver = wfs._flat_spectral(
        wmodel.state["ps"].function_space.bare, wgrid,
        column_csqr=1.0, dt=1e-3)
    assert getattr(wsolver, "__self__", None) is None
