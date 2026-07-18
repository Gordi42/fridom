"""The implicit free surface: CONSTRAINT-stage 2D projection (H3).

Prefix-mirrored shard of ``hy.modules.free_surface`` covering the
``ImplicitFreeSurface`` variant (the explicit variant lives in
``test_free_surface.py``); self-contained per the AGENTS oversized-module
rule (small builders duplicated).
"""
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.params import CSQR
from fridom.model.context import StepContext
from fridom.model.errors import LinearOperatorGapError
from fridom.model.implicit import VerticalDiffusion
from fridom.model.module import Module
from fridom.model.terms import Treatment
from fridom.spatial.operators.integrate import Integral
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


def make_model(grid, free_surface, *, n2=0.0, csqr=4.0, f0=0.0,
               dt=1e-2, stepper=None):
    """Return a linear hydrostatic model on the given free surface."""
    if stepper is None:
        stepper = fr.model.time_steppers.AdamBashforth(dt, order=3)
    return hy.Model(
        grid=grid, dt=dt, csqr=csqr, free_surface=free_surface,
        stratification=hy.ConstantStratification(n2=n2),
        coriolis=hy.FPlaneCoriolis(f0=f0), advection=False,
        time_stepper=stepper)


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
    refs = {r.name for r in hy.HydrostaticCore().field_references}
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
    ctx = StepContext(params={CSQR: jnp.asarray(3.0)},
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
    ctx = StepContext(params={CSQR: jnp.asarray(csqr)},
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
        lambda dt: fr.model.time_steppers.AdamBashforth(
            dt, order=2, eps=0.1), id="ab2-eps0.1-pyom"),
     pytest.param(
        lambda dt: fr.model.time_steppers.AdamBashforth(dt, order=3),
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
        grid=grid, csqr=4.0,
        free_surface=hy.ImplicitFreeSurface(epsilon=1.0),
        stratification=hy.ConstantStratification(n2=1.0),
        coriolis=hy.FPlaneCoriolis(f0=0.5), advection=False,
        modules_extra=(_VertMix(),),
        time_stepper=fr.model.time_steppers.CNAB2(1e-2))
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
