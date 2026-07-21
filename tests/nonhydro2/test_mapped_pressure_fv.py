r"""FV (CellAvg) family of the mapped PCG pressure solve (stage F5).

Prefix-mirrored shard of ``mapped_pressure.py`` for the finite-volume
C-grid: the family-aware corner cross form (:meth:`_to_cell`), the SPD
license on the average family, the mapped-flat identity, the
flux-consistent projection, and the FV<->nodal parity (on a
terrain-following grid the two discretizations are the *same numbers*,
because the mapped column rides uniform computational meshes and the
FV/nodal 2nd-order stencils are bit-identical). Self-contained: the
small builders mirror ``test_mapped_pressure.py``.
"""
import jax.numpy as jnp
import numpy as np

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.modules.advection import CenteredAdvection
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.nonhydro2.modules.mapped_pressure import (
    MappedPressureSolver,
)
from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.krylov import (
    _computational_integral,
    _computational_mean,
)
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodeSet

N = 16
H0 = 0.7
DSQR = 0.25


def depth(x):
    """Smooth periodic water depth H(x) (20% slope)."""
    return 1.0 + 0.2 * jnp.sin(x)


def build_grid(n=N, init=depth):
    """Terrain-following 2D grid ``zp = sigma * H(x)`` (uniform mesh)."""
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = IntervalMesh(n, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": init})
    return Grid((mx, ms), mapping=mapping), mx, ms


def build_solver(family, n=N, init=depth, **kwargs):
    """Build a family-specific mapped pressure solver on the column."""
    grid, mx, ms = build_grid(n, init)
    if family == "fv":
        grid.merge_overrides(fv_cgrid_overrides(grid.factors))
        space = mx.cell_avg * ms.cell_avg
    else:
        space = mx.center * ms.center
    kwargs.setdefault("iterations", 20)
    kwargs.setdefault("weights", {"sigma": 1.0 / DSQR})
    return MappedPressureSolver(grid, space, **kwargs), grid, mx, ms


def build_fv(n=N, init=depth, **kwargs):
    return build_solver("fv", n=n, init=init, **kwargs)


def dot(a, b):
    """Return the measure-weighted inner product CG uses."""
    return float(jnp.sum(_computational_integral(a * b).data))


def fv_space(mx, ms):
    return mx.cell_avg * ms.cell_avg


# ================================================================
#  Column discovery on the average family
# ================================================================
def test_column_discovery_fv():
    solver, *_ = build_fv()
    assert solver.base == "sigma"
    assert solver.mapped == "zp"
    assert solver.coupled == ("x",)
    assert solver.axes == ("x", "sigma")


# ================================================================
#  The SPD license on the FV C-grid
# ================================================================
def test_operator_is_exactly_symmetric_fv():
    # the corner cross form's transpose pairing holds on the FV family
    # too (the CellAvg <-> Right / CellAvg <-> Inner two-point means are
    # exact transposes under the uniform computational cell measure)
    solver, grid, mx, ms = build_fv()
    space = fv_space(mx, ms)
    p = grid.random.normal(space, seed=1)
    q = grid.random.normal(space, seed=2)
    left = dot(solver.apply(p), q)
    right = dot(p, solver.apply(q))
    assert abs(left - right) <= 1e-12 * abs(left)


def test_operator_is_negative_semidefinite_fv():
    solver, grid, mx, ms = build_fv()
    space = fv_space(mx, ms)
    for seed in (3, 4, 5):
        p = grid.random.normal(space, seed=seed)
        assert dot(solver.apply(p), p) < 0.0


def test_operator_annihilates_constants_exactly_fv():
    solver, grid, mx, ms = build_fv()
    one = grid.create_field(
        fv_space(mx, ms),
        init=lambda x, sigma: 1.0 + 0.0 * x * sigma)
    assert float(jnp.abs(solver.apply(one).data).max()) == 0.0


# ================================================================
#  FV <-> nodal parity: the operators are the SAME numbers
# ================================================================
def test_fv_operator_is_bitwise_nodal():
    # the FV mapped operator is bit-identical to the nodal one on a
    # terrain-following grid: same 2nd-order stencils, same uniform
    # computational measure, same metric coefficients (nodal faces).
    # The corner face->cell hop differs only in the codomain tag
    # (CellAvg vs Center) — same two-point-mean data.
    fv, grid_fv, mx_fv, ms_fv = build_fv()
    nod, grid_nod, mx_nod, ms_nod = build_solver("nodal")
    data = np.asarray(grid_fv.random.normal(
        fv_space(mx_fv, ms_fv), seed=7).data)
    p_fv = grid_fv.create_field(
        fv_space(mx_fv, ms_fv),
        init=lambda x, sigma: 0.0 * x * sigma).with_data(
            jnp.asarray(data))
    p_nod = grid_nod.create_field(
        mx_nod.center * ms_nod.center,
        init=lambda x, sigma: 0.0 * x * sigma).with_data(
            jnp.asarray(data))
    got = np.asarray(fv.apply(p_fv).data)
    want = np.asarray(nod.apply(p_nod).data)
    assert np.array_equal(got, want)


# ================================================================
#  Mapped-flat coefficient fold (constant H)
# ================================================================
def flat_depth(x):
    return H0 + 0.0 * x


def test_constant_h_folds_to_the_weighted_flat_laplacian_fv():
    # K^xx = H0, K^bb = w/H0, zero cross terms: A collapses to the
    # walled FV Div @ Diag @ Grad at the constant folded weights
    solver, grid, mx, ms = build_fv(init=flat_depth)
    space = fv_space(mx, ms)
    p = grid.random.normal(space, seed=6)
    registry = grid.dispatch
    gx = registry.resolve("diff", mx.cell_avg)["x"](p)
    gs = registry.resolve("diff", ms.cell_avg)["sigma"](p)
    fs = (((1.0 / DSQR) / H0) * gs).retag(
        ms.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    dx = registry.resolve("diff", mx.right)["x"]
    ds = registry.resolve(
        "diff", fs.function_space.factor("sigma"))["sigma"]
    want = dx(H0 * gx) + ds(fs)
    got = solver.apply(p)
    scale = float(jnp.abs(want.data).max())
    assert float(jnp.abs(got.data - want.data).max()) < 1e-14 * scale


def test_constant_h_preconditioner_is_exact_fv():
    # the folded-coefficient spectral inverse IS the operator inverse on
    # a constant-metric mapping: one FV iteration converges
    solver, grid, mx, ms = build_fv(init=flat_depth, iterations=1)
    rhs = grid.random.normal(fv_space(mx, ms), seed=7)
    rhs = rhs - _computational_mean(rhs)
    p, _info = solver.krylov().solve(rhs)
    residual = solver.apply(p) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-12


# ================================================================
#  The preconditioned FV solve
# ================================================================
def test_solve_converges_on_a_sloped_column_fv():
    # fixed-iteration mode: pinned for determinism
    solver, grid, mx, ms = build_fv(tolerance=None)
    rhs = grid.random.normal(fv_space(mx, ms), seed=9)
    rhs = rhs - _computational_mean(rhs)
    p = solver.solve(rhs)
    residual = solver.apply(p) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-10
    assert float(jnp.abs(_computational_mean(p).data.ravel()[0])) < 1e-12


def test_pcg_convergence_is_resolution_independent_fv():
    # a fixed FV iteration budget drives the sloped-column residual to
    # the same small level at two resolutions (the preconditioned CG
    # iteration count is roughly resolution-independent, as for nodal)
    residuals = []
    for n in (16, 32):
        # fixed-iteration mode: pinned for determinism
        solver, grid, mx, ms = build_fv(n=n, iterations=25, tolerance=None)
        rhs = grid.random.normal(fv_space(mx, ms), seed=9)
        rhs = rhs - _computational_mean(rhs)
        p = solver.solve(rhs)
        residuals.append(
            float(jnp.abs((solver.apply(p) - rhs).data).max())
            / float(jnp.abs(rhs.data).max()))
    assert all(r < 1e-9 for r in residuals)


def test_pcg_residual_matches_nodal_bitwise():
    # the FV and nodal operators + preconditioners are the same
    # numbers, so the CG iterates agree to machine precision. The FV
    # measure-weighted inner product reassociates with the storage
    # width (its pairwise-reduction tree depends on the padded length),
    # so once two-sided halo accounting narrows the FV grid to width 1
    # the FV and nodal reductions differ by ~2 ULP (nodal is width-
    # invariant; on the pre-tightening width 2 the two were bit-equal).
    # The physics is unchanged -- the residual gate above still holds --
    # so the cross-family gate is to machine precision, not the bit.
    fv, grid_fv, mx_fv, ms_fv = build_fv(iterations=12)
    nod, grid_nod, mx_nod, ms_nod = build_solver("nodal", iterations=12)
    data = np.asarray(grid_fv.random.normal(
        fv_space(mx_fv, ms_fv), seed=9).data)
    data = data - data.mean()  # host-side numpy mean (not a field verb)
    rhs_fv = grid_fv.create_field(
        fv_space(mx_fv, ms_fv),
        init=lambda x, sigma: 0.0 * x * sigma).with_data(
            jnp.asarray(data))
    rhs_nod = grid_nod.create_field(
        mx_nod.center * ms_nod.center,
        init=lambda x, sigma: 0.0 * x * sigma).with_data(
            jnp.asarray(data))
    assert np.allclose(np.asarray(fv.solve(rhs_fv).data),
                       np.asarray(nod.solve(rhs_nod).data),
                       rtol=0.0, atol=1e-14)


# ================================================================
#  The projection identity on the FV C-grid
# ================================================================
def random_velocity(grid, mx, ms, seeds=(1, 2)):
    """Build a staggered FV (u, w) pair, w wall-normal Dirichlet."""
    u = grid.random.normal(mx.right * ms.cell_avg, seed=seeds[0])
    w = grid.random.normal(
        mx.cell_avg * ms.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
        seed=seeds[1])
    return {"x": u, "sigma": w}


def test_projection_removes_the_measured_divergence_fv():
    # fixed-iteration mode: pinned for determinism
    solver, grid, mx, ms = build_fv(tolerance=None)
    vel = random_velocity(grid, mx, ms)
    div = solver.divergence(vel)
    p = solver.solve(div)
    corr = solver.velocity_correction(p)
    projected = {
        "x": vel["x"] - corr["x"].retag(vel["x"]),
        "sigma": vel["sigma"] - corr["sigma"].retag(vel["sigma"]),
    }
    after = solver.divergence(projected)
    rel = (float(jnp.abs(after.data).max())
           / float(jnp.abs(div.data).max()))
    assert rel < 1e-12


def test_divergence_vanishes_on_a_physical_streamfunction_flow_fv():
    # the FV analogue of the nodal manufactured-solution gate (stage F5,
    # gate 5): psi = sin(pi z / H) cos(x) is a boundary-conforming
    # streamfunction whose physical flow is divergence-free; the
    # J-weighted FV divergence converges to zero at 2nd order (interior)
    def u_fn(x, sigma):
        return (np.pi * jnp.cos(np.pi * sigma) * jnp.cos(x)
                / depth(x))

    def w_fn(x, sigma):
        slope = sigma * 0.2 * jnp.cos(x) / depth(x)
        return (jnp.sin(np.pi * sigma) * jnp.sin(x)
                + slope * np.pi * jnp.cos(np.pi * sigma)
                * jnp.cos(x))

    interior = []
    for n in (32, 64):
        solver, grid, mx, ms = build_fv(n=n)
        u = grid.create_field(mx.right * ms.cell_avg, init=u_fn)
        w = grid.create_field(
            mx.cell_avg * ms.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
            init=w_fn)
        div = np.abs(np.asarray(
            solver.divergence({"x": u, "sigma": w}).data))
        interior.append(div[:, 1:-1].max())
    assert np.log2(interior[0] / interior[1]) > 1.7


# ================================================================
#  The Core mapped FV projection branch (gates 1, 8)
# ================================================================
def make_mapped_fv_model(n=8, init=depth, dt=0.02, advection=False,
                         dsqr=1.0, **core_kwargs):
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": init})
    grid = Grid((mx, my, mz), mapping=mapping)
    core_kwargs.setdefault("pressure_iterations", 16)
    return nh.Model(
        grid=grid,
        core=nh.Core(family="fv", aspect_ratio=dsqr ** 0.5,
                     **core_kwargs),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=advection)


def _seed(model):
    hor = (np.arange(8) + 0.5) * (2 * np.pi / 8)
    ver = (np.arange(8) + 0.5) / 8
    x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
    model.set_fields(u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x),
                     b=0.01 * np.cos(np.pi * z))


def test_core_projects_on_a_mapped_fv_grid():
    # gate 1: explicit family="fv" on a terrain-following (walled-z)
    # grid assembles and steps; b lands on CellAvg
    # fixed-iteration mode: pinned for determinism
    model = make_mapped_fv_model(dsqr=DSQR, pressure_tolerance=None)
    assert all(isinstance(f, CellAvg)
               for f in model.state["b"].function_space.bare.factors)
    _seed(model)
    model.advance(1)
    assert not model.panicked
    # the CONSTRAINT-projected state is divergence-free in the mapped
    # FV sense (to the stage's CG residual)
    solver = MappedPressureSolver(
        model.state["u"].grid, model.state["p"].function_space,
        iterations=1)
    div = solver.divergence({
        "x": model.state["u"], "y": model.state["v"],
        "z": model.state["w"]})
    assert float(jnp.abs(div.data).max()) < 1e-10


def test_mapped_fv_model_steps_nonlinear():
    # gate 1: the nonlinear FV mapped model (centered advection) steps
    model = make_mapped_fv_model(dsqr=DSQR, advection=True)
    _seed(model)
    model.advance(3)
    assert not model.panicked
    for c in ("u", "v", "w", "b"):
        assert np.isfinite(np.asarray(model.state[c].data)).all()


def test_mapped_fv_model_conserves_physical_buoyancy():
    # gate 6: the FV mapped model conserves the physical (volume-
    # weighted) buoyancy content to machine zero over the advective
    # tendency -- the FV headline property on genuine terrain
    model = make_mapped_fv_model(dsqr=DSQR, advection=True)
    rng = np.random.default_rng(12)
    model.set_fields(**{c: rng.standard_normal(model.state[c].data.shape)
                        for c in ("u", "v", "w", "b")})
    state = model.constrain(model.state)
    tau = model.tendency(
        state, constraints=False,
        filter=fr.model.term_predicates.owned_by(CenteredAdvection))
    # the physical (volume-weighted) buoyancy content is now the plain
    # seeded verb on this maps= terrain grid (the physical-integral-
    # default flip): tau["b"].integrate() already carries the column
    # Jacobian, so hand-multiplying dzp_dz would double-count
    weighted = float(np.asarray(
        tau["b"].integrate().data).ravel()[0])
    scale = float(np.sum(np.abs(np.asarray(tau["b"].data))))
    assert abs(weighted) < 1e-11 * (scale + 1.0)


def test_mapped_fv_linear_matches_nodal_over_12_steps():
    # gate 2/5 (cross-family, mapped-flat analogue): the LINEAR mapped
    # model is bit-identical between FV and nodal up to jit fusion — the
    # projection operator is the same numbers, and the buoyancy coupling
    # (b.to(w), w.to(b)) crosses the same co-located two-point means. The
    # NONLINEAR model diverges only in b (the FV advection is genuinely
    # conservative, not consistent — gate 6), so this pins the linear
    # trajectory where the two families must agree.
    def build(family):
        mx = IntervalMesh(8, (0.0, 2 * np.pi), periodic=True, name="x")
        my = IntervalMesh(8, (0.0, 2 * np.pi), periodic=True, name="y")
        mz = IntervalMesh(8, (0.0, 1.0), periodic=False, name="z")
        mapping = CoordinateMapping(
            maps={"zp": lambda z, H: z * H}, params={"H": depth})
        grid = Grid((mx, my, mz), mapping=mapping)
        return nh.Model(
            grid=grid,
            core=nh.Core(
                aspect_ratio=(DSQR) ** 0.5,
                family=family,
                pressure_iterations=16),
            time_stepper=AdamBashforth(0.02, order=3),
            coriolis=nh.FPlaneCoriolis(f0=1.0),
            stratification=nh.ConstantStratification(n2=1.0),
            advection=False)

    fv = build("fv")
    nodal = build("nodal")
    rng = np.random.default_rng(4)
    ic = {c: rng.standard_normal(fv.state[c].data.shape)
          for c in ("u", "v", "w", "b")}
    fv.set_fields(**ic)
    nodal.set_fields(**ic)
    fv.advance(12)
    nodal.advance(12)
    for c in ("u", "v", "w", "b", "p"):
        a = np.asarray(fv.state[c].data)
        b = np.asarray(nodal.state[c].data)
        scale = max(float(np.abs(b).max()), 1e-30)
        assert np.abs(a - b).max() <= 1e-11 * scale, c


def test_mapped_fv_model_treedef_stable():
    model = make_mapped_fv_model(dsqr=DSQR, advection=True)
    import jax  # noqa: PLC0415
    _seed(model)
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(3)
    assert jax.tree_util.tree_structure(
        model._carry) == before
    assert not model.panicked
