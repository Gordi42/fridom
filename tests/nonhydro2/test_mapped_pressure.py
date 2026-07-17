"""Unit tests for the mapped PCG pressure solve (stage C3).

The SPD license of the flux-form mapped operator (exact symmetry,
negative semidefiniteness, the constants nullspace), the mapped-flat
coefficient fold, the flux-consistent projection identity, the
preconditioned solve, the dynamic-parameter seam, the taught
construction errors, and the DynamicalCore mapped projection branch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.nonhydro2.modules.mapped_pressure import (
    MappedPressureSolver,
)
from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.krylov import ConjugateGradient
from fridom.spatial.operators.multigrid import VerticalLineJacobi
from fridom.spatial.spaces.nodal import NodeSet

N = 16
H0 = 0.7
DSQR = 0.25


def depth(x):
    """Smooth periodic water depth H(x) (20% slope)."""
    return 1.0 + 0.2 * jnp.sin(x)


def build_grid(n=N, init=depth, periodic_x=True):
    """Terrain-following 2D grid, ``zp = sigma * H(x)``."""
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=periodic_x,
                      name="x")
    ms = IntervalMesh(n, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": init})
    return Grid((mx, ms), mapping=mapping), mx, ms


def build_solver(n=N, init=depth, **kwargs):
    grid, mx, ms = build_grid(n, init)
    space = mx.center * ms.center
    kwargs.setdefault("iterations", 20)
    kwargs.setdefault("weights", {"sigma": 1.0 / DSQR})
    return MappedPressureSolver(grid, space, **kwargs), grid, mx, ms


def dot(a, b):
    """Return the measure-weighted inner product CG uses."""
    return float(jnp.sum((a * b).integrate().data))


def random_velocity(grid, mx, ms, seeds=(1, 2)):
    """Build a staggered (u, w) pair, w wall-normal Dirichlet."""
    u = grid.random.normal(mx.right * ms.center, seed=seeds[0])
    w = grid.random.normal(
        mx.center * ms.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
        seed=seeds[1])
    return {"x": u, "sigma": w}


# ================================================================
#  Column discovery
# ================================================================
def test_column_discovery():
    solver, *_ = build_solver()
    assert solver.base == "sigma"
    assert solver.mapped == "zp"
    assert solver.coupled == ("x",)
    assert solver.axes == ("x", "sigma")
    assert solver.iterations == 20


# ================================================================
#  The SPD license: symmetry, sign, nullspace
# ================================================================
def test_operator_is_exactly_symmetric():
    # <A p, q> == <p, A q> in the measure-weighted product — the
    # corner cross form's transpose pairing (module docstring); the
    # tolerance is pure floating-point roundoff
    solver, grid, mx, ms = build_solver()
    space = mx.center * ms.center
    p = grid.random.normal(space, seed=1)
    q = grid.random.normal(space, seed=2)
    left = dot(solver.apply(p), q)
    right = dot(p, solver.apply(q))
    assert abs(left - right) <= 1e-12 * abs(left)


def test_operator_is_negative_semidefinite():
    solver, grid, mx, ms = build_solver()
    space = mx.center * ms.center
    for seed in (3, 4, 5):
        p = grid.random.normal(space, seed=seed)
        assert dot(solver.apply(p), p) < 0.0


def test_operator_annihilates_constants_exactly():
    solver, grid, mx, ms = build_solver()
    one = grid.create_field(
        mx.center * ms.center,
        init=lambda x, sigma: 1.0 + 0.0 * x * sigma)
    assert float(jnp.abs(solver.apply(one).data).max()) == 0.0


# ================================================================
#  Mapped-flat coefficient fold (constant H)
# ================================================================
def flat_depth(x):
    return H0 + 0.0 * x


def test_constant_h_folds_to_the_weighted_flat_laplacian():
    # K^xx = H0, K^bb = w/H0, zero cross terms: A collapses to the
    # hand-rolled walled Div @ Diag @ Grad at those constant weights
    # (the wall-normal flux Dirichlet-tagged, pressure.py's parity)
    solver, grid, mx, ms = build_solver(init=flat_depth)
    space = mx.center * ms.center
    p = grid.random.normal(space, seed=6)
    registry = grid.dispatch
    gx = registry.resolve("diff", mx.center)["x"](p)
    gs = registry.resolve("diff", ms.center)["sigma"](p)
    fs = (((1.0 / DSQR) / H0) * gs).retag(
        ms.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    dx = registry.resolve("diff", mx.right)["x"]
    ds = registry.resolve(
        "diff", fs.function_space.factor("sigma"))["sigma"]
    want = dx(H0 * gx) + ds(fs)
    got = solver.apply(p)
    scale = float(jnp.abs(want.data).max())
    assert float(jnp.abs(got.data - want.data).max()) < 1e-14 * scale


def test_constant_h_preconditioner_is_exact():
    # the folded-coefficient spectral inverse IS the operator's
    # inverse on a constant-metric mapping: one iteration converges
    solver, grid, mx, ms = build_solver(init=flat_depth,
                                        iterations=1)
    rhs = grid.random.normal(mx.center * ms.center, seed=7)
    rhs = rhs - rhs.mean()
    p, _info = solver.krylov().solve(rhs)
    residual = solver.apply(p) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-12


def test_constant_h_velocity_correction_matches_flat_gradient():
    # corr_x = dp/dx, corr_sigma = (1/dsqr) dp/dzp = (w/H0) dp/dsigma
    solver, grid, mx, ms = build_solver(init=flat_depth)
    space = mx.center * ms.center
    p = grid.random.normal(space, seed=8)
    corr = solver.velocity_correction(p)
    gx = grid.dispatch.resolve("diff", mx.center)["x"](p)
    gs = grid.dispatch.resolve("diff", ms.center)["sigma"](p)
    want_s = (1.0 / DSQR) / H0 * gs
    assert float(jnp.abs(corr["x"].data - gx.data).max()) \
        < 1e-14 * float(jnp.abs(gx.data).max())
    assert float(jnp.abs(corr["sigma"].data
                         - want_s.data).max()) \
        < 1e-14 * float(jnp.abs(want_s.data).max())


# ================================================================
#  The preconditioned solve
# ================================================================
def test_solve_converges_on_a_sloped_column():
    solver, grid, mx, ms = build_solver()
    rhs = grid.random.normal(mx.center * ms.center, seed=9)
    rhs = rhs - rhs.mean()
    p = solver.solve(rhs)
    residual = solver.apply(p) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-10
    # the solve pins the mean-free gauge
    assert float(jnp.abs(p.mean().data.ravel()[0])) < 1e-12


def test_single_precision_preconditioner_still_converges():
    # mixed-precision PCG: the f32 preconditioner only steers the
    # search directions — iterates, operator and residual arithmetic
    # stay f64, so the attainable residual is not f32-limited, the
    # convergence path is merely perturbed
    grid, mx, ms = build_grid()
    space = mx.center * ms.center
    kw = {"iterations": 20, "weights": {"sigma": 1.0 / DSQR}}
    full = MappedPressureSolver(grid, space, **kw)
    mixed = MappedPressureSolver(grid, space, single_precision=True,
                                 **kw)
    rhs = grid.random.normal(space, seed=9)
    rhs = rhs - rhs.mean()
    p_full = full.solve(rhs)
    p_mixed = mixed.solve(rhs)
    residual = mixed.apply(p_mixed) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-8
    diff = float(jnp.abs((p_mixed - p_full).data).max())
    scale = float(jnp.abs(p_full.data).max())
    assert diff / scale < 1e-6
    # the state precision is untouched
    assert p_mixed.dtype == p_full.dtype


def test_solve_accepts_an_initial_guess():
    solver, grid, mx, ms = build_solver(iterations=4)
    rhs = grid.random.normal(mx.center * ms.center, seed=10)
    rhs = rhs - rhs.mean()

    def rel_residual(p):
        residual = solver.apply(p) - rhs
        return (float(jnp.abs(residual.data).max())
                / float(jnp.abs(rhs.data).max()))

    cold = solver.solve(rhs)
    warm = solver.solve(rhs, cold)
    # restarting from the cold solution keeps converging
    assert rel_residual(warm) < 1e-3 * rel_residual(cold)


# ================================================================
#  The projection identity: divergence -> solve -> correction
# ================================================================
def test_projection_removes_the_measured_divergence():
    # the velocity update is derived from the operator's own fluxes,
    # so the post-projection divergence is the CG residual — not an
    # O(h^2) consistency remainder
    solver, grid, mx, ms = build_solver()
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


def test_divergence_vanishes_on_a_physical_streamfunction_flow():
    # psi(x, z) = sin(pi z / H) cos(x) is a boundary-conforming
    # streamfunction: u = psi_z, w = -psi_x|_z is physically
    # divergence-free and its normal flux vanishes at the sloped
    # boundary; the discrete J-weighted divergence converges to
    # zero at 2nd order
    def u_fn(x, sigma):
        return (np.pi * jnp.cos(np.pi * sigma) * jnp.cos(x)
                / depth(x))

    def w_fn(x, sigma):
        slope = sigma * 0.2 * jnp.cos(x) / depth(x)
        return (jnp.sin(np.pi * sigma) * jnp.sin(x)
                + slope * np.pi * jnp.cos(np.pi * sigma)
                * jnp.cos(x))

    interior, boundary = [], []
    for n in (32, 64):
        solver, grid, mx, ms = build_solver(n=n)
        u = grid.create_field(mx.right * ms.center, init=u_fn)
        w = grid.create_field(
            mx.center * ms.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
            init=w_fn)
        div = np.abs(np.asarray(
            solver.divergence({"x": u, "sigma": w}).data))
        interior.append(div[:, 1:-1].max())
        boundary.append(div[:, [0, -1]].max())
    # interior: 2nd order; wall rows: 1st order — the zero-flux
    # closure leaves the O(h^2) flux error of the last interior
    # face uncancelled in the wall cell's difference (the standard
    # conservative sigma-coordinate closure; the elliptic solve
    # smooths it, see the manufactured-solution validation gate)
    assert np.log2(interior[0] / interior[1]) > 1.7
    assert np.log2(boundary[0] / boundary[1]) > 0.9


def test_divergence_validates_the_component_keys():
    solver, grid, mx, ms = build_solver()
    u = grid.random.normal(mx.right * ms.center, seed=1)
    with pytest.raises(ValueError, match="one component per axis"):
        solver.divergence({"x": u})


def test_periodic_column_solves_without_retagging():
    # a fully periodic mapped column: the Neumann sibling IS the
    # pressure space (interned identity), the preconditioner skips
    # the retag round-trip, and the wall closure never engages
    mx = IntervalMesh(N, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = IntervalMesh(N, (0.0, 1.0), periodic=True, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": depth})
    grid = Grid((mx, ms), mapping=mapping)
    space = mx.center * ms.center
    solver = MappedPressureSolver(grid, space, iterations=20)
    rhs = grid.random.normal(space, seed=13)
    rhs = rhs - rhs.mean()
    p = solver.solve(rhs)
    residual = solver.apply(p) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-9
    # symmetry holds on the periodic column too
    q = grid.random.normal(space, seed=14)
    left = dot(solver.apply(p), q)
    right = dot(p, solver.apply(q))
    assert abs(left - right) <= 1e-12 * abs(left)


# ================================================================
#  Dynamic parameters (the C4 seam)
# ================================================================
def test_params_override_threads_through_the_operator():
    solver, grid, mx, ms = build_solver()
    space = mx.center * ms.center
    p = grid.random.normal(space, seed=11)
    h2 = grid.create_field(
        mx.center, init=lambda x: 1.0 + 0.1 * jnp.cos(x))
    other = MappedPressureSolver(
        grid, space, iterations=20,
        weights={"sigma": 1.0 / DSQR}, params={"H": h2})
    assert not jnp.allclose(solver.apply(p).data,
                            other.apply(p).data)
    # the overridden operator keeps the exact SPD structure
    q = grid.random.normal(space, seed=12)
    left = dot(other.apply(p), q)
    right = dot(p, other.apply(q))
    assert abs(left - right) <= 1e-12 * abs(left)


# ================================================================
#  Per-solve metric derivation (derived once, never cached)
# ================================================================
def count_metrics(monkeypatch, grid):
    """Count the ``grid.metric`` derivations of the solve."""
    calls = []
    real = type(grid).metric

    def counting(self, space, name, *, params=None):
        calls.append((space, name))
        return real(self, space, name, params=params)

    monkeypatch.setattr(type(grid), "metric", counting)
    return calls


def test_metric_derivation_is_independent_of_the_iterations(
        monkeypatch):
    # the metrics do not depend on the CG iterate: they are derived
    # ONCE per solve, so the derivation count must not grow with the
    # iteration budget (the regression this guards: re-deriving
    # inside apply() put a full metric chain — with its halo syncs —
    # in every CG iteration)
    counts = []
    for iterations in (2, 12):
        solver, grid, mx, ms = build_solver(iterations=iterations)
        rhs = grid.random.normal(mx.center * ms.center, seed=9)
        calls = count_metrics(monkeypatch, grid)
        solver.solve(rhs)
        counts.append(len(calls))
        monkeypatch.undo()
    assert counts[0] == counts[1]
    assert counts[0] > 0


def test_solve_matches_the_unmemoized_operator_to_rounding():
    # the memo is a pure trace-structure change: the CG iterates are
    # the same arithmetic on the same values. This was a *bitwise*
    # gate while the CG loop was unrolled. Since ROADMAP 3.6 the loop
    # is a lax.scan, and the memo now decides whether the metric
    # fields enter the scan body as hoisted constants (memoized) or
    # are recomputed inside it (not memoized) — two different body
    # computations, which XLA fuses and FMA-contracts differently.
    # The arithmetic is unchanged; the last bits are not. The measured
    # deviation is ~1 ulp (2.8e-17 absolute, 2.3e-16 relative), two
    # orders below the 6e-15 mapped-flat identity gate, so the claim
    # is now "identical to rounding" rather than "bitwise".
    solver, grid, mx, ms = build_solver(iterations=12)
    rhs = grid.random.normal(mx.center * ms.center, seed=15)
    rhs = rhs - rhs.mean()
    reference = ConjugateGradient(
        solver.apply,  # no cache: every application re-derives
        preconditioner=solver._preconditioner(),
        iterations=solver.iterations, project_mean=True)(rhs)
    assert np.allclose(np.asarray(solver.solve(rhs).data),
                       np.asarray(reference.data),
                       rtol=0.0, atol=1e-15)


def test_project_matches_the_separate_calls_bitwise():
    # DynamicalCore's projection runs divergence -> solve ->
    # correction on ONE shared derivation; the result must equal the
    # three separate calls exactly
    solver, grid, mx, ms = build_solver()
    vel = random_velocity(grid, mx, ms)
    p, corr = solver.project(vel)
    p_want = solver.solve(solver.divergence(vel))
    corr_want = solver.velocity_correction(p_want)
    assert np.array_equal(np.asarray(p.data),
                          np.asarray(p_want.data))
    for a in ("x", "sigma"):
        assert np.array_equal(np.asarray(corr[a].data),
                              np.asarray(corr_want[a].data)), a


def test_metrics_are_never_cached_across_solves(monkeypatch):
    # the C4 safety property: the memo lives inside ONE solve. A
    # second solve on the same solver re-derives every metric (a memo
    # kept on the solver — or on the grid — would silently freeze the
    # geometry of the first step under MovingGeometry)
    solver, grid, mx, ms = build_solver(iterations=4)
    rhs = grid.random.normal(mx.center * ms.center, seed=16)
    calls = count_metrics(monkeypatch, grid)
    solver.solve(rhs)
    first = len(calls)
    solver.solve(rhs)
    assert first > 0
    assert len(calls) == 2 * first


def test_moved_geometry_solves_differ_and_stay_correct():
    # two solves at DIFFERENT params= geometries (the shape a
    # MovingGeometry step takes) must give different answers, and
    # each must equal the solve on a grid whose mapping carries that
    # geometry statically — a metric cached across solves would hand
    # the second solve the first geometry
    def moved(x):
        return 1.0 + 0.3 * jnp.cos(x)

    solutions = {}
    for name, init in (("static", depth), ("moved", moved)):
        grid, mx, ms = build_grid()
        space = mx.center * ms.center
        rhs = grid.random.normal(space, seed=17)
        rhs = rhs - rhs.mean()
        h = grid.create_field(mx.center, init=init)
        dynamic = MappedPressureSolver(
            grid, space, iterations=20,
            weights={"sigma": 1.0 / DSQR}, params={"H": h})
        static, sgrid, smx, sms = build_solver(init=init)
        static_rhs = sgrid.random.normal(
            smx.center * sms.center, seed=17)
        static_rhs = static_rhs - static_rhs.mean()
        p_dyn = np.asarray(dynamic.solve(rhs).data)
        p_stat = np.asarray(static.solve(static_rhs).data)
        np.testing.assert_allclose(p_dyn, p_stat, rtol=1e-12,
                                   atol=1e-14)
        solutions[name] = p_dyn
    assert not np.allclose(solutions["static"], solutions["moved"])


# ================================================================
#  Taught construction errors
# ================================================================
def test_unmapped_grid_is_rejected():
    mx = IntervalMesh(N, (0.0, 2 * np.pi), name="x")
    ms = IntervalMesh(N, (0.0, 1.0), periodic=False, name="sigma")
    grid = Grid((mx, ms))
    with pytest.raises(ValueError, match="no coordinate mapping"):
        MappedPressureSolver(grid, mx.center * ms.center,
                             iterations=4)


def test_chart_only_mapping_is_rejected():
    mu = IntervalMesh(N, (0.0, 2 * np.pi), name="u")
    mv = IntervalMesh(N, (0.0, 2 * np.pi), name="v")
    mapping = CoordinateMapping(
        chart={"X": lambda u, v: (jnp.cos(u), jnp.sin(u), v)})
    grid = Grid((mu, mv), mapping=mapping)
    with pytest.raises(ValueError, match="no mapped column"):
        MappedPressureSolver(grid, mu.center * mv.center,
                             iterations=4)


def test_two_mapped_columns_are_rejected():
    mx = IntervalMesh(N, (0.0, 2 * np.pi), name="x")
    ms = IntervalMesh(N, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(maps={
        "zp": lambda sigma: 2.0 * sigma,
        "xp": lambda x: 3.0 * x,
    })
    grid = Grid((mx, ms), mapping=mapping)
    with pytest.raises(NotImplementedError,
                       match="exactly one mapped column"):
        MappedPressureSolver(grid, mx.center * ms.center,
                             iterations=4)


def test_bounded_coupled_axis_is_rejected():
    # the symmetric corner form's transpose pairing is derived for
    # periodic coupled axes (module docstring)
    grid, mx, ms = build_grid(periodic_x=False)
    with pytest.raises(NotImplementedError, match="bounded mesh"):
        MappedPressureSolver(grid, mx.center * ms.center,
                             iterations=4)


def test_space_must_resolve_the_base_coordinate():
    grid, mx, ms = build_grid()
    with pytest.raises(ValueError, match="base coordinate"):
        MappedPressureSolver(grid, mx.center * ms.constant,
                             iterations=4)


def test_unknown_weight_axes_are_rejected():
    grid, mx, ms = build_grid()
    with pytest.raises(ValueError, match="unknown weight axes"):
        MappedPressureSolver(grid, mx.center * ms.center,
                             iterations=4, weights={"zeta": 2.0})


# ================================================================
#  The DynamicalCore mapped projection branch
# ================================================================
def make_mapped_model(n=8, init=depth, dt=0.02, family="nodal", **kwargs):
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": init})
    grid = Grid((mx, my, mz), mapping=mapping)
    # family="nodal" is explicit: since the 2026-07-17 mapped auto flip
    # a plain nh.Model on a mapped grid resolves to FV, but this file is
    # the NODAL mapped battery (the FV mapped battery is
    # test_mapped_pressure_fv.py) — pin nodal so the split stays clean.
    # 16 PCG iterations, not the model default 30: the projection has
    # converged there (measured post-step mapped divergence 3.4e-12
    # already at 12, 1.1e-13 at 30; the gate below is 1e-10) and the
    # unrolled CG loop is what the mapped model's trace pays for
    return nh.Model(grid=grid, dt=dt, advection=False, family=family,
                    coriolis=nh.FPlaneCoriolis(f0=1.0),
                    pressure_iterations=16, **kwargs)


def test_core_projects_on_a_mapped_grid():
    model = make_mapped_model(dsqr=DSQR)
    hor = (np.arange(8) + 0.5) * (2 * np.pi / 8)
    ver = (np.arange(8) + 0.5) / 8
    x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
    model.set_fields(u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x),
                     b=0.01 * np.cos(np.pi * z))
    model.advance(1)
    assert not model.panicked
    solver = MappedPressureSolver(
        model.state["u"].grid, model.state["p"].function_space,
        iterations=1)
    div = solver.divergence({
        "x": model.state["u"], "y": model.state["v"],
        "z": model.state["w"]})
    # the state projected by the CONSTRAINT stage is divergence-free
    # in the mapped sense, to the stage's CG residual
    assert float(jnp.abs(div.data).max()) < 1e-10


def test_mapped_model_is_treedef_stable():
    model = make_mapped_model(dsqr=DSQR)
    ver = (np.arange(8) + 0.5) / 8
    _, _, z = np.meshgrid(ver, ver, ver, indexing="ij")
    model.set_fields(b=0.01 * np.cos(np.pi * z))
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(3)
    assert jax.tree_util.tree_structure(model._carry) == before
    assert not model.panicked


# ================================================================
#  Smoothing surfaces (B1): diagonal() and vertical_bands()
# ================================================================
def steep_depth(x):
    """Return a steep periodic water depth (depth ratio 9.0)."""
    return 1.0 + 0.8 * jnp.sin(x)


def _probe_diag_and_zbands_2d(solver, grid, space, z_axis, period=4):
    """Extract (diag, lower, upper) along ``z_axis`` by p-coloring.

    Period 4 (not 3): the periodic x-count (8) is not divisible by 3, so
    a 3-coloring aliases distance-1 wrap neighbours; period 4 keeps the
    three consecutive residues of any +-1 stencil distinct on the torus.
    """
    shape = tuple(space.shape)
    a_data = jax.jit(
        lambda d: solver.apply(grid.create_field(space, data=d)).data)
    p = period
    ii, jj = np.indices(shape)
    stack = np.zeros((p, p, *shape))
    for cx in range(p):
        for cy in range(p):
            mask = ((ii % p == cx) & (jj % p == cy)).astype(np.float64)
            stack[cx, cy] = np.asarray(a_data(jnp.asarray(mask)))
    ci, cj = ii % p, jj % p
    assert z_axis == 1  # (x, sigma): the column axis is the second
    diag = stack[ci, cj, ii, jj]
    lower = stack[ci, (jj - 1) % p, ii, jj]
    upper = stack[ci, (jj + 1) % p, ii, jj]
    return diag, lower, upper


def test_diagonal_is_probe_exact_on_a_steep_case():
    # the analytic diagonal (diagonal-flux legs + the corner cross rows)
    # matches the probe-extracted diagonal to roundoff, every cell
    # (boundary-adjacent columns included)
    solver, grid, mx, ms = build_solver(n=8, init=steep_depth)
    space = mx.center * ms.center
    diag_probe, _, _ = _probe_diag_and_zbands_2d(solver, grid, space, 1)
    diag = np.asarray(solver.diagonal().data)
    scale = np.abs(diag_probe).max()
    assert np.abs(diag_probe - diag).max() <= 1e-11 * scale


def test_vertical_bands_diag_is_the_full_diagonal():
    solver, *_ = build_solver(n=8, init=steep_depth)
    bands = solver.vertical_bands()
    assert np.array_equal(np.asarray(bands.diag.data),
                          np.asarray(solver.diagonal().data))


def test_vertical_bands_are_symmetric_with_neumann_ends():
    solver, *_ = build_solver(n=8, init=steep_depth)
    bands = solver.vertical_bands()
    assert bands.axis == 1
    lo = np.asarray(bands.lower.data)
    up = np.asarray(bands.upper.data)
    # symmetric per column: lower[c] == upper[c-1]
    assert np.abs(lo[:, 1:] - up[:, :-1]).max() < 1e-12
    # Neumann ends: no coupling through the walls
    assert np.abs(lo[:, 0]).max() == 0.0
    assert np.abs(up[:, -1]).max() == 0.0


def test_vertical_band_offdiagonals_match_the_probed_z_coupling():
    # the vertical-leg band +K^bb/dz^2 IS the exact z-off-diagonal of A:
    # the slope cross residues couple off-column (x +- 1), never the
    # pure vertical neighbour, so the tridiagonal misses nothing there
    solver, grid, mx, ms = build_solver(n=8, init=steep_depth)
    space = mx.center * ms.center
    _, lo_p, up_p = _probe_diag_and_zbands_2d(solver, grid, space, 1)
    bands = solver.vertical_bands()
    lo = np.asarray(bands.lower.data)
    up = np.asarray(bands.upper.data)
    scale = np.abs(np.asarray(bands.diag.data)).max()
    assert np.abs(up[:, :-1] - up_p[:, :-1]).max() <= 1e-11 * scale
    assert np.abs(lo[:, 1:] - lo_p[:, 1:]).max() <= 1e-11 * scale


def test_vertical_line_sweep_reduces_the_steep_residual():
    solver, grid, mx, ms = build_solver(n=8, init=steep_depth)
    space = mx.center * ms.center
    smoother = VerticalLineJacobi(solver.vertical_bands(), omega=0.8)
    b = grid.random.normal(space, seed=7)
    b = b - b.mean()
    x0 = grid.create_field(space)
    x1 = smoother.sweep(x0, b, solver.apply)
    r0 = b - solver.apply(x0)
    r1 = b - solver.apply(x1)
    assert dot(r1, r1) < dot(r0, r0)


def test_grad_through_a_line_sweep_is_finite():
    # the smoother is step-path once selected: jax.grad through one
    # vertical-line sweep (guarded Thomas + the mapped operator) is
    # finite and non-zero (differentiability policy)
    solver, grid, mx, ms = build_solver(n=8, init=steep_depth)
    space = mx.center * ms.center
    smoother = VerticalLineJacobi(solver.vertical_bands(), omega=0.8)
    b = grid.random.normal(space, seed=8)

    def loss(scale):
        x1 = smoother.sweep(
            grid.create_field(space), scale * b, solver.apply)
        return jnp.sum(x1.data ** 2)

    grad = jax.grad(loss)(2.0)
    assert bool(jnp.isfinite(grad))
    assert float(grad) != 0.0
