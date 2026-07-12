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
def make_mapped_model(n=8, init=depth, dt=0.02, **kwargs):
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": init})
    grid = Grid((mx, my, mz), mapping=mapping)
    return nh.Model(grid=grid, dt=dt, advection=False, **kwargs)


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
