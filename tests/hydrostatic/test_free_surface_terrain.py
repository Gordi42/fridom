r"""The free surface on a terrain-following (sigma) grid.

Depth fix (research record ``stretched_terrain_combined.md`` §6): the
physical column depth ``H(x, y) = \int J\,dz`` (not the computational
extent) sets the depth-mean divisor and the barotropic energy weight,
and the depth-mean divergence is the flux-form transport divergence
``\int[\partial_x(Ju) + \partial_y(Jv)]\,dz / H`` — the exact adjoint
(under the physical-volume inner product) of the ``-\nabla_h ps``
momentum force, so the barotropic gravity pair conserves energy to
roundoff. The variable-coefficient **implicit** free surface (H3) and
the transport-depth-consistent **split** subcycle stay deferred behind
taught errors. Self-contained builders (AGENTS oversized-module rule).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.operators.integrate import Integral

IM = fr.spatial.meshes.IntervalMesh
N2, CSQR = 2.0, 1.0


def _depth(x, y):
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _terrain_grid(n):
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(n, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": _depth}))


def _model(grid, *, free_surface=None, coriolis=None, dt=2e-3,
           stepper=None):
    return hy.Model(
        grid=grid, dt=dt, csqr=CSQR,
        stratification=hy.ConstantStratification(n2=N2),
        coriolis=coriolis, advection=False,
        free_surface=free_surface or hy.ExplicitFreeSurface(),
        time_stepper=stepper
        or fr.model.time_steppers.AdamBashforth(dt, order=3))


def _bound_fs(model):
    """Return the assembled model's (already bound) free surface."""
    return model.module(hy.ExplicitFreeSurface)


# ================================================================
#  Physical column depth H(x, y) = int J dz (not the extent)
# ================================================================
def test_physical_depth_matches_the_analytic_depth():
    grid = _terrain_grid(16)
    fs = _bound_fs(_model(grid))
    coll = fr.spatial.Collocated().resolve(grid)
    one = grid.create_field(coll, data=jnp.ones(coll.shape))
    depth_field = fs._physical_depth(one)
    xs = grid.evaluation_nodes(depth_field.function_space, "x").data
    ys = grid.evaluation_nodes(depth_field.function_space, "y").data
    # the column extent is 1, so H(x, y) = int_{-1}^0 H dz = H(x, y)
    assert float(jnp.abs(depth_field.data - _depth(xs, ys)).max()) < 1e-13
    # ... and it is genuinely different from the computational extent 1
    assert float(jnp.abs(depth_field.data - 1.0).max()) > 0.1


def test_physical_depth_equals_the_jacobian_integral_seam():
    # the free surface's in-trace depth agrees with the wired
    # Integral(jacobian=) seam (the canonical physical column extent).
    # Build the model first: it re-negotiates the grid to the
    # hydrostatic core's extra_halo, so ``one`` must be created on the
    # frozen (final-width) grid, not the provisionally-narrower base.
    grid = _terrain_grid(12)
    fs = _bound_fs(_model(grid))
    coll = fr.spatial.Collocated().resolve(grid)
    one = grid.create_field(coll, data=jnp.ones(coll.shape))
    seam = Integral(jacobian=("zp",))["z"](one)
    mine = fs._physical_depth(one)
    assert np.allclose(np.asarray(mine.data), np.asarray(seam.data),
                       atol=1e-13)


# ================================================================
#  H4 (barotropic leg): the surface-pressure <-> depth-mean pair
#  conserves energy to roundoff under the physical-volume metric
# ================================================================
def test_barotropic_energy_is_conserved_to_roundoff():
    # perturb (u, v, ps), no buoyancy, no rotation: the isolated
    # barotropic gravity pair. Under the physical-volume (J-weighted)
    # inner product with the depth-integrated ps weight H/c^2, the skew
    # <X, M dX/dt> vanishes to machine precision -- the flux-form
    # depth-mean divergence is the exact adjoint of -grad ps.
    grid = _terrain_grid(16)
    model = _model(grid)
    rng = np.random.default_rng(3)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=np.zeros(model.state["b"].shape),
        ps=rng.standard_normal(model.state["ps"].shape))
    st = model.state
    dX = model.tendency(st)
    p3 = st["b"].function_space

    def jint(f):
        # the physical (Jacobian-weighted) volume integral is now the
        # plain seeded verb on a maps= terrain grid (the physical-
        # integral-default flip): f.integrate() carries the column
        # Jacobian, so hand-multiplying dzp_dz would double-count
        return float(f.integrate().data.ravel()[0])
    terms = [jint(st["u"] * dX["u"]), jint(st["v"] * dX["v"]),
             jint((st["ps"].to(p3) / CSQR) * dX["ps"].to(p3))]
    skew = sum(terms)
    scale = sum(abs(t) for t in terms)
    assert abs(skew) < 1e-12 * scale


# ================================================================
#  The explicit free surface runs finite and stays bounded
# ================================================================
def test_explicit_free_surface_runs_finite_and_bounded():
    grid = _terrain_grid(16)
    model = _model(grid, coriolis=hy.FPlaneCoriolis(f0=1.0))
    rng = np.random.default_rng(1)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        b=0.1 * rng.standard_normal(model.state["b"].shape),
        ps=0.1 * rng.standard_normal(model.state["ps"].shape))
    ps0 = float(jnp.abs(model.state["ps"].data).max())
    model.run(40, progress=False)
    assert bool(jnp.isfinite(model.state["ps"].data).all())
    # bounded (a barotropic gravity wave oscillates, does not blow up)
    assert float(jnp.abs(model.state["ps"].data).max()) < 10.0 * ps0 + 1.0


def test_flat_depth_mean_is_byte_identical():
    # off a terrain grid the depth mean keeps the flat scalar 1/H path
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=True, name="x"),
        IM(8, (0.0, 1.0), periodic=True, name="y"),
        IM(6, (-1.0, 0.0), periodic=False, name="z")))
    fs = _bound_fs(_model(grid))
    assert fs._column is None
    assert fs._inv_depth == pytest.approx(1.0)


# ================================================================
#  The implicit free surface now engages on a terrain grid (H3; the
#  volume-exact solve, GM-D1/D2) — the taught error is gone. The
#  gates live in test_free_surface_terrain_implicit.py.
# ================================================================
def test_implicit_free_surface_engages_on_terrain():
    grid = _terrain_grid(8)
    model = _model(grid, free_surface=hy.ImplicitFreeSurface())
    assert "ps" in model.state.component_names
    fs = model.module(hy.ImplicitFreeSurface)
    assert fs._column == ("zp", "z")
    rng = np.random.default_rng(0)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        ps=0.1 * rng.standard_normal(model.state["ps"].shape))
    model.advance(4)
    assert not model.panicked
    assert bool(jnp.isfinite(model.state["ps"].data).all())


def test_split_free_surface_is_a_taught_error_on_terrain():
    grid = _terrain_grid(8)
    with pytest.raises(NotImplementedError, match="terrain"):
        _model(grid,
                free_surface=hy.SplitExplicitFreeSurface(substeps=4),
                stepper=fr.model.time_steppers.AdamBashforth(2e-3, order=2))
