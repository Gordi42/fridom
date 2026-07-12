"""
Stretched-mesh numerics (coordinate-systems plan, stage C0).

The measure-field mechanism end-to-end on ``MappedIntervalMesh``:
staggered finite differences converge at the operator's order to
the analytic derivative of a smooth function (periodic and bounded
stretchings), ``integrate`` is exact where uniform integration is
exact by construction (cell averages against the primal measure)
and second-order convergent on nodal spaces, and the flux-form FV
derivative telescopes to the boundary fluxes exactly (the discrete
Gauss theorem survives the stretching).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)

RESOLUTIONS = (16, 32, 64)

#: minimum observed convergence order for a 2nd-order scheme
ORDER_FLOOR = 1.7


def tanh_map(s):
    """Boundary-refined stretching on [0, 1] (bounded runs)."""
    return jnp.tanh(2.0 * s) / jnp.tanh(2.0)


def wavy_map(s):
    """Smooth periodic stretching on [0, 1] (periodic runs)."""
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


def observed_orders(errors):
    """Pairwise log2 error ratios of a dyadic refinement chain."""
    errors = np.asarray(errors)
    return np.log2(errors[:-1] / errors[1:])


# ================================================================
#  Finite differences converge at the operator's order
# ================================================================
def test_bounded_fd_converges_at_second_order():
    errors = []
    for n in RESOLUTIONS:
        mesh = MappedIntervalMesh(n, (0.0, 1.0), tanh_map,
                                  name="z")
        grid = Grid((mesh,))
        f = grid.create_field(mesh.center,
                              init=lambda z: jnp.sin(jnp.pi * z))
        df = f.diff("z")
        x = grid.evaluation_nodes(mesh.inner).data
        errors.append(float(jnp.abs(
            df.data - jnp.pi * jnp.cos(jnp.pi * x)).max()))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


def test_periodic_fd_converges_at_second_order():
    errors = []
    for n in RESOLUTIONS:
        mesh = MappedIntervalMesh(n, (0.0, 1.0), wavy_map,
                                  periodic=True, name="p")
        grid = Grid((mesh,))
        f = grid.create_field(
            mesh.center,
            init=lambda p: jnp.sin(2.0 * jnp.pi * p))
        df = f.diff("p")
        x = grid.evaluation_nodes(mesh.right).data
        errors.append(float(jnp.abs(
            df.data
            - 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x)).max()))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


def test_fv_derivative_converges_at_second_order():
    # the full flux_diff @ reconstruct chain on average spaces
    errors = []
    for n in RESOLUTIONS:
        mesh = MappedIntervalMesh(n, (0.0, 1.0), wavy_map,
                                  periodic=True, name="p")
        grid = Grid((mesh,))
        f = grid.create_field(
            mesh.cell_avg,
            init=lambda p: jnp.sin(2.0 * jnp.pi * p))
        df = f.diff("p")
        x = grid.evaluation_nodes(mesh.cell_avg).data
        errors.append(float(jnp.abs(
            df.data
            - 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x)).max()))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


# ================================================================
#  Integration against the measure fields
# ================================================================
def test_integrate_is_exact_on_stretched_cell_averages():
    # exact-by-construction: true cell averages contracted against
    # the primal cell widths telescope to the antiderivative
    for n in RESOLUTIONS:
        mesh = MappedIntervalMesh(n, (0.0, 1.0), tanh_map,
                                  name="z")
        grid = Grid((mesh,))
        faces = tanh_map(jnp.arange(n + 1) / n)
        w = grid.measure(mesh.cell_avg, name="z")
        # cell averages of f(z) = z^2 (antiderivative z^3 / 3)
        averages = jnp.diff(faces**3 / 3.0) / w.data
        f = grid.create_field(mesh.cell_avg, data=averages)
        assert jnp.allclose(f.integrate("z").data[0], 1.0 / 3.0,
                            atol=1e-14)


def test_nodal_integrate_converges_at_second_order():
    errors = []
    for n in RESOLUTIONS:
        mesh = MappedIntervalMesh(n, (0.0, 1.0), tanh_map,
                                  name="z")
        grid = Grid((mesh,))
        f = grid.create_field(mesh.center,
                              init=lambda z: jnp.sin(jnp.pi * z))
        errors.append(abs(float(f.integrate("z").data[0])
                          - 2.0 / np.pi))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


# ================================================================
#  Conservation: the discrete Gauss theorem on stretched cells
# ================================================================
def test_flux_diff_telescopes_to_the_boundary_fluxes():
    mesh = MappedIntervalMesh(32, (0.0, 1.0), tanh_map, name="z")
    grid = Grid((mesh,))
    flux = grid.random.normal(mesh.outer, seed=42)
    div = grid.dispatch.resolve("flux_diff", mesh.outer)["z"](flux)
    total = div.integrate("z").data[0]
    boundary = flux.data[-1] - flux.data[0]
    assert total == pytest.approx(float(boundary), rel=1e-12)


def test_periodic_flux_diff_sums_to_zero():
    mesh = MappedIntervalMesh(32, (0.0, 1.0), wavy_map,
                              periodic=True, name="p")
    grid = Grid((mesh,))
    flux = grid.random.normal(mesh.right, seed=7)
    div = grid.dispatch.resolve("flux_diff", mesh.right)["p"](flux)
    assert jnp.allclose(div.integrate("p").data[0], 0.0,
                        atol=1e-12)
