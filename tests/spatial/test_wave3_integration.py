"""
Cross-track integration of the Wave-3 clusters (ROADMAP 1.3-1.5).

One suite exercising the seams between the FV/average operators, the
transforms + spectral operators, and the multi-device decomposition:
spectral Poisson-style flows, FV advection-like mixes, dealiased
products over adopted refined meshes, device-count invariance of the
mixed pipelines (bitwise under the forced-4 suite), and the halo
trace over a tendency mixing FD, FV, and a transform.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.decomposition.halo import trace_halo
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.fourier import Fourier


def bitwise(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def build_grid(device_ids):
    mx = IntervalMesh(16, (0.0, 1.0), name="x")  # periodic
    my = IntervalMesh(16, (0.0, 2.0), name="y")  # periodic
    return Grid((mx, my), device_ids=device_ids)


@pytest.fixture
def grids(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    return build_grid(None), build_grid((0,))


# ================================================================
#  (a) Spectral Poisson-style flow (transform + spectral diff)
# ================================================================
def test_spectral_second_derivative_is_exact():
    mx = IntervalMesh(32, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    f = grid.create_field(
        init=lambda x: jnp.sin(2 * jnp.pi * x)
        + 0.5 * jnp.cos(8 * jnp.pi * x))
    t = grid.dispatch.resolve("transform", mx.center)
    d2_hat = t.forward(f).diff("x").diff("x")
    back = t.backward(d2_hat)
    assert back.function_space.bare is mx.center
    x = grid.evaluation_nodes(mx.center).data
    exact = (-(2 * jnp.pi) ** 2 * jnp.sin(2 * jnp.pi * x)
             - 0.5 * (8 * jnp.pi) ** 2 * jnp.cos(8 * jnp.pi * x))
    assert jnp.abs(back.data - exact).max() < 1e-9


def test_spectral_poisson_solve_via_wavenumbers():
    # solve u'' = rhs by dividing the spectrum by -k^2 (mean-free)
    mx = IntervalMesh(32, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    u_exact = grid.create_field(
        init=lambda x: jnp.sin(4 * jnp.pi * x))
    rhs = grid.create_field(
        init=lambda x: -(4 * jnp.pi) ** 2 * jnp.sin(4 * jnp.pi * x))
    t = grid.dispatch.resolve("transform", mx.center)
    rhs_hat = t.forward(rhs)
    k = grid.wavenumbers(rhs_hat.function_space)
    inv = jnp.where(k.data == 0, 0.0, -1.0 / jnp.where(
        k.data == 0, 1.0, k.data**2))
    u_hat = rhs_hat.with_data(rhs_hat.data * inv)
    u = t.backward(u_hat)
    assert jnp.abs(u.data - u_exact.data).max() < 1e-10


def test_rfftn_spectrum_round_trips_through_create_field():
    # regression (hermitian_project guard): a valid 2-D rfftn
    # spectrum entering create_field(data=...) must be preserved
    # exactly (conjugate pairing, not flat imag-zeroing) and round
    # trip through backward to the original real samples
    grid = build_grid((0,))
    f = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.cos(
            jnp.pi * y) + x * 0 + y * 0)
    t = grid.dispatch.resolve(
        "transform", f.function_space.bare)
    f_hat = t.forward(f)
    coeff_space = f_hat.function_space
    rebuilt = grid.create_field(coeff_space, data=f_hat.data)
    assert bitwise(rebuilt.data, f_hat.data)  # NOT modified
    back = t.backward(rebuilt)
    assert jnp.abs(back.data - f.data).max() < 1e-13


# ================================================================
#  (b) FV advection-like mix (diff on averages, reconstruct,
#      integrate conservation)
# ================================================================
def test_fv_advection_mix_conserves():
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    q = grid.create_field(
        mx.cell_avg, init=lambda x: 1.0 + 0.5 * jnp.sin(
            2 * jnp.pi * x))
    u = grid.create_field(
        mx.right, init=lambda x: jnp.cos(2 * jnp.pi * x))
    # FV derivative on averages: flux_diff @ reconstruct chain
    dq = q.diff("x")
    assert dq.function_space.bare is mx.cell_avg
    assert jnp.abs(dq.integrate("x").data).max() < 1e-12
    # advection-like flux: reconstruct q to the faces, multiply by
    # the face velocity, and take the exact flux difference
    flux = u * q.to(mx.right)
    div = grid.dispatch.resolve("flux_diff", mx.right)["x"](flux)
    assert div.function_space.bare is mx.cell_avg
    # discrete Gauss: the cell-measure integral telescopes to zero
    assert jnp.abs(div.integrate("x").data).max() < 1e-12


def test_dealiased_product_over_the_adopted_refined_mesh():
    # the ("multiply", refined center) row resolves by refined-mesh
    # adoption: the parent mesh's CollocationProduct instance fires
    # on the padded transform's finer nodal space
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    plain = grid.dispatch.resolve("transform", mx.center)
    padded = Fourier(grid, pad=degree(2))
    f = grid.create_field(init=lambda x: jnp.sin(2 * jnp.pi * x))
    g = grid.create_field(init=lambda x: jnp.cos(2 * jnp.pi * x))
    fine_f = padded.backward(plain.forward(f))
    fine_g = padded.backward(plain.forward(g))
    fine_mesh = fine_f.function_space.bare.mesh
    assert fine_mesh.n_cells == 24
    assert fine_mesh.refined_from is mx
    product_hat = padded.forward(fine_f * fine_g)
    # sin(2 pi x) cos(2 pi x) = 0.5 sin(4 pi x), resolved exactly
    exact = grid.create_field(
        init=lambda x: 0.5 * jnp.sin(4 * jnp.pi * x))
    assert jnp.abs(product_hat.data
                   - plain.forward(exact).data).max() < 1e-14


# ================================================================
#  (c) Device-count invariance of the mixed pipelines
# ================================================================
def test_transform_pipeline_is_device_count_invariant(grids):
    many, one = grids

    def compute(grid):
        f = grid.create_field(
            init=lambda x, y: jnp.sin(2 * jnp.pi * x)
            * jnp.cos(jnp.pi * y) + x * y)
        t = grid.dispatch.resolve(
            "transform", f.function_space.bare)
        f_hat = t.forward(f)
        d_hat = f_hat.diff("x")
        back = t.backward(d_hat)
        return f_hat, d_hat, back

    for a, b in zip(compute(many), compute(one), strict=True):
        assert bitwise(a.data, b.data)


def test_fv_pipeline_is_device_count_invariant(grids):
    many, one = grids

    def compute(grid):
        mx = grid.factors[0]
        q = grid.create_field(
            mx.cell_avg * grid.factors[1].center,
            init=lambda x, y: jnp.sin(2 * jnp.pi * x) + y)
        dq = q.diff("x")  # FV chain (halo 2, one application)
        faces = q.to(q.function_space.bare.replace(x=mx.right))
        return dq, faces, dq.integrate("x")

    for a, b in zip(compute(many), compute(one), strict=True):
        assert bitwise(a.data, b.data)


def test_reshard_transform_backward_round_trip(grids):
    # distributed-transform path: move to the pencil keeping x
    # local, transform along x, come back — bitwise equal to the
    # one-device run
    many, one = grids
    decomp = many.decomposition
    pencil = decomp.layout_for(("x",))

    def compute(grid, layout):
        f = grid.create_field(
            init=lambda x, y: jnp.sin(2 * jnp.pi * x)
            * jnp.cos(jnp.pi * y))
        moved = f.reshard(layout) if layout is not None else f
        t = Fourier(grid, axes=("x",))
        f_hat = t.forward(moved)
        return t.backward(f_hat)

    back_many = compute(many, pencil)
    assert back_many.function_space.layout == pencil
    back_one = compute(one, None)
    assert bitwise(back_many.data, back_one.data)


# ================================================================
#  (d) Halo trace over a tendency mixing FD, FV, and a transform
# ================================================================
def test_trace_halo_over_a_mixed_tendency():
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    my = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")
    grid = Grid((mx, my))
    t = Fourier(grid, axes=("x",))
    # u's y-factor is Outer so the bounded FD chain runs through
    # the exterior-free Outer -> Center -> Inner signatures
    # (BC-free Inner -> Center is gated by R1, boundary_plan.md)
    spaces = (mx.center * my.outer, mx.cell_avg * my.center)

    def tendency(state):
        u, q = state[0], state[1]
        q.diff("x")             # FV chain: composed window [-1,+1] = 1
        u.diff("y").diff("y")   # bounded Outer->Center->Inner: reach 0
        u_hat = t(u)            # transform: halo 0
        u_hat.diff("x")         # spectral derivative: halo 0

    spec = trace_halo(tendency, spaces, grid.dispatch)
    # two-sided accounting: the periodic FV derivative composes to
    # width 1 (not the scalar sum 2); the bounded double difference
    # shrinks the codomain each hop and reads no exterior slot (0)
    assert spec["x"] == 1
    assert spec["y"] == 0
