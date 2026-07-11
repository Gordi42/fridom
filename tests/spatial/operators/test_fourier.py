"""Fourier-transform tests: shapes, round trips, invariants."""
import jax.numpy as jnp
import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.scalars import Scalars

TWO_PI = 2.0 * jnp.pi


def _grid1d(n=8):
    mesh = IntervalMesh(n, (0.0, 1.0), name="x")
    return Grid((mesh,)), mesh


def _grid2d(nx=8, ny=6):
    mx = IntervalMesh(nx, (0.0, 1.0), name="x")
    my = IntervalMesh(ny, (0.0, 2.0), name="y")
    return Grid((mx, my)), mx, my


# ================================================================
#  Shape tables (rfft layout as shape, section 3.2)
# ================================================================
def test_real_origin_gives_the_half_spectrum():
    grid, mesh = _grid1d(8)
    coeff = Fourier(grid).forward(grid.create_field())
    space = coeff.function_space.bare
    assert space is mesh.fourier(origin=mesh.center)
    assert space.shape == (5,)
    assert space.scalars is Scalars.REAL
    assert jnp.issubdtype(coeff.dtype, jnp.complexfloating)


def test_complex_origin_gives_the_full_spectrum():
    grid, mesh = _grid1d(8)
    space = mesh.center.as_complex()
    f = grid.create_field(space,
                          data=jnp.arange(8.0) + 1j * jnp.ones(8))
    coeff = Fourier(grid).forward(f)
    bare = coeff.function_space.bare
    assert bare is mesh.fourier(origin=space)
    assert bare.shape == (8,)
    assert bare.scalars is Scalars.COMPLEX


def test_multi_axis_real_shape_table():
    # MUST-test: the half spectrum lands on the FIRST-transformed
    # axis only; later factors target complexified origins.
    grid, mx, my = _grid2d(8, 6)
    coeff = Fourier(grid).forward(grid.create_field())
    space = coeff.function_space.bare
    assert space.shape == (5, 6)
    fx, fy = space.factors
    assert fx is mx.fourier(origin=mx.center)
    assert fx.scalars is Scalars.REAL
    assert fy is my.fourier(origin=my.center.as_complex())
    assert fy.scalars is Scalars.COMPLEX


def test_single_axis_transform_puts_the_half_spectrum_there():
    grid, mx, my = _grid2d(8, 6)
    coeff = Fourier(grid, axes="y").forward(grid.create_field())
    space = coeff.function_space.bare
    assert space.shape == (8, 3 + 1)
    assert space.factors[0] is mx.center
    assert space.factors[1] is my.fourier(origin=my.center)


def test_three_axis_real_shape_table():
    mx = IntervalMesh(4, (0.0, 1.0), name="x")
    my = IntervalMesh(4, (0.0, 1.0), name="y")
    mz = IntervalMesh(4, (0.0, 1.0), name="z")
    grid = Grid((mx, my, mz))
    coeff = Fourier(grid).forward(grid.create_field())
    space = coeff.function_space.bare
    assert space.shape == (3, 4, 4)
    halves = [f.scalars is Scalars.REAL for f in space.factors]
    assert halves == [True, False, False]


def test_complex_storage_domains_get_full_spectra_everywhere():
    # a complex factor anywhere makes every stage a full fft
    grid, mx, my = _grid2d(8, 6)
    space = mx.center * my.center.as_complex()
    f = grid.create_field(space, data=jnp.ones((8, 6),
                                               dtype=complex))
    coeff = Fourier(grid).forward(f)
    fx, fy = coeff.function_space.bare.factors
    assert fx is mx.fourier(origin=mx.center.as_complex())
    assert fy is my.fourier(origin=my.center.as_complex())
    assert coeff.shape == (8, 6)


# ================================================================
#  Round trips (forward/backward = id to fp tolerance)
# ================================================================
@pytest.mark.parametrize("n", [
    pytest.param(8, id="even"),
    pytest.param(9, id="odd"),
])
def test_1d_round_trip_is_exact(n):
    grid, mesh = _grid1d(n)
    f = grid.random.normal(mesh.center, seed=1)
    op = Fourier(grid)
    back = op.backward(op.forward(f))
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-14)


def test_2d_round_trip_and_real_landing():
    grid, _, _ = _grid2d(8, 6)
    f = grid.create_field(
        init=lambda x, y: (jnp.sin(TWO_PI * x) * jnp.cos(TWO_PI * y)
                           + jnp.cos(TWO_PI * y / 2.0)))
    op = Fourier(grid)
    coeff = op.forward(f)
    back = op.backward(coeff)
    # backward of a half-spectrum-bearing space lands on the REAL
    # origins (the represented field is real)
    assert back.function_space is f.function_space
    assert jnp.issubdtype(back.dtype, jnp.floating)
    assert jnp.allclose(back.data, f.data, atol=1e-14)


def test_coefficient_side_round_trip_is_exact():
    grid, mesh = _grid1d(8)
    op = Fourier(grid)
    coeff = op.forward(grid.random.normal(mesh.center, seed=2))
    again = op.forward(op.backward(coeff))
    assert again.function_space is coeff.function_space
    assert jnp.allclose(again.data, coeff.data, atol=1e-14)


def test_complex_origin_round_trip():
    grid, mesh = _grid1d(8)
    space = mesh.center.as_complex()
    data = jnp.exp(1j * jnp.linspace(0.0, 3.0, 8))
    f = grid.create_field(space, data=data)
    op = Fourier(grid)
    back = op.backward(op.forward(f))
    assert back.function_space.bare is space
    assert jnp.allclose(back.data, data, atol=1e-14)


def test_known_mode_coefficient_values():
    grid, _ = _grid1d(8)
    f = grid.create_field(init=lambda x: jnp.cos(TWO_PI * x))
    coeff = Fourier(grid).forward(f)
    # amplitude convention: cos = (e^{ikx} + c.c.)/2; the stored
    # index-based coefficients carry the center-origin phase
    # e^{+i k x0} with x0 = dx/2
    expected = 0.5 * jnp.exp(1j * TWO_PI * 0.5 / 8)
    assert jnp.abs(coeff.data[1] - expected) < 1e-14
    others = coeff.data.at[1].set(0)
    assert jnp.max(jnp.abs(others)) < 1e-14


# ================================================================
#  Hermitian value invariant (self-conjugate modes)
# ================================================================
def test_self_conjugate_modes_are_exactly_real():
    grid, mesh = _grid1d(8)
    f = grid.random.normal(mesh.center, seed=3)
    coeff = Fourier(grid).forward(f)
    assert coeff.data[0].imag == 0.0
    assert coeff.data[-1].imag == 0.0


def test_2d_hermitian_pairing_on_the_half_spectrum_planes():
    grid, _, _ = _grid2d(8, 6)
    f = grid.random.normal(
        grid.create_field().function_space, seed=4)
    coeff = Fourier(grid).forward(f)
    for kx in (0, 4):  # kx = 0 and Nyquist planes
        row = coeff.data[kx]
        paired = jnp.conj(jnp.roll(row[::-1], 1))
        assert jnp.allclose(row, paired, atol=1e-14)


def test_round_trip_preserves_the_invariant():
    grid, mesh = _grid1d(8)
    op = Fourier(grid)
    coeff = op.forward(grid.random.normal(mesh.center, seed=5))
    again = op.forward(op.backward(coeff))
    assert again.data[0].imag == 0.0
    assert again.data[-1].imag == 0.0


# ================================================================
#  Parseval (amplitude convention)
# ================================================================
def test_parseval_1d():
    grid, mesh = _grid1d(8)
    f = grid.random.normal(mesh.center, seed=6)
    coeff = Fourier(grid).forward(f)
    physical = jnp.mean(f.data ** 2)
    weights = jnp.full(5, 2.0).at[0].set(1.0).at[-1].set(1.0)
    spectral = jnp.sum(weights * jnp.abs(coeff.data) ** 2)
    assert jnp.abs(physical - spectral) < 1e-13


# ================================================================
#  Average origins are ordinary origins (G2)
# ================================================================
def test_average_origin_forward_and_round_trip():
    grid, mesh = _grid1d(8)
    f = grid.create_field(mesh.cell_avg,
                          init=lambda x: jnp.sin(TWO_PI * x))
    op = Fourier(grid)
    coeff = op.forward(f)
    space = coeff.function_space.bare
    assert space is mesh.fourier(origin=mesh.cell_avg)
    assert space.shape == (5,)
    back = op.backward(coeff)
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-14)


def test_staggered_origins_are_distinct_coefficient_spaces():
    grid, mesh = _grid1d(8)
    center = Fourier(grid).forward(grid.create_field(mesh.center))
    right = Fourier(grid).forward(grid.create_field(mesh.right))
    assert (center.function_space.bare
            is not right.function_space.bare)


# ================================================================
#  Layout (section 5.1: single-device trivial pencil)
# ================================================================
def test_codomain_layout_is_the_final_pencil():
    grid, _ = _grid1d(8)
    f = grid.create_field()
    coeff = Fourier(grid).forward(f)
    # single device: the plan has no reshards, the final pencil is
    # the operand's (trivial default) layout
    assert coeff.function_space.layout == (
        grid.decomposition.default_layout)
    assert coeff.function_space.layout == f.function_space.layout


# ================================================================
#  Designed-for surface
# ================================================================
def test_truncation_mask_is_designed_for():
    grid, mesh = _grid1d(8)
    with pytest.raises(NotImplementedError, match="Symbol"):
        Fourier(grid).truncation_mask(
            mesh.fourier(origin=mesh.center))


def test_fourier_rejects_bounded_meshes():
    mesh = IntervalMesh(8, (0.0, 1.0), periodic=False, name="x")
    grid = Grid((mesh,))
    f = grid.create_field(mesh.center)
    with pytest.raises(SpaceMismatchError, match="no Fourier"):
        Fourier(grid).forward(f)


# ================================================================
#  Fused all-Fourier fast path (one rfftn/fftn per plan)
# ================================================================
def _grid3d(nx=8, ny=6, nz=5):
    mx = IntervalMesh(nx, (0.0, 1.0), name="x")
    my = IntervalMesh(ny, (0.0, 2.0), name="y")
    mz = IntervalMesh(nz, (0.0, 3.0), name="z")
    return Grid((mx, my, mz))


def _force_staged(monkeypatch):
    """Disable the fused kernels: force the per-stage path."""
    monkeypatch.setattr(Fourier, "_forward_fused_kernel",
                        lambda *_args: None)
    monkeypatch.setattr(Fourier, "_backward_fused_kernel",
                        lambda *_args: None)


def test_fused_forward_matches_the_staged_stages(monkeypatch):
    grid = _grid3d()
    f = grid.random.normal(
        grid.create_field().function_space, seed=8)
    fused = Fourier(grid).forward(f)
    _force_staged(monkeypatch)
    staged = Fourier(grid).forward(f)
    assert fused.function_space is staged.function_space
    assert fused.dtype == staged.dtype
    scale = jnp.max(jnp.abs(staged.data))
    assert jnp.max(jnp.abs(fused.data - staged.data)) < 1e-13 * scale


def test_fused_backward_matches_the_staged_stages(monkeypatch):
    grid = _grid3d()
    op = Fourier(grid)
    coeff = op.forward(grid.random.normal(
        grid.create_field().function_space, seed=9))
    fused = op.backward(coeff)
    _force_staged(monkeypatch)
    staged = Fourier(grid).backward(coeff)
    assert fused.function_space is staged.function_space
    assert jnp.issubdtype(fused.dtype, jnp.floating)
    scale = jnp.max(jnp.abs(staged.data))
    assert jnp.max(jnp.abs(fused.data - staged.data)) < 1e-13 * scale


def test_fused_complex_domain_matches_the_staged_stages(monkeypatch):
    grid, mx, my = _grid2d(8, 6)
    space = mx.center.as_complex() * my.center.as_complex()
    data = (jnp.arange(48.0).reshape(8, 6)
            + 1j * jnp.linspace(-1.0, 1.0, 48).reshape(8, 6))
    f = grid.create_field(space, data=data)
    op = Fourier(grid)
    fused = op.forward(f)
    back = op.backward(fused)
    _force_staged(monkeypatch)
    staged_op = Fourier(grid)
    staged = staged_op.forward(f)
    staged_back = staged_op.backward(staged)
    assert fused.function_space is staged.function_space
    scale = jnp.max(jnp.abs(staged.data))
    assert jnp.max(jnp.abs(fused.data - staged.data)) < 1e-13 * scale
    assert jnp.max(jnp.abs(back.data - staged_back.data)) < 1e-13
    assert jnp.max(jnp.abs(back.data - data)) < 1e-13


def test_padded_plans_return_none_from_the_fused_kernels():
    grid, mesh = _grid1d(8)
    plain = Fourier(grid)
    padded = Fourier(grid, pad=degree(2))
    coeff = plain.forward(grid.random.normal(mesh.center, seed=10))
    fine = padded.backward(coeff)  # staged path (trims + phases)
    plan_b = padded.backward_plan(coeff.function_space)
    assert padded._backward_fused_kernel(
        jnp.asarray(coeff.data), plan_b) is None
    plan_f = padded.forward_plan(fine.function_space)
    assert padded._forward_fused_kernel(
        jnp.asarray(fine.data), plan_f) is None
    # pad-then-trim through the staged path stays exact
    again = padded.forward(fine)
    assert jnp.allclose(again.data, coeff.data, atol=1e-14)


def test_all_constant_plans_bypass_the_fused_kernels():
    grid, mx, my = _grid2d(8, 6)
    f = grid.create_field(mx.constant * my.constant)
    op = Fourier(grid)
    coeff = op.forward(f)  # zero stages: fused kernels return None
    back = op.backward(coeff)
    assert coeff.function_space is f.function_space
    assert back.function_space is f.function_space
