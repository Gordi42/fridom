"""Dealiasing tests: PadFactor/degree and the padded transforms."""
from fractions import Fraction

import jax.numpy as jnp
import pytest

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.dealias import PadFactor, degree
from fridom.framework2.grid.operators.fourier import Fourier
from fridom.framework2.grid.operators.products import (
    CollocationProduct,
)

TWO_PI = 2.0 * jnp.pi


def _grid(n=8):
    mesh = IntervalMesh(n, (0.0, 1.0), name="x")
    return Grid((mesh,)), mesh


# ================================================================
#  PadFactor and the degree factory
# ================================================================
def test_degree_two_is_the_three_halves_rule():
    assert degree(2) == PadFactor(Fraction(3, 2))


def test_degree_three_pads_by_two():
    assert degree(3).factor == Fraction(2)


def test_degree_one_is_the_identity_pad():
    assert degree(1).factor == Fraction(1)


def test_padfactor_normalizes_ints():
    assert PadFactor(2).factor == Fraction(2)


def test_padfactor_rejects_floats():
    with pytest.raises(TypeError, match="exact Fractions"):
        PadFactor(1.5)


def test_padfactor_rejects_coarsening():
    with pytest.raises(ValueError, match="refine"):
        PadFactor(Fraction(1, 2))


def test_degree_validates():
    with pytest.raises(TypeError, match="integers"):
        degree(1.5)
    with pytest.raises(ValueError, match="positive"):
        degree(0)


# ================================================================
#  Padded backward: onto the refined mesh's finer nodal space
# ================================================================
def test_padded_backward_lands_on_the_finer_nodal_space():
    grid, mesh = _grid(8)
    f = grid.create_field(init=lambda x: jnp.sin(TWO_PI * x))
    coeff = Fourier(grid).forward(f)
    padded = Fourier(grid, pad=degree(2))
    fine = padded.backward(coeff)
    fine_mesh = mesh.refined(Fraction(3, 2))
    assert fine.function_space.bare is fine_mesh.center
    assert fine.shape == (12,)
    # the padded synthesis samples the same trig interpolant
    x_fine = grid.evaluation_nodes(fine_mesh.center).data
    assert jnp.allclose(fine.data, jnp.sin(TWO_PI * x_fine))


def test_padded_backward_splits_the_nyquist_mode_exactly():
    grid, mesh = _grid(8)
    # pure Nyquist content: sin(2 pi 4 x) at centers is (-1)^j (the
    # Nyquist *cosine* vanishes identically at centers)
    f = grid.create_field(init=lambda x: jnp.sin(TWO_PI * 4 * x))
    coeff = Fourier(grid).forward(f)
    assert jnp.abs(coeff.data[-1] - 1.0) < 1e-14
    fine = Fourier(grid, pad=degree(2)).backward(coeff)
    x_fine = grid.evaluation_nodes(
        mesh.refined(Fraction(3, 2)).center).data
    assert jnp.allclose(fine.data, jnp.sin(TWO_PI * 4 * x_fine))


# ================================================================
#  Padded forward: the codomain exception and the exact trim
# ================================================================
def test_padded_forward_codomain_is_the_coarse_space():
    grid, mesh = _grid(8)
    fine_mesh = mesh.refined(Fraction(3, 2))
    padded = Fourier(grid, pad=degree(2))
    codomain = padded.codomain(fine_mesh.center)
    # deliberate exception: origin is the COARSE Center(8)
    assert codomain is mesh.fourier(origin=mesh.center)


def test_padded_forward_rejects_the_coarse_domain():
    grid, mesh = _grid(8)
    padded = Fourier(grid, pad=degree(2))
    with pytest.raises(SpaceMismatchError, match="refined"):
        padded.codomain(mesh.center)


@pytest.mark.parametrize(("n", "p"), [
    pytest.param(8, 2, id="even-n-3/2"),
    pytest.param(9, 3, id="odd-n-2"),
])
def test_pad_then_trim_is_the_identity(n, p):
    grid, mesh = _grid(n)
    f = grid.random.normal(mesh.center, seed=7)
    coeff = Fourier(grid).forward(f)
    padded = Fourier(grid, pad=degree(p))
    round_trip = padded.forward(padded.backward(coeff))
    assert round_trip.function_space is coeff.function_space
    assert jnp.allclose(round_trip.data, coeff.data)


# ================================================================
#  Aliasing removal on a quadratic product (the 3/2 rule)
# ================================================================
def test_quadratic_product_dealiasing():
    n = 8
    grid, mesh = _grid(n)
    # u = sin(2 pi 3 x): u^2 = 1/2 - cos(2 pi 6 x)/2; mode 6 exceeds
    # the Nyquist mode 4 and, on the coarse grid, aliases onto -2.
    u = grid.create_field(init=lambda x: jnp.sin(TWO_PI * 3 * x))
    fourier = Fourier(grid)
    padded = Fourier(grid, pad=degree(2))

    # aliased path: collocation product on the coarse grid — the
    # unresolved mode 6 folds onto mode 2 with amplitude 1/4
    aliased = fourier.forward(u * u)
    assert jnp.abs(jnp.abs(aliased.data[2]) - 0.25) < 1e-14

    # dealiased path: padded backward, product, padded forward
    fine_mesh = mesh.refined(Fraction(3, 2))
    grid.dispatch[("multiply", fine_mesh.center)] = (
        CollocationProduct())
    u_fine = padded.backward(fourier.forward(u))
    dealiased = padded.forward(u_fine * u_fine)
    assert dealiased.function_space is aliased.function_space

    # exact truncated product: only the mean survives in-band
    expected = jnp.zeros(n // 2 + 1, dtype=dealiased.dtype)
    expected = expected.at[0].set(0.5)
    assert jnp.allclose(dealiased.data, expected, atol=1e-14)


@pytest.mark.parametrize(("n", "p"), [
    pytest.param(8, 2, id="even-n-3/2"),
    pytest.param(9, 3, id="odd-n-2"),
])
def test_pad_then_trim_is_the_identity_on_full_spectra(n, p):
    # complex origins exercise the full-spectrum embed/trim pair
    grid, mesh = _grid(n)
    space = mesh.center.as_complex()
    rng = jnp.linspace(0.0, 1.0, n)
    f = grid.create_field(space, data=jnp.exp(2j * jnp.pi * rng))
    coeff = Fourier(grid).forward(f)
    assert coeff.shape == (n,)
    padded = Fourier(grid, pad=degree(p))
    fine = padded.backward(coeff)
    assert fine.function_space.bare is mesh.refined(
        Fraction(p + 1, 2)).center.as_complex()
    round_trip = padded.forward(fine)
    assert round_trip.function_space is coeff.function_space
    assert jnp.allclose(round_trip.data, coeff.data, atol=1e-14)


def test_full_spectrum_padded_synthesis_hits_the_fine_nodes():
    grid, mesh = _grid(8)
    space = mesh.center.as_complex()
    x = grid.evaluation_nodes(mesh.center).data
    f = grid.create_field(space, data=jnp.exp(1j * TWO_PI * x))
    coeff = Fourier(grid).forward(f)
    fine = Fourier(grid, pad=degree(2)).backward(coeff)
    x_fine = grid.evaluation_nodes(
        mesh.refined(Fraction(3, 2)).center).data
    assert jnp.allclose(fine.data, jnp.exp(1j * TWO_PI * x_fine),
                        atol=1e-14)
