"""Wet-aware (immersed / cut-cell) column oracles for VerticalDiffusion.

The prefix-mirrored shard of ``test_implicit_kernel`` (AGENTS.md
oversized-module rule) covering the immersed arm of the measure-aware
band builder ``_diffusion_bands`` and its ``_face_fraction`` helper
(immersed_closures_sadourny_plan §5, the implicit twin of the CL-D2
explicit spelling): a dry cell is an identity row, a wet/dry face
carries no flux (the free-slip immersed boundary), a wet partial cell
carries its true wet width, and the ``theta``-weighted wet content
telescopes to the wall fluxes (conserved for Neumann). The gates:

- all-wet ≡ unimmersed, **bitwise** (band and solve, every BC);
- immersed **staircase** ≡ the equivalent **walled / shortened** column,
  bitwise (the min-rule ``alpha = 0`` at the wet/dry face is the exact
  no-flux wall);
- dry cells are identity rows (``lower = diag = upper = 0``) and the
  solve leaves the dry values untouched;
- a genuine partial column matches a hand-built physical reference and
  conserves the wet-width-weighted sum to machine precision;
- the immersed geometry is grid-global, so it does not disturb the
  ``merge_key`` discipline.

Self-contained: the small builders are duplicated, not imported.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.model.implicit import (
    VerticalDiffusion,
    _diffusion_bands,
    _face_fraction,
)
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain, Slip
from fridom.spatial.meshes.interval import IntervalMesh

N = 12
KAPPA = 0.05


# ================================================================
#  Builders
# ================================================================
def kappa_const(value):
    def kappa(_module, _state, _ctx, _name):
        return value
    return kappa


def z_mesh(n=N, extent=(0.0, 1.0)):
    return IntervalMesh(n, extent, periodic=False, name="z")


def plain_grid(n=N, extent=(0.0, 1.0)):
    """Return a single bounded z column, no immersed geometry."""
    mz = z_mesh(n, extent)
    return Grid((mz,)), mz


def immersed_grid(indicator, *, n=N, extent=(0.0, 1.0), order=None,
                  slip=Slip.FREE_SLIP):
    """Return a single bounded z column carrying an immersed wet region."""
    mz = z_mesh(n, extent)
    grid = Grid((mz,), immersed=ImmersedDomain(
        indicator, order=order, slip=slip))
    return grid, grid.factors[0]


def bands(grid, mesh, kappa=KAPPA, bc=("neumann", "neumann")):
    field = grid.create_field(mesh.center, data=jnp.zeros(
        mesh.center.shape))
    lower, diag, upper, _ = _diffusion_bands(field, "z", kappa, bc)
    return (np.asarray(lower), np.asarray(diag), np.asarray(upper))


def apply_op(grid, mesh, values, kappa=KAPPA, bc=("neumann", "neumann")):
    op = VerticalDiffusion("z", ("b",), kappa_const(kappa), bc=bc)
    state = VectorField(
        {"b": grid.create_field(mesh.center, data=jnp.asarray(values))})
    return np.asarray(op.apply(None, state, None)["b"].data)


def solve_op(grid, mesh, values, dt_gamma, kappa=KAPPA,
             bc=("neumann", "neumann")):
    op = VerticalDiffusion("z", ("b",), kappa_const(kappa), bc=bc)
    rhs = {"b": grid.create_field(mesh.center, data=jnp.asarray(values))}
    return np.asarray(op.solve(None, rhs, dt_gamma, None)["b"].data)


# ================================================================
#  1. all-wet ≡ unimmersed, bitwise (the alpha = theta = 1 no-op)
# ================================================================
@pytest.mark.parametrize(
    "bc", [("neumann", "neumann"), ("dirichlet", "neumann"),
           ("dirichlet", "dirichlet")])
def test_all_wet_band_is_bitwise_the_unimmersed_band(bc):
    # an all-wet immersed column has alpha = theta = 1 exactly, so the
    # wet weighting is a bitwise no-op (a * 1.0 / 1.0): every band entry
    # equals the plain column bit-for-bit
    gp, mp = plain_grid()
    gw, mw = immersed_grid(lambda z: z * 0.0 + 1.0)
    for band_p, band_w in zip(bands(gp, mp, bc=bc),
                              bands(gw, mw, bc=bc), strict=True):
        assert np.array_equal(band_p, band_w)


@pytest.mark.parametrize(
    "bc", [("neumann", "neumann"), ("dirichlet", "dirichlet")])
def test_all_wet_solve_is_bitwise_the_unimmersed_solve(bc):
    # the full solve reduces bit-for-bit as well
    gp, mp = plain_grid()
    gw, mw = immersed_grid(lambda z: z * 0.0 + 1.0)
    rng = np.random.default_rng(4)
    col = rng.standard_normal(N)
    sp = solve_op(gp, mp, col, 0.4, bc=bc)
    sw = solve_op(gw, mw, col, 0.4, bc=bc)
    assert np.array_equal(sp, sw)


# ================================================================
#  2. staircase ≡ walled / shortened column, bitwise
# ================================================================
def _staircase_grid():
    # cells at (i+0.5)/12: z > 0.5 wet => i = 6..11 wet, i = 0..5 dry
    return immersed_grid(lambda z: (z > 0.5).astype(float))


def test_staircase_theta_is_a_clean_half_column():
    grid, mesh = _staircase_grid()
    theta = np.asarray(grid.immersed.fraction(mesh.center).data)
    assert np.array_equal(theta, np.array([0.0] * 6 + [1.0] * 6))


@pytest.mark.parametrize("high", ["neumann", "dirichlet"])
def test_staircase_apply_matches_the_walled_shortened_column(high):
    # A-G1 keystone (implicit): the wet sub-column of a face-aligned
    # {0, 1} staircase reproduces the plain 6-cell walled column on the
    # wet sub-domain BITWISE. The wet column's BOTTOM faces the dry
    # region, so it is the free-slip immersed boundary (Neumann, the
    # min-rule alpha = 0 exact no-flux wall) regardless of the declared
    # low BC; its TOP is the real domain wall carrying `high`.
    grid, mesh = _staircase_grid()
    wal, mwal = plain_grid(6, extent=(0.5, 1.0))
    rng = np.random.default_rng(7)
    wet = rng.standard_normal(6)
    full = np.concatenate([np.full(6, 3.21), wet])  # arbitrary dry fill
    got = apply_op(grid, mesh, full, bc=("dirichlet", high))
    ref = apply_op(wal, mwal, wet, bc=("neumann", high))
    assert np.array_equal(got[6:], ref)
    # the dry half carries no tendency at all
    assert np.array_equal(got[:6], np.zeros(6))


def test_staircase_bottom_is_free_slip_independent_of_the_low_bc():
    # the immersed wet/dry interface is free-slip (alpha = 0), so the
    # declared LOW wall BC (buried in the dry region) cannot reach the
    # wet column: dirichlet-low and neumann-low give identical wet rows
    grid, mesh = _staircase_grid()
    rng = np.random.default_rng(8)
    full = np.concatenate([rng.standard_normal(6), rng.standard_normal(6)])
    got_d = apply_op(grid, mesh, full, bc=("dirichlet", "neumann"))
    got_n = apply_op(grid, mesh, full, bc=("neumann", "neumann"))
    assert np.array_equal(got_d[6:], got_n[6:])


# ================================================================
#  3. dry cells are identity rows; the solve leaves them untouched
# ================================================================
def test_dry_cells_are_identity_rows():
    grid, mesh = _staircase_grid()
    lower, diag, upper = bands(grid, mesh, bc=("dirichlet", "dirichlet"))
    for band in (lower, diag, upper):
        assert np.array_equal(band[:6], np.zeros(6))


def test_solve_leaves_dry_values_untouched():
    # a dry (theta = 0) identity row solves x = rhs: the dry values pass
    # through unchanged and no diffusion leaks into them
    grid, mesh = _staircase_grid()
    rng = np.random.default_rng(9)
    full = np.concatenate([rng.standard_normal(6), rng.standard_normal(6)])
    solved = solve_op(grid, mesh, full, 3.0, bc=("neumann", "neumann"))
    assert np.array_equal(solved[:6], full[:6])
    # the wet half genuinely moved (the solve is not a global identity)
    assert np.abs(solved[6:] - full[6:]).max() > 1e-6


def test_constant_wet_field_has_zero_tendency():
    # a uniform field over the wet region is annihilated (the differences
    # q_{c+1} - q_c vanish): no spurious partial-cell mixing. Machine
    # zero, not bitwise — the partial-cell diag = -(low + up) does not
    # cancel the off-diagonals to the last bit when low != up
    grid, mesh = immersed_grid(
        lambda z: ((z > 0.22) & (z < 0.9)).astype(float), order=3)
    got = apply_op(grid, mesh, np.full(N, 4.5))
    assert np.abs(got).max() < 1e-14


# ================================================================
#  4. genuine partial column: hand-built reference + conservation
# ================================================================
def _partial_grid(order=3):
    grid, mesh = immersed_grid(
        lambda z: ((z > 0.22) & (z < 0.9)).astype(float), order=order)
    return grid, mesh


def test_partials_are_genuinely_fractional():
    # the quadrature theta is strictly in (0, 1) somewhere (the seal
    # precondition — a {0, 1} staircase would not exercise the /theta)
    grid, mesh = _partial_grid()
    theta = np.asarray(grid.immersed.fraction(mesh.center).data)
    assert np.any((theta > 0.0) & (theta < 1.0))


def test_partial_band_matches_a_hand_built_wet_reference():
    # gate (iv): the Neumann partial-column band equals the physical
    # reference alpha_f kappa / (dz_f theta_c dz_c) built from the wet
    # fraction, the min-rule face fractions (walls = 1) and the measures
    grid, mesh = _partial_grid()
    lower, diag, upper = bands(grid, mesh, bc=("neumann", "neumann"))
    theta = np.asarray(grid.immersed.fraction(mesh.center).data)
    mcell = np.asarray(grid.measure(mesh.center, "z").data).reshape(-1)
    mface = np.asarray(grid.measure(mesh.outer, "z").data).reshape(-1)

    # min-rule face fractions, domain walls unweighted (alpha = 1)
    inter = np.minimum(theta[:-1], theta[1:])
    faces = np.concatenate([[1.0], inter, [1.0]])
    a_up, a_low = faces[1:], faces[:-1]
    wet = theta > 0.0
    theta_safe = np.where(wet, theta, 1.0)
    up = np.where(wet, a_up * KAPPA / (mface[1:] * mcell) / theta_safe, 0.0)
    low = np.where(wet, a_low * KAPPA / (mface[:-1] * mcell) / theta_safe,
                   0.0)
    ref_lower = low.copy()
    ref_lower[0] = 0.0
    ref_upper = up.copy()
    ref_upper[-1] = 0.0
    ref_diag = -(ref_lower + ref_upper)  # Neumann drops both walls

    assert np.allclose(lower, ref_lower, rtol=1e-13, atol=1e-15)
    assert np.allclose(upper, ref_upper, rtol=1e-13, atol=1e-15)
    assert np.allclose(diag, ref_diag, rtol=1e-13, atol=1e-15)


def test_partial_neumann_conserves_the_wet_width_weighted_sum():
    # gate (iii) at the band level: sum_c theta_c dz_c (L q)_c telescopes
    # to the (zero) Neumann wall fluxes — machine-zero wet content rate
    grid, mesh = _partial_grid()
    theta = np.asarray(grid.immersed.fraction(mesh.center).data)
    mcell = np.asarray(grid.measure(mesh.center, "z").data).reshape(-1)
    rng = np.random.default_rng(2)
    col = rng.standard_normal(N)
    tendency = apply_op(grid, mesh, col)
    rate = float(np.sum(theta * mcell * tendency))
    assert abs(rate) < 1e-14


def test_partial_solve_conserves_the_wet_width_weighted_sum():
    # the implicit Neumann solve preserves the wet content across a step
    grid, mesh = _partial_grid()
    theta = np.asarray(grid.immersed.fraction(mesh.center).data)
    mcell = np.asarray(grid.measure(mesh.center, "z").data).reshape(-1)
    rng = np.random.default_rng(3)
    col = rng.standard_normal(N)
    before = float(np.sum(theta * mcell * col))
    solved = solve_op(grid, mesh, col, 0.6)
    after = float(np.sum(theta * mcell * solved))
    assert after == pytest.approx(before, abs=1e-13)


# ================================================================
#  5. _face_fraction unit: min-rule interior, walls = 1, shared face
# ================================================================
def test_face_fraction_is_the_min_rule_with_unweighted_walls():
    theta = jnp.asarray([0.0, 0.3, 1.0, 0.7, 0.0])
    alpha_up, alpha_low = _face_fraction(theta, 0, 5)
    # interior faces are the pairwise min of adjacent cells (0, 0.3,
    # 0.7, 0); the two domain walls are unweighted at 1
    assert np.array_equal(np.asarray(alpha_up),
                          np.array([0.0, 0.3, 0.7, 0.0, 1.0]))
    assert np.array_equal(np.asarray(alpha_low),
                          np.array([1.0, 0.0, 0.3, 0.7, 0.0]))
    # the shared face is the SAME element (alpha_up[c] == alpha_low[c+1])
    assert np.array_equal(np.asarray(alpha_up)[:-1],
                          np.asarray(alpha_low)[1:])


# ================================================================
#  6. merge-key discipline is undisturbed by grid-global geometry
# ================================================================
def test_immersed_geometry_does_not_enter_the_merge_key():
    # the wet weighting is a grid property, not operator state, so the
    # merge_key stays (type, axis, bc): same-BC legs kappa-merge, an
    # unlike-BC (no-slip) leg does not merge-collide with a Neumann leg
    neu_a = VerticalDiffusion("z", ("b",), kappa_const(0.1))
    neu_b = VerticalDiffusion("z", ("u",), kappa_const(0.2))
    dir_c = VerticalDiffusion("z", ("u",), kappa_const(0.3),
                              bc=("dirichlet", "dirichlet"))
    assert neu_a.merge_key() == neu_b.merge_key()
    assert neu_a.merge_key() != dir_c.merge_key()
    merged = neu_a.merged_with(neu_b)
    assert merged.fields == ("b", "u")


def test_kappa_summed_merge_stays_exact_on_an_immersed_grid():
    # two same-BC legs on one immersed grid kappa-merge: the band of the
    # merged (0.1 + 0.2) operator equals the sum of the two single bands
    # (the wet weighting is linear in kappa)
    grid, mesh = _partial_grid()
    field = grid.create_field(mesh.center, data=jnp.zeros(N))
    b1 = _diffusion_bands(field, "z", 0.1, ("neumann", "neumann"))
    b2 = _diffusion_bands(field, "z", 0.2, ("neumann", "neumann"))
    bs = _diffusion_bands(field, "z", 0.3, ("neumann", "neumann"))
    for single1, single2, summed in zip(b1[:3], b2[:3], bs[:3],
                                        strict=True):
        assert np.allclose(np.asarray(single1) + np.asarray(single2),
                           np.asarray(summed), rtol=1e-13, atol=1e-15)
