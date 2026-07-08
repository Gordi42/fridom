"""Phase-H1 symbolic eigenmodes: BlockSymbol-assembled eigh(iML, M).

The symbolic ``BlockSymbol`` assembly of ``L(k)`` from the model's
linear blocks (``fridom.framework2.model.symbolic_eigen``) is validated
on a shallow-water and a nonhydro periodic model against three
references:

- the **H0 numeric probe** (``model.eigen``): symbolic and numeric sit
  in the *same* raw staggered spectral basis, so both spectrum and
  eigenvectors agree to machine precision (away from the real-FFT
  Nyquist mode, a representation boundary);
- the **analytic discrete** nonhydro dispersion (``nh.eigenmodes``): the
  spectrum is basis-invariant, so it agrees to machine precision;
- the **analytic continuum** shallow-water dispersion (``sw.eigenmodes``):
  the symbolic spectrum is the *discrete* operator's, so it matches the
  continuum only to the second-order discretization error.

The eigenvectors are validated by M-orthonormality and by eigenspace-
projector agreement with the H0 numeric modes (same basis); a direct
component-wise comparison against the analytic ports is off by the
per-component staggering phase (raw-DFT-vs-staggered-transform), so it
is not asserted (see the module's basis note).
"""
import numpy as np
import pytest

import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.block_symbol import BlockSymbol
from fridom.framework2.model import symbolic_eigen
from fridom.framework2.model.eigen import numeric_eigenpairs
from fridom.framework2.model.model import Model
from fridom.framework2.model.symbolic_eigen import (
    SymbolicEigenmodes,
    symbolic_eigenpairs,
)
from fridom.framework2.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.framework2.modules.coriolis import FPlaneCoriolis
from fridom.nonhydro2.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)


# ================================================================
#  Model builders (shared with test_eigen.py)
# ================================================================
def sw_model(n=16, *, f0=1.0, csqr=1.0):
    """Return a doubly-periodic staggered shallow-water model."""
    mx = IntervalMesh(n, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, 1.0), periodic=True, name="y")
    return sw.Model(
        grid=Grid((mx, my)), csqr=csqr, rossby_number=0.2,
        coriolis=sw.modules.FPlaneCoriolis(f0=f0), advection=True,
        time_stepper=AdamBashforth(5e-3, order=3))


def nh_model(n=8, *, f0=1.0, n2=1.0, dsqr=1.0):
    """Return a triply-periodic nonhydro model (explicit dsqr, N2, f)."""
    grid = Grid(tuple(
        IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name=nm)
        for nm in ("x", "y", "z")))
    return Model(
        grid=grid,
        modules=(
            DynamicalCore(dsqr=dsqr, rossby_number=1.0),
            FPlaneCoriolis(f0=f0),
            ConstantStratification(n2=n2),
            CenteredAdvection()),
        time_stepper=AdamBashforth(0.02, order=3))


def _interior(omega, half):
    """Sorted spectrum over the interior (non-Nyquist) low modes."""
    trim = tuple(slice(1, half) for _ in range(omega.ndim - 1))
    return np.sort(np.asarray(omega)[trim], axis=-1)


# ================================================================
#  Shallow water (the unconstrained full A(k))
# ================================================================
def test_shallow_water_returns_three_branches():
    se = symbolic_eigenpairs(sw_model())
    assert isinstance(se, SymbolicEigenmodes)
    assert se.components == ("u", "v", "p")
    assert se.omega.shape[-1] == 3
    assert isinstance(se.operator, BlockSymbol)


def test_shallow_water_spectrum_is_real():
    se = symbolic_eigenpairs(sw_model())
    assert not np.iscomplexobj(np.asarray(se.omega))


def test_shallow_water_geostrophic_branch_is_zero():
    se = symbolic_eigenpairs(sw_model())
    omega = np.asarray(se.omega)
    assert np.abs(omega[..., 1]).max() < 1e-9


def test_shallow_water_spectrum_is_symmetric():
    se = symbolic_eigenpairs(sw_model())
    omega = np.asarray(se.omega)
    assert np.abs(omega[..., 0] + omega[..., 2]).max() < 1e-9


def test_shallow_water_matches_numeric_probe_spectrum():
    # symbolic and H0-numeric share the raw staggered basis: identical
    # spectrum away from the real-FFT Nyquist mode (idx n//2).
    model_kwargs = {"n": 16, "f0": 1.0, "csqr": 1.0}
    se = symbolic_eigenpairs(sw_model(**model_kwargs))
    ne = numeric_eigenpairs(sw_model(**model_kwargs))
    err = np.abs(_interior(se.omega, 8) - _interior(ne.omega, 8))
    assert err.max() < 1e-12


@pytest.mark.parametrize(("f0", "csqr"), [(1.0, 1.0), (0.5, 4.0)])
def test_shallow_water_fundamental_matches_continuum(f0, csqr):
    # the resolved fundamental k = (2pi, 0) matches the continuous
    # dispersion to the second-order discretization error (< 1%).
    se = symbolic_eigenpairs(sw_model(n=16, f0=f0, csqr=csqr))
    omega = np.asarray(se.omega)
    expect = np.sqrt(f0 ** 2 + csqr * (2 * np.pi) ** 2)
    numeric = omega[1, 0, 2]  # +branch at kx = 2pi, ky = 0
    assert abs(numeric - expect) / expect < 1e-2


def test_shallow_water_eigenvectors_are_m_orthonormal():
    se = symbolic_eigenpairs(sw_model())
    assert float(se.orthonormality_error()) < 1e-10


def test_shallow_water_eigenvectors_match_numeric_probe():
    # same-basis eigenvector validation: the M-orthogonal eigenspace
    # projectors agree with the H0 numeric modes to machine precision
    # (handles the degenerate geostrophic / +/- pairs without ordering).
    se = symbolic_eigenpairs(sw_model())
    ne = numeric_eigenpairs(sw_model())
    weights = np.asarray(se.weights)
    metric = np.diag(weights.astype(complex))
    qsym = np.asarray(se.q)
    qnum = np.asarray(ne.q)
    osym = np.asarray(se.omega)
    onum = np.asarray(ne.omega)
    worst = 0.0
    for i in range(1, 8):
        for j in range(1, 8):
            worst = max(worst, _projector_error(
                qsym[i, j], osym[i, j], qnum[i, j], onum[i, j], metric))
    assert worst < 1e-9


# ================================================================
#  Nonhydro (the Leray-constrained Schur complement)
# ================================================================
def test_nonhydro_returns_four_component_rows():
    se = symbolic_eigenpairs(nh_model())
    assert se.components == ("u", "v", "w", "b")
    assert se.omega.shape[-1] == 4


def test_nonhydro_metric_weights_are_read_from_the_model():
    se = symbolic_eigenpairs(nh_model(dsqr=2.0, n2=3.0))
    assert se.weights == pytest.approx((1.0, 1.0, 2.0, 1.0 / 3.0))


def test_nonhydro_spectrum_is_real():
    se = symbolic_eigenpairs(nh_model())
    assert not np.iscomplexobj(np.asarray(se.omega))


@pytest.mark.parametrize(
    ("f0", "n2", "dsqr"), [(1.0, 1.0, 1.0), (1.5, 3.0, 2.0)])
def test_nonhydro_spectrum_matches_the_analytic_discrete(f0, n2, dsqr):
    # the Leray-constrained symbolic operator reproduces the analytic
    # DISCRETE dispersion to machine precision on the interior modes
    # (the k = 0 mean is masked to 0 by the analytic convention).
    model = nh_model(n=8, f0=f0, n2=n2, dsqr=dsqr)
    se = symbolic_eigenpairs(model)
    omega = np.asarray(se.omega).reshape(5, 5, 5, 4)

    em = nh.eigenmodes.Eigenmodes(model.grid, f0=f0, n2=n2, dsqr=dsqr)
    analytic = np.sort(np.stack(
        [np.broadcast_to(np.asarray(em.omega(s)), (8, 8, 8))
         for s in (-1, 0, 1)], axis=-1), axis=-1)

    def drop_extra_zero(row):
        # the divergence constraint adds one exact zero -> 4 -> 3 DOF.
        return np.sort(np.delete(row, np.argmin(np.abs(row))))

    err = 0.0
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                sym3 = drop_extra_zero(omega[i, j, k])
                err = max(err, np.abs(sym3 - analytic[i, j, k]).max())
    assert err < 1e-9


def test_nonhydro_spectrum_matches_numeric_probe():
    # symbolic vs H0 numeric (both discrete, same basis): machine
    # precision on the interior modes.
    se = symbolic_eigenpairs(nh_model(dsqr=2.0, n2=3.0, f0=1.5))
    ne = numeric_eigenpairs(nh_model(dsqr=2.0, n2=3.0, f0=1.5))
    osym = np.sort(np.asarray(se.omega).reshape(5, 5, 5, 4), axis=-1)
    onum = np.sort(np.asarray(ne.omega).reshape(8, 8, 8, 4), axis=-1)
    err = np.abs(osym[1:4, 1:4, 1:4] - onum[1:4, 1:4, 1:4])
    assert err.max() < 1e-12


def test_nonhydro_has_a_divergence_free_nullspace():
    # generic modes carry two zeros: the geostrophic mode and the
    # constraint (divergence-free-complement) nullspace.
    se = symbolic_eigenpairs(nh_model())
    omega = np.asarray(se.omega)
    near_zero = np.sum(np.abs(omega) < 1e-9, axis=-1)
    assert near_zero.max() >= 2


def test_nonhydro_eigenvectors_are_m_orthonormal():
    se = symbolic_eigenpairs(nh_model(dsqr=2.0, n2=3.0))
    assert float(se.orthonormality_error()) < 1e-10


def test_nonhydro_operator_is_a_block_symbol():
    se = symbolic_eigenpairs(nh_model())
    assert isinstance(se.operator, BlockSymbol)
    assert se.operator.matrix.shape[-2:] == (4, 4)


# ================================================================
#  Structural gate + internal branches
# ================================================================
def test_rejects_a_beta_plane_model():
    mx = IntervalMesh(16, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(16, (0.0, 1.0), periodic=True, name="y")
    model = sw.Model(
        grid=Grid((mx, my)), csqr=1.0,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=1.0, beta=2.0),
        time_stepper=AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"coriolis\.f0|Fourier"):
        symbolic_eigenpairs(model)


def test_at_time_is_threaded_to_the_metric_and_blocks():
    # a constant-parameter model is time-invariant; an explicit at_time
    # gives the identical spectrum (exercises the at_time path).
    a = symbolic_eigenpairs(sw_model(), at_time=0.0)
    b = symbolic_eigenpairs(sw_model(), at_time=1.5)
    assert np.abs(np.asarray(a.omega) - np.asarray(b.omega)).max() < 1e-12


def test_blocks_sharing_an_entry_accumulate(monkeypatch):
    # defensive accumulation branch of _assemble_raw: two blocks on the
    # same (out, src) sum. Duplicate a real block and assert the entry
    # doubles vs the single-block operator.
    model = sw_model()
    blocks = symbolic_eigen.linear_blocks(model)
    first = blocks[0]
    prog = ("u", "v", "p")

    single = symbolic_eigen._assemble_raw(model, prog, 0.0)
    monkeypatch.setattr(
        symbolic_eigen, "linear_blocks",
        lambda _model, **_kw: (*blocks, first))
    doubled = symbolic_eigen._assemble_raw(model, prog, 0.0)

    row = prog.index(first.out)
    col = prog.index(first.src)
    single_entry = np.asarray(single.matrix)[..., row, col]
    doubled_entry = np.asarray(doubled.matrix)[..., row, col]
    assert np.abs(doubled_entry - 2.0 * single_entry).max() < 1e-12


# ================================================================
#  Eigenspace-projector helper
# ================================================================
def _projector_error(qsym, osym, qnum, onum, metric):
    """Max deviation between the sym/num M-orthogonal eigenprojectors."""
    def projectors(q, omega):
        result = []
        used = np.zeros(len(omega), bool)
        for a in range(len(omega)):
            if used[a]:
                continue
            group = np.abs(omega - omega[a]) < 1e-6
            used |= group
            block = q[:, group]
            result.append(
                (omega[a], block @ (block.conj().T @ metric)))
        return result

    worst = 0.0
    for value, proj in projectors(qsym, osym):
        match = min(projectors(qnum, onum),
                    key=lambda pair: abs(pair[0] - value))
        worst = max(worst, np.abs(proj - match[1]).max())
    return worst
