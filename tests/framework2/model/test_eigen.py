"""Phase-H0 numeric eigenmodes: eigh(iML, M) vs the analytic modes.

The transfer-function probe + generalized Hermitian eigensolve
(``fridom.framework2.model.eigen``) is validated against the analytic
Tier-0 eigenmode ports on BOTH a shallow-water and a nonhydro periodic
model:

- the **spectrum** ``omega`` is basis-invariant and is compared to the
  analytic dispersion (nonhydro: the discrete relation, to machine
  precision; shallow water: the continuum relation at the resolved
  fundamental, to the discretization error);
- the **eigenvectors** are validated by self-consistency (real spectrum
  + M-orthonormality). A raw-DFT-vs-staggered-transform phase makes a
  direct component-wise / projector comparison against the analytic
  ``q`` awkward (reported in the H0 notes), so it is not asserted here.
"""
import numpy as np
import pytest

import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.composed import Divergence
from fridom.framework2.model.eigen import (
    NumericEigenmodes,
    numeric_eigenpairs,
)
from fridom.framework2.model.model import Model
from fridom.framework2.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.framework2.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
)
from fridom.nonhydro2.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)


# ================================================================
#  Model builders
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


# ================================================================
#  Shallow water (unconstrained)
# ================================================================
def test_shallow_water_returns_three_branches():
    ne = numeric_eigenpairs(sw_model())
    assert isinstance(ne, NumericEigenmodes)
    assert ne.components == ("u", "v", "p")
    assert ne.omega.shape[-1] == 3


def test_shallow_water_spectrum_is_real():
    ne = numeric_eigenpairs(sw_model())
    assert not np.iscomplexobj(np.asarray(ne.omega))


def test_shallow_water_geostrophic_branch_is_zero():
    # the middle (sorted) branch is the geostrophic omega = 0 everywhere
    ne = numeric_eigenpairs(sw_model())
    omega = np.asarray(ne.omega)
    assert np.abs(omega[..., 1]).max() < 1e-9


def test_shallow_water_spectrum_is_symmetric():
    # inertia-gravity branches come in +/- omega pairs
    ne = numeric_eigenpairs(sw_model())
    omega = np.asarray(ne.omega)
    assert np.abs(omega[..., 0] + omega[..., 2]).max() < 1e-9


@pytest.mark.parametrize(("f0", "csqr"), [(1.0, 1.0), (0.5, 4.0)])
def test_shallow_water_fundamental_matches_continuum(f0, csqr):
    # the resolved fundamental k = (2pi, 0) matches the continuous
    # dispersion to the (2nd-order) discretization error (< 1% at n=16)
    ne = numeric_eigenpairs(sw_model(n=16, f0=f0, csqr=csqr))
    omega = np.asarray(ne.omega)
    expect = np.sqrt(f0 ** 2 + csqr * (2 * np.pi) ** 2)
    numeric = omega[1, 0, 2]  # +branch at kx = 2pi, ky = 0
    assert abs(numeric - expect) / expect < 1e-2


def test_shallow_water_eigenvectors_are_m_orthonormal():
    ne = numeric_eigenpairs(sw_model())
    assert float(ne.orthonormality_error()) < 1e-10


# ================================================================
#  Nonhydro constrained modes
# ================================================================
def test_nonhydro_returns_four_component_rows():
    ne = numeric_eigenpairs(nh_model())
    assert ne.components == ("u", "v", "w", "b")
    assert ne.omega.shape[-1] == 4


def test_nonhydro_metric_weights_are_read_from_the_model():
    ne = numeric_eigenpairs(nh_model(dsqr=2.0, n2=3.0))
    # diag(1, 1, dsqr, 1/N^2) on (u, v, w, b)
    assert ne.weights == pytest.approx((1.0, 1.0, 2.0, 1.0 / 3.0))


def test_nonhydro_spectrum_is_real():
    ne = numeric_eigenpairs(nh_model())
    assert not np.iscomplexobj(np.asarray(ne.omega))


@pytest.mark.parametrize(
    ("f0", "n2", "dsqr"), [(1.0, 1.0, 1.0), (1.5, 3.0, 2.0)])
def test_nonhydro_spectrum_matches_the_analytic_discrete(f0, n2, dsqr):
    # The energy-metric eigensolve reproduces the analytic DISCRETE
    # dispersion to machine precision on every mode except the k = 0
    # mean (the analytic ports mask its physical inertial +/- f to 0).
    model = nh_model(n=8, f0=f0, n2=n2, dsqr=dsqr)
    ne = numeric_eigenpairs(model)
    omega = np.asarray(ne.omega).reshape(-1, 4)

    em = nh.eigenmodes.Eigenmodes(model.grid, f0=f0, n2=n2, dsqr=dsqr)
    # em.omega(s).data lives on the rfft half-lattice (x half-spectrum);
    # unfold to the full fftn lattice via evenness: full index j on the
    # x axis carries |mode| = min(j, 8 - j) = the half-lattice index.
    fold = np.minimum(np.arange(8), 8 - np.arange(8))

    def full_omega(s):
        half = np.broadcast_to(np.asarray(em.omega(s).data), (5, 8, 8))
        return half[fold]

    analytic = np.sort(np.stack(
        [full_omega(s) for s in (-1, 0, 1)], axis=-1),
        axis=-1).reshape(-1, 3)

    # the divergence constraint removes one velocity DOF -> one extra
    # exact zero; drop the eigenvalue nearest zero and compare 3 vs 3.
    def drop_extra_zero(row):
        return np.sort(np.delete(row, np.argmin(np.abs(row))))

    numeric3 = np.array([drop_extra_zero(row) for row in omega])
    err = np.abs(numeric3 - analytic)
    err[0] = 0.0  # exclude the k = 0 mean (masking convention)
    assert err.max() < 1e-9


def test_nonhydro_has_a_divergence_free_nullspace():
    # generic modes carry two zeros: the geostrophic mode and the
    # constraint (divergence-free-complement) nullspace.
    ne = numeric_eigenpairs(nh_model())
    omega = np.asarray(ne.omega)
    near_zero = np.sum(np.abs(omega) < 1e-9, axis=-1)
    # some interior mode has >= 2 zero eigenvalues
    assert near_zero.max() >= 2


def test_nonhydro_eigenvectors_are_m_orthonormal():
    ne = numeric_eigenpairs(nh_model(dsqr=2.0, n2=3.0))
    assert float(ne.orthonormality_error()) < 1e-10


# ================================================================
#  Structural gates + the H1 constrained-matvec surface
# ================================================================
def test_rejects_a_walled_grid():
    mx = IntervalMesh(8, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(8, (0.0, 1.0), periodic=True, name="y")
    mz = IntervalMesh(8, (0.0, 1.0), periodic=False, name="z")
    model = nh.Model(
        grid=Grid((mx, my, mz)), advection=False,
        coriolis=FPlaneCoriolis(f0=1.0),
        time_stepper=AdamBashforth(5e-3, order=3))
    with pytest.raises(
            ValueError,
            match=r"bounded axes \('z',\).*channel_eigenpairs"):
        numeric_eigenpairs(model)


def test_rejects_a_beta_plane_model():
    mx = IntervalMesh(16, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(16, (0.0, 1.0), periodic=True, name="y")
    model = sw.Model(
        grid=Grid((mx, my)), csqr=1.0,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=1.0, beta=2.0),
        time_stepper=AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"coriolis\.f0|Fourier"):
        numeric_eigenpairs(model)


def test_rejects_a_varying_csqr_model():
    # the beta-style gate extended to metric coefficients: a
    # varying csqr(y) on a periodic grid is not translation
    # invariant; the taught error points at the channel engine
    mx = IntervalMesh(8, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(8, (0.0, 1.0), periodic=True, name="y")
    model = sw.Model(
        grid=Grid((mx, my)),
        csqr=lambda y: 1.0 + 0.5 * np.sin(2 * np.pi * y),
        advection=False,
        time_stepper=AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"csqr.*channel"):
        numeric_eigenpairs(model)


def test_rejects_a_varying_stratification_model():
    grid = Grid(tuple(
        IntervalMesh(8, (0.0, 2 * np.pi), periodic=True, name=nm)
        for nm in ("x", "y", "z")))
    model = nh.Model(
        grid=grid, advection=False,
        stratification=nh.MeridionalStratification(
            n2=lambda y: 1.0 + y * y),
        time_stepper=AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"n2.*channel"):
        numeric_eigenpairs(model)


def test_constrained_tendency_is_the_projected_tendency():
    # the H1 accumulator fix: tendency(constraints=True) returns the
    # PROJECTED prognostic tendency — the diagnostic 'p' stays on the
    # internal overlay — with a machine-zero discrete divergence
    model = nh_model()
    rng = np.random.default_rng(7)
    model.set_fields(**{c: rng.standard_normal(
        model.state[c].data.shape) for c in ("u", "v", "w", "b")})
    tau = model.tendency(model.state, constraints=True)
    assert tau.component_names == ("u", "v", "w", "b")
    div = Divergence()(VectorField(
        {c: tau[c] for c in ("u", "v", "w")}))
    assert np.abs(np.asarray(div.data)).max() < 1e-12


def test_constrain_matvec_is_the_idempotent_leray_projector():
    # the H1 public projector surface: model.constrain projects onto
    # the divergence-free subspace and is idempotent
    model = nh_model()
    rng = np.random.default_rng(8)
    model.set_fields(**{c: rng.standard_normal(
        model.state[c].data.shape) for c in ("u", "v", "w", "b")})
    pz = model.constrain(model.state)
    assert pz.component_names == ("u", "v", "w", "b")
    div = Divergence()(VectorField(
        {c: pz[c] for c in ("u", "v", "w")}))
    assert np.abs(np.asarray(div.data)).max() < 1e-12
    twice = model.constrain(pz)
    err = max(float(np.abs(np.asarray(twice[c].data)
                           - np.asarray(pz[c].data)).max())
              for c in ("u", "v", "w", "b"))
    assert err < 1e-12


def test_betaplane_coriolis_module_is_shared():
    # guard: the builders use the shared fr.modules Coriolis
    assert nh.FPlaneCoriolis is FPlaneCoriolis
    assert nh.BetaPlaneCoriolis is BetaPlaneCoriolis
