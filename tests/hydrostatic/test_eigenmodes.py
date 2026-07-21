"""Numeric eigenmodes of the discrete hydrostatic C-grid (HY-D7).

The labeled eigenbasis of the assembled explicit hydrostatic linear
operator on the doubly-periodic flat-bottom grid (periodic x/y, bounded
z with the free surface). Covers the passthrough surface, the H2 energy
metric, machine-precision biorthogonality, the per-plane geostrophic /
inertia-gravity split with its six-family census, the barotropic
Poincaré dispersion oracle, the discrete baroclinic vertical wavenumber,
the mode accessor, the implicit / rigid-lid refusal and the barotropic
projector. All numbers are the measured probe values.
"""
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic import eigenmodes as eig
from fridom.hydrostatic.eigenmodes import (
    FAMILIES,
    FAMILY_NAMES,
    HydrostaticEigenmodes,
    barotropic_projector,
)
from fridom.model.errors import LinearOperatorGapError
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh

NX, NZ = 8, 6
N2, F0, CSQR, DEPTH = 4.0, 0.3, 400.0, 1.0
COMPONENTS = ("u", "v", "b", "ps")


# ================================================================
#  Model builders and module-scoped fixtures (compile reuse)
# ================================================================
def make_grid(nx, nz, depth=DEPTH):
    """Doubly-periodic x/y + bounded-z flat-bottom grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def make_model(grid, *, n2=N2, csqr=CSQR, f0=F0, free_surface=None):
    """Assemble the linear explicit hydrostatic model."""
    if free_surface is None:
        free_surface = hy.ExplicitFreeSurface()
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=csqr),
        time_stepper=AdamBashforth(1e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=f0),
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=free_surface,
        advection=False)


def k_disc_sq(n_mode, n_cells, length=1.0):
    """Discrete C-grid horizontal wavenumber squared of a mode."""
    dx = length / n_cells
    k = 2.0 * np.pi * n_mode / length
    return (2.0 * np.sin(k * dx / 2.0) / dx) ** 2


@pytest.fixture(scope="module")
def model():
    """One explicit hydrostatic model shared across the module."""
    return make_model(make_grid(NX, NZ))


@pytest.fixture(scope="module")
def em(model):
    """Build the labeled hydrostatic eigenmodes of the model."""
    return eig.from_model(model)


# ================================================================
#  (a) The labeled surface
# ================================================================
def test_from_model_builds_the_labeled_surface(model, em):
    assert isinstance(em, HydrostaticEigenmodes)
    assert em.components == COMPONENTS
    d = 3 * NZ + 1
    assert em.slices["u"] == slice(0, NZ)
    assert em.slices["v"] == slice(NZ, 2 * NZ)
    assert em.slices["b"] == slice(2 * NZ, 3 * NZ)
    assert em.slices["ps"] == slice(3 * NZ, d)
    assert em.periodic_axis == "y"
    assert em.bounded_axis == "z"
    assert em.omega.shape == (NX, NX // 2 + 1, d)
    assert em.q.shape == (NX, NX // 2 + 1, d, d)
    assert em.families == FAMILIES
    assert em.family_names == FAMILY_NAMES
    assert em.grid is model.grid


def test_metric_matches_the_h2_energy_metric(em):
    # u/v -> dz, b -> dz/N^2, ps -> H/c^2 (depth-integrated)
    metric = np.asarray(em.metric)
    dz = DEPTH / NZ
    assert np.allclose(metric[em.slices["u"]], dz, atol=1e-14)
    assert np.allclose(metric[em.slices["v"]], dz, atol=1e-14)
    assert np.allclose(metric[em.slices["b"]], dz / N2, atol=1e-14)
    assert np.allclose(metric[em.slices["ps"]], DEPTH / CSQR,
                       atol=1e-14)


def test_eigenbasis_is_biorthogonal_to_machine_precision(em):
    # measured: orthonormality 3.1e-15, hermiticity 1.7e-16
    # (L is exactly M-skew under the hydrostatic metric)
    assert float(em.basis.orthonormality_error()) < 1e-12
    assert em.basis.hermiticity_error < 1e-10


# ================================================================
#  (b) The per-plane spectrum split and the family census
# ================================================================
def test_spectrum_split_and_family_counts_at_kx1_ky0(em):
    omega = np.asarray(em.omega)[1, 0]
    labels = np.asarray(em.labels)[1, 0]
    assert int(np.sum(np.abs(omega) < 1e-8)) == NZ + 1
    assert int(np.sum(np.abs(omega) >= 1e-8)) == 2 * NZ
    counts = {name: int(np.sum(labels == code))
              for name, code in FAMILIES.items()}
    assert counts["barotropic_vortical"] == 1
    assert counts["barotropic_wave+"] == 1
    assert counts["barotropic_wave-"] == 1
    assert counts["baroclinic_vortical"] == NZ
    assert counts["baroclinic_wave+"] == NZ - 1
    assert counts["baroclinic_wave-"] == NZ - 1


def test_kh_zero_plane_carries_no_barotropic_wave(em):
    # the k_h = 0 waves are pure inertial oscillations carrying no
    # ps, so none clear ps_wave_tol -> all baroclinic; the single
    # free-surface geostrophic column still splits off barotropic
    labels = np.asarray(em.labels)[0, 0]
    assert (labels == FAMILIES["barotropic_wave+"]).sum() == 0
    assert (labels == FAMILIES["barotropic_wave-"]).sum() == 0
    assert (labels == FAMILIES["baroclinic_wave+"]).sum() == NZ
    assert (labels == FAMILIES["baroclinic_wave-"]).sum() == NZ
    assert (labels == FAMILIES["barotropic_vortical"]).sum() == 1


# ================================================================
#  (c) Dispersion: the discrete-symbol oracles
# ================================================================
def test_barotropic_poincare_dispersion(em):
    # the barotropic gravity branch matches the z-constant symbol
    # sqrt(f^2 + c^2 k_disc^2) to the documented tolerance (1.65e-3)
    omega = np.asarray(em.omega)
    labels = np.asarray(em.labels)
    (col,) = np.where(labels[1, 0] == FAMILIES["barotropic_wave+"])[0]
    got = abs(omega[1, 0, col])
    expect = np.sqrt(F0 ** 2 + CSQR * k_disc_sq(1, NX))
    assert abs(got - expect) / expect < 3e-3


def _gravest_baroclinic_m2(omega_plane, kx):
    """Discrete m^2 of the gravest baroclinic mode of a plane."""
    absom = np.sort(np.abs(omega_plane))
    absom = absom[absom > F0 + 1e-6]
    kh = np.sqrt(k_disc_sq(kx, NX))
    baroclinic = absom[absom < 0.5 * np.sqrt(CSQR) * kh]
    om1 = baroclinic.max()
    return N2 * k_disc_sq(kx, NX) / (om1 ** 2 - F0 ** 2)


def test_baroclinic_m_disc_is_consistent_across_kx():
    # the discrete vertical wavenumber m^2 = N^2 k_h^2 / (w^2 - f^2)
    # of the gravest baroclinic mode is kx-independent to machine
    # precision (measured 5e-15) and within ~5% of (pi/H)^2 (the
    # documented vertical-discretization tolerance)
    nz = 16
    em16 = eig.from_model(make_model(make_grid(NX, nz)))
    omega = np.asarray(em16.omega)
    m1 = _gravest_baroclinic_m2(omega[1, 0], 1)
    m2 = _gravest_baroclinic_m2(omega[2, 0], 2)
    assert abs(m1 - m2) / m1 < 1e-10
    m_cont = (np.pi / DEPTH) ** 2
    assert abs(m1 - m_cont) / m_cont < 0.06


# ================================================================
#  (d) The mode-indexed single-mode accessor
# ================================================================
def test_mode_accessor_barotropic_wave(em):
    omega, state = em.mode(
        "barotropic_wave", {"x": 1, "y": 0, "z": 0}, branch=1)
    expect = np.sqrt(F0 ** 2 + CSQR * k_disc_sq(1, NX))
    assert abs(omega - expect) / expect < 3e-3
    assert isinstance(state, hy.State)
    assert state.component_names == COMPONENTS
    for name in COMPONENTS:
        assert state[name].function_space.bare == em.spaces[name]


def test_mode_accessor_baroclinic_wave_and_vortical(em):
    omega_bc, st_bc = em.mode(
        "baroclinic_wave", {"x": 1, "y": 0, "z": 0}, branch=1)
    assert 0.0 < abs(omega_bc) < 10.0
    assert isinstance(st_bc, hy.State)
    for family in ("barotropic_vortical", "baroclinic_vortical"):
        omega0, st0 = em.mode(family, {"x": 1, "y": 0, "z": 0})
        assert abs(omega0) < 1e-8
        assert isinstance(st0, hy.State)
        assert st0.component_names == COMPONENTS


# ================================================================
#  (e) The implicit / rigid-lid refusal
# ================================================================
def test_implicit_and_rigid_lid_are_refused():
    grid = make_grid(NX, NZ)
    for free_surface in (hy.ImplicitFreeSurface(),
                         hy.ImplicitFreeSurface(epsilon=0.0)):
        model = make_model(grid, free_surface=free_surface)
        with pytest.raises(LinearOperatorGapError,
                           match="linear operator"):
            eig.from_model(model)


def test_explicit_free_surface_is_accepted():
    grid = make_grid(NX, NZ)
    model = make_model(grid, free_surface=hy.ExplicitFreeSurface())
    assert isinstance(eig.from_model(model), HydrostaticEigenmodes)


# ================================================================
#  (f) The barotropic projector helper and the labeler tolerances
# ================================================================
def test_barotropic_projector_is_m_self_adjoint_and_idempotent(em):
    metric = np.asarray(em.metric)
    proj = barotropic_projector(metric, em.slices)
    dim = metric.shape[0]
    assert proj.shape == (dim, dim)
    # M-self-adjoint: diag(metric) @ P is symmetric
    weighted = metric[:, None] * proj
    assert np.abs(weighted - weighted.T).max() < 1e-14
    # idempotent (the measure-weighted mean projector on each segment)
    assert np.abs(proj @ proj - proj).max() < 1e-12
    # b is purely baroclinic; ps is the identity slot
    b_seg = em.slices["b"]
    assert np.abs(proj[b_seg, b_seg]).max() == 0.0
    ps_seg = em.slices["ps"]
    assert proj[ps_seg, ps_seg][0, 0] == 1.0


def test_labeler_tolerances_plumb_through(model):
    em = HydrostaticEigenmodes(
        model, zero_tol=1e-9, ps_wave_tol=5e-3, barotropic_tol=0.4)
    assert isinstance(em, HydrostaticEigenmodes)
    labels = np.asarray(em.labels)
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2, 3, 4, 5]
