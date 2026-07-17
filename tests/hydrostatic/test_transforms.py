"""Hydrostatic eigenmode projections (``hy.transforms``, HY-D7).

The Tier-1 vortical / wave and barotropic / baroclinic projections as
composable ``fr.model.StateTransform``s, backed by the numeric
``HydrostaticEigenmodes``. Covers the four public dual-source factories,
their mode-code repr, the two orthogonal-and-complete partitions
(``Vortical + Wave == Barotropic + Baroclinic == identity`` on the
linear subspace), idempotency, cross-annihilation, the ``from_model``
dual source and the implicit refusal. All numbers are the measured probe
values (round-trip 1.45e-15, idempotency 1.3e-15, orthogonality
8.7e-16).
"""
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic import eigenmodes as eig
from fridom.hydrostatic import transforms as tr
from fridom.model._eigenbasis import family_projection
from fridom.model.errors import LinearOperatorGapError

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
    return hy.Model(
        grid=grid, dt=1e-3, csqr=csqr, free_surface=free_surface,
        stratification=hy.ConstantStratification(n2=n2),
        coriolis=hy.FPlaneCoriolis(f0=f0), advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-3, order=3))


def _state(model, seed=0):
    """Build a random physical ``(u, v, b, ps)`` probe state."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in COMPONENTS})
    return hy.State({c: model.state[c] for c in COMPONENTS})


def _add(a, b):
    """Component-wise sum of two states as a new hy.State."""
    return hy.State({c: a[c] + b[c] for c in COMPONENTS})


def _absmax(a, b):
    """Max component-wise absolute difference of two states."""
    return max(
        float(np.abs(np.asarray(a[c].data)
                     - np.asarray(b[c].data)).max())
        for c in COMPONENTS)


def _scale(state):
    """Max component-wise magnitude of a state."""
    return max(float(np.abs(np.asarray(state[c].data)).max())
               for c in COMPONENTS)


@pytest.fixture(scope="module")
def model():
    """One explicit hydrostatic model shared across the module."""
    return make_model(make_grid(NX, NZ))


@pytest.fixture(scope="module")
def em(model):
    """Build the labeled hydrostatic eigenmodes of the model."""
    return eig.from_model(model)


# ================================================================
#  The four factories and their mode-code repr
# ================================================================
def test_projections_build_with_the_expected_mode_codes(em):
    assert "modes=(0, 3)" in repr(tr.VorticalProjection(em))
    assert "modes=(1, 2, 4, 5)" in repr(tr.WaveProjection(em))
    assert "modes=(0, 1, 2)" in repr(tr.BarotropicProjection(em))
    assert "modes=(3, 4, 5)" in repr(tr.BaroclinicProjection(em))


# ================================================================
#  The two orthogonal, complete partitions
# ================================================================
def test_vortical_plus_wave_reconstructs_the_state(model, em):
    z = _state(model, seed=0)
    recon = _add(tr.VorticalProjection(em)(z), tr.WaveProjection(em)(z))
    assert _absmax(recon, z) < 1e-12 * _scale(z)


def test_barotropic_plus_baroclinic_reconstructs_the_state(model, em):
    z = _state(model, seed=1)
    recon = _add(tr.BarotropicProjection(em)(z),
                 tr.BaroclinicProjection(em)(z))
    assert _absmax(recon, z) < 1e-12 * _scale(z)


def test_projections_are_idempotent(model, em):
    z = _state(model, seed=2)
    for factory in (tr.VorticalProjection, tr.WaveProjection,
                    tr.BarotropicProjection, tr.BaroclinicProjection):
        proj = factory(em)
        once = proj(z)
        twice = proj(once)
        assert _absmax(twice, once) < 1e-12 * (_scale(once) + 1e-30)


def test_partitions_are_mutually_orthogonal(model, em):
    z = _state(model, seed=3)
    vort = tr.VorticalProjection(em)(z)
    wave_of_vort = tr.WaveProjection(em)(vort)
    assert _scale(wave_of_vort) < 1e-12 * (_scale(vort) + 1e-30)
    baro = tr.BarotropicProjection(em)(z)
    bcl_of_baro = tr.BaroclinicProjection(em)(baro)
    assert _scale(bcl_of_baro) < 1e-12 * (_scale(baro) + 1e-30)


# ================================================================
#  The dual source and the string-selection path
# ================================================================
def test_from_model_and_explicit_agree(model, em):
    z = _state(model, seed=4)
    from_model = tr.VorticalProjection.from_model(model)
    explicit = tr.VorticalProjection(em)
    assert from_model.modes == explicit.modes == (0, 3)
    assert _absmax(from_model(z), explicit(z)) < 1e-12 * _scale(z)


def test_projector_string_equals_the_barotropic_wave_subprojection(
        model, em):
    # the base-class string selection "barotropic_wave" (both signed
    # branches) equals the family_projection the factories compose
    z = _state(model, seed=5)
    via_string = em.projector("barotropic_wave")(z)
    via_family = family_projection(em, "barotropic_wave")(z)
    assert _absmax(via_string, via_family) < 1e-12 * (_scale(z) + 1e-30)


# ================================================================
#  The implicit refusal through the factory
# ================================================================
def test_implicit_free_surface_refused_via_factory():
    model = make_model(make_grid(NX, NZ),
                       free_surface=hy.ImplicitFreeSurface())
    with pytest.raises(LinearOperatorGapError,
                       match="linear operator"):
        tr.VorticalProjection.from_model(model)
