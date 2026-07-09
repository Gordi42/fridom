"""Shallow-water eigenmode projections as StateTransforms (wave 7 C).

Validates the staggered ``sw.transforms`` projections and the shared
``EigenProjection`` / ``ProjectionFactory`` base: idempotency, the
``WaveProjection = P(+1) + P(-1)`` algebra identity, and partition of
unity ``Vortical + Wave + Divergence == Identity`` on the model's
staggered state. The discrete ``{vortical, +gravity, -gravity}``
basis is complete per wavenumber (three modes span the three
components, the patched ``k = 0`` inertial triple included) except
the interpolation-Nyquist planes, where the geostrophic column is a
structural zero — so ``DivergenceProjection`` is the zero map on
Nyquist-free (band-limited) states and picks up exactly the
Nyquist-vortical residual otherwise.
"""
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw
from fridom.framework2.transforms.errors import SignatureMismatchError
from fridom.framework2.transforms.projection import EigenProjection

from .conftest import N, make_grid, make_model

COMPONENTS = ("u", "v", "p")


def _eig(f0=1.0, csqr=1.0):
    model = make_model(csqr=csqr, f0=f0)
    return sw.eigenmodes.from_model(model), model


def _state(model, *, seed=None):
    """Build a staggered ``(u, v, p)`` probe state on the model.

    Band-limited (Nyquist-free) trig fields by default so the
    three-mode basis is complete on the probe; ``seed`` switches to
    random data (which carries Nyquist-vortical content).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        shape = np.asarray(model.state["u"].data).shape
        model.set_fields(**{
            c: rng.standard_normal(shape) for c in COMPONENTS})
    else:
        x = (np.arange(N) + 0.5) / N
        gx, gy = np.meshgrid(x, x, indexing="ij")
        model.set_fields(
            u=np.sin(2 * np.pi * gx) * np.cos(2 * np.pi * gy),
            v=0.3 * np.cos(2 * np.pi * gx)
            - 0.2 * np.sin(2 * np.pi * gy),
            p=0.1 * np.sin(2 * np.pi * (gx + gy)))
    return sw.State({c: model.state[c] for c in COMPONENTS})


def _absmax(a, b):
    return max(
        float(np.abs(np.asarray(a[c].data) - np.asarray(b[c].data)).max())
        for c in COMPONENTS)


# ================================================================
#  Idempotency
# ================================================================
def test_vortical_and_wave_are_idempotent():
    em, model = _eig()
    z = _state(model, seed=1)
    fr.transforms.assert_idempotent(sw.transforms.VorticalProjection(em), z)
    fr.transforms.assert_idempotent(sw.transforms.WaveProjection(em), z)


def test_divergence_is_the_zero_map_on_band_limited_states():
    # off the Nyquist planes the three-mode basis is complete, so the
    # residual vanishes on a band-limited probe.
    em, model = _eig()
    z = _state(model)
    div = sw.transforms.DivergenceProjection(em)
    once = div(z)
    assert max(float(np.abs(np.asarray(once[c].data)).max())
               for c in COMPONENTS) < 1e-10
    # trivially idempotent (relative_l2 is ill-defined at zero, so an
    # absolute distance is used here)
    fr.transforms.assert_idempotent(div, z, norm=_absmax)


def test_divergence_captures_the_nyquist_vortical_residual():
    # random data carries interpolation-Nyquist vortical content the
    # discrete mode family structurally drops; the residual picks it
    # up (and stays idempotent).
    em, model = _eig()
    z = _state(model, seed=7)
    div = sw.transforms.DivergenceProjection(em)
    once = div(z)
    assert max(float(np.abs(np.asarray(once[c].data)).max())
               for c in ("u", "v")) > 1e-3
    fr.transforms.assert_idempotent(div, z)


# ================================================================
#  The algebra: Wave == P(+1) + P(-1); partition of unity
# ================================================================
def test_wave_equals_sum_of_single_mode_projections():
    em, model = _eig()
    z = _state(model, seed=2)
    wave = sw.transforms.WaveProjection(em)
    manual = (sw.transforms.mode_projection(em, 1)
              + sw.transforms.mode_projection(em, -1))
    assert _absmax(wave(z), manual(z)) < 1e-12


def test_partition_of_unity_reconstructs_the_state():
    # V + W + D == I EXACTLY on any state (D is the complement), the
    # patched k = 0 triple and the Nyquist planes included.
    em, model = _eig(f0=0.7, csqr=2.0)
    z = _state(model, seed=4)
    partition = (sw.transforms.VorticalProjection(em)
                 + sw.transforms.WaveProjection(em)
                 + sw.transforms.DivergenceProjection(em))
    assert _absmax(partition(z), z) < 1e-12


def test_wave_projection_merges_into_one_idempotent_projection():
    # P(+1) + P(-1) merges (same eigenmodes) into a single idempotent
    # EigenProjection over both modes, not a generic Sum node.
    em, _ = _eig()
    wave = sw.transforms.WaveProjection(em)
    assert isinstance(wave, EigenProjection)
    assert wave.modes == (-1, 1)
    assert wave.idempotent


# ================================================================
#  The shared base: signatures, dual constructors, repr
# ================================================================
def test_projection_has_a_concrete_staggered_signature():
    em, model = _eig()
    proj = sw.transforms.VorticalProjection(em)
    assert proj.domain is proj.codomain
    assert proj.domain.grid is model.grid
    assert proj.domain.names == ("u", "v", "p")
    assert proj.eigenmodes is em
    # the signature accepts the model's own staggered state
    proj.domain.validate_input(_state(model))


def test_call_rejects_a_state_missing_a_mapped_component():
    em, model = _eig()
    partial = sw.State({
        "u": model.state["u"],
        "v": model.state["v"]})
    with pytest.raises(SignatureMismatchError, match="p"):
        sw.transforms.VorticalProjection(em)(partial)


def test_from_model_builds_the_same_projection():
    _, model = _eig()
    proj = sw.transforms.WaveProjection.from_model(model)
    assert isinstance(proj, EigenProjection)
    assert proj.modes == (-1, 1)


def test_from_model_rejects_a_beta_plane():
    grid = make_grid()
    model = sw.Model(
        grid=grid, csqr=1.0,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=1.0, beta=2.0),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"coriolis\.f0"):
        sw.transforms.VorticalProjection.from_model(model)


def test_reprs_are_informative():
    em, _ = _eig()
    assert "VorticalProjection" in repr(sw.transforms.VorticalProjection(em))
    assert repr(sw.transforms.WaveProjection) == "WaveProjection"
    assert "modes=(0,)" in repr(sw.transforms.mode_projection(em, 0))


def test_projection_is_tier_one_and_costless():
    em, _ = _eig()
    proj = sw.transforms.WaveProjection(em)
    assert proj.traceable
    assert proj.cost().model_steps == 0
