"""Shallow-water eigenmode projections as StateTransforms (wave 7 C).

Validates the collocated ``sw.transforms`` projections and the shared
``EigenProjection`` / ``ProjectionFactory`` base: idempotency, the
``WaveProjection = P(+1) + P(-1)`` algebra identity, and partition of
unity ``Vortical + Wave + Divergence == Identity`` on a collocated
state. The shallow-water ``{vortical, +gravity, -gravity}`` basis is
complete (three modes span the three components), so
``DivergenceProjection`` is the (trivially idempotent) zero map here.
"""
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw
from fridom.framework2.transforms.errors import SignatureMismatchError
from fridom.framework2.transforms.projection import EigenProjection

from .conftest import make_grid, make_model


def _eig(f0=1.0, csqr=1.0):
    model = make_model(csqr=csqr, f0=f0)
    return sw.eigenmodes.from_model(model), model


def _state(em, model):
    """Build a physical collocated (u, v, p) probe state."""
    grid = model.grid
    center = em.center_space
    u = grid.create_field(
        center, name="u",
        init=lambda x, y: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))
    v = grid.create_field(
        center, name="v",
        init=lambda x, y: 0.3 * np.cos(2 * np.pi * x)
        - 0.2 * np.sin(2 * np.pi * y))
    p = grid.create_field(
        center, name="p",
        init=lambda x, y: 0.1 * np.sin(2 * np.pi * (x + y)))
    return sw.State({"u": u, "v": v, "p": p})


def _absmax(a, b):
    return max(
        float(np.abs(np.asarray(a[c].data) - np.asarray(b[c].data)).max())
        for c in ("u", "v", "p"))


# ================================================================
#  Idempotency
# ================================================================
def test_vortical_and_wave_are_idempotent():
    em, model = _eig()
    z = _state(em, model)
    fr.transforms.assert_idempotent(sw.transforms.VorticalProjection(em), z)
    fr.transforms.assert_idempotent(sw.transforms.WaveProjection(em), z)


def test_divergence_is_the_zero_map_and_idempotent():
    # the SW eigenbasis is complete, so the residual vanishes.
    em, model = _eig()
    z = _state(em, model)
    div = sw.transforms.DivergenceProjection(em)
    once = div(z)
    assert max(float(np.abs(np.asarray(once[c].data)).max())
               for c in ("u", "v", "p")) < 1e-10
    # trivially idempotent (relative_l2 is ill-defined at zero, so an
    # absolute distance is used here)
    fr.transforms.assert_idempotent(div, z, norm=_absmax)


# ================================================================
#  The algebra: Wave == P(+1) + P(-1); partition of unity
# ================================================================
def test_wave_equals_sum_of_single_mode_projections():
    em, model = _eig()
    z = _state(em, model)
    wave = sw.transforms.WaveProjection(em)
    manual = (sw.transforms.mode_projection(em, 1)
              + sw.transforms.mode_projection(em, -1))
    assert _absmax(wave(z), manual(z)) < 1e-12


def test_partition_of_unity_reconstructs_the_state():
    em, model = _eig(f0=0.7, csqr=2.0)
    z = _state(em, model)
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
def test_projection_has_a_concrete_collocated_signature():
    em, model = _eig()
    proj = sw.transforms.VorticalProjection(em)
    assert proj.domain is proj.codomain
    assert proj.domain.grid is model.grid
    assert proj.domain.names == ("u", "v", "p")
    assert proj.eigenmodes is em


def test_call_rejects_a_state_missing_a_mapped_component():
    em, model = _eig()
    grid = model.grid
    center = em.center_space
    partial = sw.State({
        "u": grid.create_field(center, name="u"),
        "v": grid.create_field(center, name="v")})
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
