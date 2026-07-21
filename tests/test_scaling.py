"""The fr.scaling policy objects (src/fridom/scaling.py)."""
import dataclasses

import pytest

import fridom as fr

ALL = (fr.scaling.Dimensional, fr.scaling.Advective,
       fr.scaling.Rotational, fr.scaling.GravityWave,
       fr.scaling.InternalWave, fr.scaling.ExternalWave)

TRAITS = {
    fr.scaling.Dimensional: (False, None),
    fr.scaling.Advective: (True, None),
    fr.scaling.Rotational: (True, "rotation"),
    fr.scaling.GravityWave: (True, "gravity_wave"),
    fr.scaling.InternalWave: (True, "internal_wave"),
    fr.scaling.ExternalWave: (True, "external_wave"),
}


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_traits(cls):
    scaling = cls()
    nondim, mechanism = TRAITS[cls]
    assert scaling.nondimensional is nondim
    assert scaling.mechanism == mechanism


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_frozen_host_objects(cls):
    scaling = cls()
    assert dataclasses.is_dataclass(scaling)
    with pytest.raises(dataclasses.FrozenInstanceError):
        scaling.L = 1.0
    # host-side policy objects: never jaxified (no dynamic leaves)
    assert not hasattr(scaling, "dynamic_jax_attrs")


def test_reference_scales_are_stored_only():
    scaling = fr.scaling.GravityWave(L=1e5, U=0.1, g=9.81)
    assert scaling.L == 1e5
    assert scaling.U == 0.1
    assert scaling.g == 9.81
    # defaults: None (no reference scales named)
    bare = fr.scaling.Dimensional()
    assert (bare.L, bare.U, bare.g) == (None, None, None)


def test_equality_is_by_value():
    assert fr.scaling.GravityWave() == fr.scaling.GravityWave()
    assert fr.scaling.GravityWave(L=1.0) != fr.scaling.GravityWave()
    assert fr.scaling.GravityWave() != fr.scaling.Rotational()
