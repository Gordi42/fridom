"""tests for exceptions.py ."""

import pytest

import fridom.framework as fr


@pytest.mark.parametrize(*(
    "max_args, provided_args, raised",
    [
        pytest.param(
            2, {"arg1": 1, "arg2": 2, "arg3": 3}, True, id="too_many_args"),
        pytest.param(
            3, {"arg1": 1, "arg2": 2, "arg3": 3}, False, id="not_too_many_args"),
        pytest.param(
            2, {"arg1": 1, "arg2": 2, "arg3": None}, False, id="arg_with_none"),
    ],
))
def test_too_many_arguments_error(max_args, provided_args, raised):
    if not raised:
        fr.exceptions.TooManyArgumentsError.check(max_args, **provided_args)
        return
    with pytest.raises(fr.exceptions.TooManyArgumentsError):
        fr.exceptions.TooManyArgumentsError.check(max_args, **provided_args)

@pytest.mark.parametrize(*(
    "topo, raised",
    [
        pytest.param((True, True, True), False, id="full_domain"),
        pytest.param((True, True, False), True, id="lacking_z"),
        pytest.param((True, False, True), True, id="lacking_y"),
        pytest.param((False, True, True), True, id="lacking_x"),
        pytest.param((False, False, False), True, id="no_domain"),

    ],
))
def test_partial_domain_error(topo, raised):
    grid = fr.grid.cartesian.Grid(N=(3, 3, 3), L=(1, 1, 1))
    mset = fr.ModelSettingsBase(grid).setup()
    field = fr.ScalarField(mset, topo=topo)
    if not raised:
        fr.exceptions.PartialDomainError.check(field)
        return

    msg = "Operation not available for fields with topo="
    with pytest.raises(fr.exceptions.PartialDomainError, match=msg):
        fr.exceptions.PartialDomainError.check(field)

def test_field_space_error():
    grid = fr.grid.cartesian.Grid(N=(3, 3, 3), L=(1, 1, 1))
    mset = fr.ModelSettingsBase(grid).setup()
    field = fr.ScalarField(mset, is_spectral=True)
    # Check if the field is in spectral space should pass
    fr.exceptions.FieldSpaceError.check_if_spectral(field)

    # Check if the field is in physical space should raise an error
    msg = "Operation not available for fields in spectral space."
    with pytest.raises(fr.exceptions.FieldSpaceError, match=msg):
        fr.exceptions.FieldSpaceError.check_if_physical(field)

    # now the same for physical space
    field = fr.ScalarField(mset, is_spectral=False)
    # Check if the field is in physical space should pass
    fr.exceptions.FieldSpaceError.check_if_physical(field)

    # Check if the field is in spectral space should raise an error
    msg = "Operation not available for fields in physical space."
    with pytest.raises(fr.exceptions.FieldSpaceError, match=msg):
        fr.exceptions.FieldSpaceError.check_if_spectral(field)
