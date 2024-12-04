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
