"""Test the FieldBase class."""
import pytest
import numpy as np

import fridom.framework as fr

# ================================================================
#  Create a test field class
# ================================================================

class Field(fr.FieldBase):
    def __init__(self,
                 mset,
                 value = 0,
                 is_spectral = False) -> None:  # noqa: FBT002
        super().__init__(mset)
        self.value = value
        self._is_spectral = is_spectral

    def fft(self, padding=None):
        super()._fft_possible()
        _ = padding
        return Field(self.mset, self.value, is_spectral=True)

    def ifft(self, padding=None):
        super()._ifft_possible()
        _ = padding
        return Field(self.mset, self.value, is_spectral=False)

    @property
    def is_spectral(self):
        return self._is_spectral

    @property
    def info(self) -> dict:
        return {"value": self.value, "is_spectral": self.is_spectral}

    def conj(self):
        conj_value = self.value.conjugate()
        return Field(self.mset, conj_value, is_spectral=self.is_spectral)

    def dot(self, other):
        return self * other.conj()

    @staticmethod
    def _apply_operation(op, field, other):
        if isinstance(other, Field):
            res_value = op(field.value, other.value)
        else:
            res_value = op(field.value, other)
        return Field(field.mset, res_value, is_spectral=field.is_spectral)

# ================================================================
#  Fixtures
# ================================================================

@pytest.fixture
def grid():
    # the dimension etc. of the grid is not important for this test
    return fr.grid.cartesian.Grid(N=(1, ), L=(1, ))

@pytest.fixture
def mset(grid):
    mset = fr.ModelSettingsBase(grid=grid)
    mset.setup()
    return mset

@pytest.fixture(params=[1.0, -2.0, 1.0 + 1.0j])
def value(request):
    return request.param

@pytest.fixture(params=[True, False])
def is_spectral(request):
    return request.param

@pytest.fixture
def field(mset, value, is_spectral):
    return Field(mset, value, is_spectral)

@pytest.fixture(params=[
    1.0,
     -1.0 + 3.0j,
     Field(mset, 1.0),
     Field(mset, 1.0 + 1.0j, is_spectral=True),
     np.array([1.0, 2.0, 3.0]),
])
def other(request):
    return request.param

# # ================================================================
#  Tests
# ================================================================

def test_init(mset):
    field = Field(mset)
    assert field.mset is mset
    assert field.value == 0
    assert not field.is_spectral

def test_properties(field, value, is_spectral, mset):
    assert field.mset is mset
    assert field.grid is mset.grid
    assert field.value == value
    assert field.is_spectral == is_spectral
    # check that the mset and grid properties are read-only
    with pytest.raises(AttributeError):
        field.mset = None
    with pytest.raises(AttributeError):
        field.grid = None

@pytest.mark.parametrize("fft_available", [True, False])
def test_unavailable_fft_on_grid(mset, fft_available):
    mset.grid.fourier_transform_available = fft_available
    field = Field(mset, is_spectral=False)
    msg = "Fourier transform not available for this grid"
    if not fft_available:
        with pytest.raises(NotImplementedError, match=msg):
            field.fft()
    else:
        field.fft()

    field = Field(mset, is_spectral=True)
    if not fft_available:
        with pytest.raises(NotImplementedError, match=msg):
            field.ifft()
    else:
        field.ifft()

def test_fft(mset, is_spectral):
    field = Field(mset, is_spectral=is_spectral)
    if is_spectral:
        msg = "Field is in spectral space, cannot perform fft"
        with pytest.raises(ValueError, match=msg):
            field.fft()
        return
    new_field = field.fft()
    assert new_field.mset is field.mset
    assert new_field.value == field.value
    assert new_field.is_spectral

def test_ifft(mset, is_spectral):
    field = Field(mset, is_spectral=is_spectral)
    if not is_spectral:
        msg = "Field is not in spectral space, cannot perform ifft"
        msg = "Field is not in spectral space, cannot perform ifft"
        with pytest.raises(ValueError, match=msg):
            field.ifft()
        return
    new_field = field.ifft()
    assert new_field.mset is field.mset
    assert new_field.value == field.value
    assert not new_field.is_spectral

def test_repf(mset):
    field = Field(mset)
    exp = "Field(\n  value=0, \n  is_spectral=False, \n)"
    assert repr(field) == exp

def test_info(field, value, is_spectral):
    info = field.info
    assert info == {"value": value, "is_spectral": is_spectral}

def test_conj(field, value):
    new_field = field.conj()
    assert new_field.mset is field.mset
    assert new_field.value == value.conjugate()
    assert new_field.is_spectral == field.is_spectral

@pytest.mark.parametrize("dot_op", 
    [
        pytest.param(lambda x, y: x.dot(y), id="dot method"),
        pytest.param(lambda x, y: x @ y, id="matmul operator"),
    ],
)
@pytest.mark.parametrize( "other_value", [1, -1, 1j])
def test_dot(mset, value, other_value, dot_op):
    field = Field(mset, value)
    other = Field(mset, other_value)
    res = dot_op(field, other)
    assert res.mset is field.mset
    assert res.value == value * (other_value.conjugate())

@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda x, y: x + y, id="add"),
        pytest.param(lambda x, y: x - y, id="sub"),
        pytest.param(lambda x, y: x * y, id="mul"),
        pytest.param(lambda x, y: x / y, id="div"),
        pytest.param(lambda x, y: x ** y, id="pow"),
    ],
)
def test_operator_overloads(field, other, op):
    # test the operator overload in the forward order
    res = op(field, other)
    assert res.mset is field.mset
    assert res.is_spectral == field.is_spectral
    other_value = other.value if isinstance(other, Field) else other
    if isinstance(other_value, np.ndarray):
        assert np.all(res.value == op(field.value, other_value))
    else:
        assert res.value == op(field.value, other_value)
    # let's also check the reverse order
    if isinstance(other, np.ndarray):
        # np.array(...) + Field(...) can not be done at the moment
        # the problem is that the numpy operation does not raise an error
        # but returns a numpy array, which is not what we want
        # hence the __radd__ method is not called. And we have no control
        # over this. So we skip the reverse order tests for numpy arrays
        return
    if isinstance(other, Field):
        # not really interesting to test the reverse order for two fields
        # so we skip this as well
        return
    res = op(other, field)
    # first check if the result is actually a Field
    assert isinstance(res, Field)
    # then check if the model settings are correct
    assert res.mset is field.mset
    # then check if the value is correct
    assert res.is_spectral == field.is_spectral
    assert res.value == op(other, field.value)

def test_neg(field):
    res = -field
    assert res.mset is field.mset
    assert res.is_spectral == field.is_spectral
    assert res.value == -field.value
