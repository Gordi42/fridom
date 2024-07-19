import pytest
from copy import deepcopy
from fridom.framework import config
import fridom.framework as fr

@pytest.fixture(params=[2, 3], ids=["2D", "3D"])
def n_dims(request):
    return request.param

@pytest.fixture(params=[True, False], ids=["Spectral", "Physical"])
def is_spectral(request):
    return request.param

@pytest.fixture()
def dtype_in(is_spectral):
    return config.dtype_comp if is_spectral else config.dtype_real

@pytest.fixture()
def dtype_out(is_spectral):
    return config.dtype_real if is_spectral else config.dtype_comp

@pytest.fixture(params=[3, 5], ids=["n=3", "n=5"])
def n_fields(request):
    return request.param

@pytest.fixture()
def mset(backend, n_dims):
    grid = fr.grid.cartesian.Grid(N=tuple([32]*n_dims), L=tuple([1.0]*n_dims), 
                                 shared_axes=[0])
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset

@pytest.fixture()
def position(n_dims):
    return fr.grid.cartesian.Position(
        tuple([fr.grid.cartesian.AxisOffset.CENTER]*n_dims))

@pytest.fixture()
def field_list(mset, is_spectral, n_fields, position):
    field_list = [fr.FieldVariable(
        mset, name=f"v{i}", is_spectral=is_spectral, position=position) 
                  for i in range(n_fields)]
    for field in field_list:
        field[:] = config.ncp.random.rand(*field.shape)
    return field_list

@pytest.fixture()
def state(mset, field_list, is_spectral):
    return fr.StateBase(mset, field_list, is_spectral=is_spectral)

@pytest.fixture()
def mset_1d(backend):
    grid = fr.grid.cartesian.Grid(N=(3,), L=(1.0,), shared_axes=[0])
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset

@pytest.fixture()
def position_1d():
    return fr.grid.cartesian.Position((fr.grid.cartesian.AxisOffset.CENTER,))

@pytest.fixture()
def zeros_p(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=False, name="zeros_p", position=position_1d)

@pytest.fixture()
def zeros_s(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=True, name="zeros_s", position=position_1d)

@pytest.fixture()
def ones_p(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=False, name="ones_p", position=position_1d) + 1.0

@pytest.fixture()
def ones_s(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=True, name="ones_s", position=position_1d) + 1.0

@pytest.fixture()
def imag_s(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=True, name="imag_s", position=position_1d) + 1.0j

@pytest.fixture()
def state_01(mset_1d, position_1d):
    v1 = fr.FieldVariable(
        mset_1d, is_spectral=False, name="v1", position=position_1d)
    v2 = fr.FieldVariable(
        mset_1d, is_spectral=False, name="v2", position=position_1d) + 1.0
    return fr.StateBase(mset_1d, [v1, v2], is_spectral=False)

@pytest.fixture()
def state_11(mset_1d, position_1d):
    v1 = fr.FieldVariable(
        mset_1d, is_spectral=False, name="v1", position=position_1d) + 1.0
    v2 = fr.FieldVariable(
        mset_1d, is_spectral=False, name="v2", position=position_1d) + 1.0
    return fr.StateBase(mset_1d, [v1, v2], is_spectral=False)

def test_init(mset, field_list, state):
    assert state.mset is mset
    assert state.grid is mset.grid
    field_dict = {f.name: f for f in field_list}
    assert field_dict == field_dict

def test_copy(state):
    state_copy = deepcopy(state)
    assert state_copy is not state
    for f1, f2 in zip(state_copy.field_list, state.field_list):
        assert f1 is not f2

def test_fft(state, dtype_in, dtype_out):
    state_fft = state.fft()
    assert state.is_spectral != state_fft.is_spectral
    assert state_fft.field_list[0].dtype == dtype_out
    state_fft_fft = state_fft.fft()
    assert state_fft_fft.is_spectral == state.is_spectral
    assert state_fft_fft.field_list[0].dtype == dtype_in
    # Check that the data is the same
    # Don't check if the state is spectral because the data will be different
    if not state.is_spectral: 
        for f1, f2 in zip(state_fft_fft.field_list, state.field_list):
            assert config.ncp.allclose(f1, f2)

@pytest.mark.mpi_skip
def test_dot(mset_1d, ones_p, ones_s, state_01, state_11, position_1d):
    state = state_01; state2 = state_11
    dot = state.dot(state2)
    assert isinstance(dot, fr.FieldVariable)
    assert config.ncp.allclose(dot, ones_p)

    # test complex
    v1 = fr.FieldVariable(
        mset_1d, is_spectral=True, name="v1", position=position_1d) + 1
    v2 = fr.FieldVariable(
        mset_1d, is_spectral=True, name="v2", position=position_1d) + 1 - 1j
    state = fr.StateBase(mset_1d, [v1, v2], is_spectral=True)
    v1 = fr.FieldVariable(
        mset_1d, is_spectral=True, name="v1", position=position_1d) + 1j
    v2 = fr.FieldVariable(
        mset_1d, is_spectral=True, name="v2", position=position_1d) + 1 
    state2 = fr.StateBase(mset_1d, [v1, v2], is_spectral=True)
    dot = state.dot(state2)
    assert config.ncp.allclose(dot, ones_s-2j)

@pytest.mark.mpi_skip
def test_norm_l2(mset_1d, state_01, state_11, ones_p):
    state = state_01; state2 = state_11
    # yields state = [(0, 1), (0, 1), (0, 1)]
    # with l2 norm = sqrt((1+1+1) * 1/3) = 1
    #                               ^^^
    #                               dV
    norm = state.norm_l2()
    assert config.ncp.allclose(norm, 1.0)

    # yields state = [(1, 1), (1, 1), (1, 1)]
    # with l2 norm = sqrt((2+2+2) * 1/3) = sqrt(2)
    norm = state2.norm_l2()
    assert config.ncp.allclose(norm, 2**0.5)

@pytest.mark.mpi_skip
def test_norm_of_diff(mset_1d, state_01, state_11):
    # test norm of difference between two identical states
    # should be 0
    norm = state_01.norm_of_diff(state_01)
    assert norm == 0

    # test norm of difference between two different states
    # the l2 norm of state - state2 is:
    # sqrt((1+1+1) * 1/3) = 1
    assert (state_01 - state_11).norm_l2() == 1

    # hence, the norm of the difference should be
    # 2 * |z - z'|_2 / (|z|_2 + |z'|_2)
    # 2 *    1       / ( 1    + 2**(1/2))
    norm = state_01.norm_of_diff(state_11)
    assert norm == 2 / (1 + 2**0.5)

@pytest.mark.mpi_skip
def test_add(state_01, state_11, zeros_p, ones_p):
    state = state_01; state2 = state_11

    # add two states
    state3 = state + state2
    assert config.ncp.allclose(state.field_list[0], zeros_p)
    assert config.ncp.allclose(state.field_list[1], ones_p)
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], ones_p)
    assert config.ncp.allclose(state3.field_list[1], ones_p + 1)

    # add state and scalar
    state3 = state + 1
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], ones_p)
    assert config.ncp.allclose(state3.field_list[1], ones_p + 1)

    # add state and array
    state3 = state + ones_p
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], ones_p)
    assert config.ncp.allclose(state3.field_list[1], ones_p + 1)

    # add number and state
    state3 = 1 + state
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], ones_p)
    assert config.ncp.allclose(state3.field_list[1], ones_p + 1)

@pytest.mark.mpi_skip
def test_sub(ones_p, state_01, state_11):
    state = state_01; state2 = state_11

    # subtract two states
    state3 = state - state2
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], -ones_p[:])
    assert config.ncp.allclose(state3.field_list[1], 0)

    # subtract state and scalar
    state3 = state - 1
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], -ones_p[:])
    assert config.ncp.allclose(state3.field_list[1], 0)

    # subtract state and array
    state3 = state - ones_p
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], -ones_p[:])
    assert config.ncp.allclose(state3.field_list[1], 0)

    # subtract number and state
    state3 = 1 - state
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], ones_p)
    assert config.ncp.allclose(state3.field_list[1], 0)

@pytest.mark.mpi_skip
def test_mul(zeros_p, ones_p, state_01, state_11):
    state = state_01; state2 = state_11

    # multiply two states
    state3 = state * state2
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], zeros_p)
    assert config.ncp.allclose(state3.field_list[1], ones_p)

    # multiply state and scalar
    state3 = state * 2
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], zeros_p)
    assert config.ncp.allclose(state3.field_list[1], 2)

    # multiply state and array
    state3 = state * ones_p
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], zeros_p)
    assert config.ncp.allclose(state3.field_list[1], ones_p)

    # multiply number and state
    state3 = 2 * state
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], zeros_p)
    assert config.ncp.allclose(state3.field_list[1], 2)

@pytest.mark.mpi_skip
def test_truediv(zeros_p, ones_p, state_01, state_11):
    state = state_01; state2 = state_11

    # divide two states
    state3 = state / state2
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], zeros_p)
    assert config.ncp.allclose(state3.field_list[1], ones_p)

    # divide state and scalar
    state3 = state / 2
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], zeros_p)
    assert config.ncp.allclose(state3.field_list[1], 0.5)

    # divide state and array
    state3 = state / ones_p
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], zeros_p)
    assert config.ncp.allclose(state3.field_list[1], 1)

    # divide number and state
    state3 = 2 / state2
    assert isinstance(state3, fr.StateBase)
    assert config.ncp.allclose(state3.field_list[0], 2)
    assert config.ncp.allclose(state3.field_list[1], 2)