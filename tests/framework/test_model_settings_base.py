"""Tests for the ModelSettingsBase class."""
import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    return fr.ModelSettingsBase(grid=grid).setup()


# ================================================================
#  Tests
# ================================================================

def test_init(): ...

def test_attributes(): ...

def test_setup(): ...

def test_repr(): ...

def test_halo(): ...

def test_unknown_attribute_raises():
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    with pytest.raises(AttributeError, match="no attribute 'bogus'"):
        fr.ModelSettingsBase(grid=grid, bogus=42)

def test_time_stepper_setter_after_setup(mset):
    time_stepper = fr.time_steppers.RungeKutta(dt=1.0)
    mset.time_stepper = time_stepper
    assert mset.time_stepper is time_stepper
    assert time_stepper.is_setup

def test_progress_bar_setter_after_setup(mset):
    progress_bar = fr.modules.ProgressBar()
    mset.progress_bar = progress_bar
    assert mset.progress_bar is progress_bar
    assert progress_bar.is_setup

def test_nan_checker_setter_after_setup(mset):
    nan_checker = fr.modules.NaNChecker()
    mset.nan_checker = nan_checker
    assert mset.nan_checker is nan_checker
    assert nan_checker.is_setup

class HaloModule(fr.modules.Module):

    """A module that raises its halo requirement during the setup."""

    name = "Halo Module"

    def _on_setup(self) -> None:
        self.required_halo = 3

def test_tendencies_setter_updates_halo(mset):
    assert mset.grid.halo == 1

    # the halo requirement only becomes visible during the setup, which
    # forces a new grid setup
    tendencies = fr.modules.ModuleContainer(name="tendencies")
    tendencies.add_module(HaloModule())
    mset.tendencies = tendencies

    assert mset.tendencies is tendencies
    assert mset.grid.halo == 3

def test_diagnostics_setter(mset):
    diagnostics = fr.modules.ModuleContainer(name="diagnostics")
    mset.diagnostics = diagnostics
    assert mset.diagnostics is diagnostics

def test_pre_step_diagnostics_setter(mset):
    container = fr.modules.ModuleContainer(name="pre step")
    mset.pre_step_diagnostics = container
    assert mset.pre_step_diagnostics is container

def test_timer_setter(mset):
    timer = fr.timing_module.TimingModule()
    mset.timer = timer
    assert mset.timer is timer

def test_raise_error_setter(mset):
    mset.raise_error_when_something_goes_wrong = True
    assert mset.raise_error_when_something_goes_wrong

def test_doc_example():
    class MyModelSettings(fr.ModelSettingsBase):
        def __init__(self, grid, **kwargs):
            super().__init__(grid)
            self.model_name = "MyModel"
            self.my_parameter = 1.0
            self.set_attributes(**kwargs)
        def __str__(self) -> str:
            res = super().__str__()
            res += f"  My parameter: {self.my_parameter}\\n"
            return res

    mset = MyModelSettings(None, my_parameter=2.0)
    assert mset.model_name == "MyModel"
    assert mset.my_parameter == 2.0
    assert "My parameter: 2" in str(mset)
