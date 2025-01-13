"""Test of framework/modules/restart_module.py."""
import numpy as np
import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(N=(16, ), L=(1, ))
    return fr.ModelSettingsBase(grid).setup()

# ================================================================
#  Tests
# ================================================================

@pytest.mark.parametrize("kwargs", [
    pytest.param({}, id="default"),
    pytest.param({"realtime_interval": np.timedelta64(1, "h")}, id="realtime_interval"),
    pytest.param({"clock_trigger": fr.ClockTrigger(step_size=100)}, id="clock_trigger"),
    pytest.param({"restart_command": "python model.py"}, id="restart_command"),
    pytest.param({"file_path": "res_dir/restart.dill"}, id="file_path"),
])
def test_init(mset, kwargs):
    """Test if the init method runs without errors."""
    restart_module = fr.modules.RestartModule(**kwargs)
    mset.restart_module = restart_module
    mset.setup()
    assert mset.restart_module is restart_module

#TODO(Silvano): Add a test for the should_restart method
#TODO(Silvano): Add a test for the should_reload method
#TODO(Silvano): Add a test with a model restart
