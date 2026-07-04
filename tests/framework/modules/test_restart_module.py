"""Test of framework/modules/restart_module.py."""
import logging
import shutil
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def file_path(directory):
    yield Path(directory) / "restart.dill"
    shutil.rmtree(directory)

@pytest.fixture
def capture_logs():
    """Fixture to capture log output."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    fr.log.addHandler(handler)
    # other tests may have silenced the logger; the tests below rely on
    # NOTICE-level messages
    old_level = fr.log.level
    fr.log.setLevel("NOTICE")
    yield stream
    fr.log.setLevel(old_level)
    fr.log.removeHandler(handler)

@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(16, ), domain_size=(1, ))
    return fr.ModelSettingsBase(grid).setup()

# ================================================================
#  Tests
# ================================================================

@pytest.mark.parametrize("kwargs", [
    pytest.param({}, id="default"),
    pytest.param({"realtime_interval": np.timedelta64(1, "h")},
                 id="realtime_interval"),
    pytest.param({"clock_trigger": fr.ClockTrigger(step_size=100)},
                 id="clock_trigger"),
    pytest.param({"restart_command": "python model.py"}, id="restart_command"),
])
def test_init(mset, kwargs, file_path):
    """Test if the init method runs without errors."""
    restart_module = fr.modules.RestartModule(**kwargs, file_path=file_path)
    mset.restart_module = restart_module
    mset.setup()
    # check that the module is correctly set
    assert mset.restart_module is restart_module

@pytest.mark.parametrize(*(
    "kwargs, should_be_enabled",
    [
        pytest.param({},
                     False, id="default"),
        pytest.param({"realtime_interval": np.timedelta64(1, "h")},
                     True, id="realtime_interval"),
        pytest.param({"clock_trigger": fr.ClockTrigger(step_size=100)},
                     True, id="clock_trigger"),
        pytest.param({"restart_command": "python model.py"},
                     False, id="restart_command"),
    ],
))
def test_enabled_disabled(mset, kwargs, file_path, should_be_enabled):
    """Test if the module is disabled."""
    assert not mset.restart_module.is_enabled()
    mset.restart_module = fr.modules.RestartModule(**kwargs,
                                                   file_path=file_path)
    mset.setup()
    assert mset.restart_module.is_enabled() == should_be_enabled

@pytest.mark.parametrize(*(
    "realtime_interval, should_raise",
    [
        pytest.param(np.timedelta64(1, "h"), False, id="timedelta"),
        pytest.param(5, True, id="int"),
        pytest.param("5h", True, id="str"),
    ],
))
def test_wrong_realtime_interval(realtime_interval, file_path, should_raise):
    if not should_raise:
        fr.modules.RestartModule(realtime_interval=realtime_interval,
                                 file_path=file_path)
        return
    with pytest.raises(TypeError, match="The interval"):
        fr.modules.RestartModule(realtime_interval=realtime_interval,
                                file_path=file_path)

def test_restart_command(file_path):
    """Test the restart command."""
    command = "python model.py"
    mod = fr.modules.RestartModule(restart_command=command,
                                   file_path=file_path)
    assert mod.restart_command == command

def run_model(mset):
    """Run the model."""
    # Add a counting module to test the restart
    counter = fr.modules.Counter(clock_trigger=fr.ClockTrigger(step_size=1))
    mset.diagnostics.add_module(counter)
    mset.setup()
    # Run the model for 100 steps
    model = fr.Model(mset)

    # Set a trigger to restart the model
    def on_restart() -> None:
        # Set the model to panicked (to abort the run)
        model.model_state.panicked = True
        # run the model again
        run_model(mset)

    model.mset.restart_module.restart_command = on_restart
    model.run(steps=100)
    # print the counter at the end of the run
    if not model.model_state.panicked:
        fr.log.notice("Counter: %s",
                      model.diagnostics.get("Counter")[0].counter)

def test_restart_with_clock(capture_logs, mset, file_path):
    # Check that the restart directory is empty
    assert not list(file_path.parent.iterdir())

    # Create a restart module with a clock trigger
    restart_module = fr.modules.RestartModule(
        file_path=file_path, clock_trigger=fr.ClockTrigger(step_size=30),
    )
    mset.restart_module = restart_module
    mset.setup()
    # Run the model
    run_model(mset)
    # Check that restart files were created
    restart_files = ["restart_31_0.dill", "restart_61_0.dill",
                     "restart_91_0.dill"]
    assert set(restart_files) == {
        f.name for f in file_path.parent.iterdir()}
    # Check that the counter is printed
    logs = capture_logs.getvalue()
    assert "Counter: 100" in logs
