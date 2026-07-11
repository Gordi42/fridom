"""Test of framework/modules/restart_module.py."""
import logging
import shutil
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import fridom.framework as fr
import fridom.framework.modules.restart_module as restart_module_module


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

def test_should_restart_realtime(mset, file_path):
    # a zero realtime interval triggers a restart immediately
    module = fr.modules.RestartModule(
        realtime_interval=np.timedelta64(0, "s"), file_path=file_path)
    module.setup(mset=mset)

    mz = fr.ModelState(mset)
    assert module.should_restart(mz=mz)
    assert module.file is not None


def test_should_not_restart(mset, file_path):
    module = fr.modules.RestartModule(
        realtime_interval=np.timedelta64(1000, "h"), file_path=file_path)
    module.setup(mset=mset)

    mz = fr.ModelState(mset)
    assert not module.should_restart(mz=mz)


@pytest.mark.parametrize("kwargs", [
    pytest.param({"realtime_interval": np.timedelta64(1, "h")},
                 id="realtime"),
    pytest.param({"clock_trigger": fr.ClockTrigger(step_size=100)},
                 id="clock-trigger"),
])
def test_reset(mset, file_path, kwargs):
    module = fr.modules.RestartModule(**kwargs, file_path=file_path)
    module.setup(mset=mset)
    module.reset()


def make_model_mock():
    model = MagicMock()
    model.model_state.clock.it = 3
    model.model_state.clock.time = 1.0
    return model


@pytest.mark.parametrize("stderr", ["error", ""])
def test_restart_with_string_command(mset, file_path, monkeypatch, stderr):
    module = fr.modules.RestartModule(
        realtime_interval=np.timedelta64(0, "s"), file_path=file_path,
        restart_command="sbatch job.sh")
    module.setup(mset=mset)
    module.set_full_filename(3)

    # pretend to be an mpi run so that the restart command is executed
    monkeypatch.setattr(restart_module_module.fr.utils,
                        "MPI_AVAILABLE", True)
    run_mock = MagicMock(
        return_value=MagicMock(stdout="submitted", stderr=stderr))
    monkeypatch.setattr(subprocess, "run", run_mock)

    model = make_model_mock()
    with pytest.raises(SystemExit):
        module.restart(model)

    assert model.stop.called
    assert model.save.called
    assert run_mock.call_args.args[0] == ["sbatch", "job.sh"]


def test_restart_from_command_without_mpi(mset, file_path, monkeypatch):
    # without mpi, no subprocess is spawned and the process just exits
    module = fr.modules.RestartModule(
        realtime_interval=np.timedelta64(0, "s"), file_path=file_path)
    module.setup(mset=mset)

    monkeypatch.setattr(restart_module_module.fr.utils,
                        "MPI_AVAILABLE", False)
    run_mock = MagicMock()
    monkeypatch.setattr(subprocess, "run", run_mock)

    with pytest.raises(SystemExit):
        module._restart_from_command()

    assert not run_mock.called


def test_restart_without_command_raises(mset, file_path, monkeypatch):
    # ensure no slurm job id leaks in from the environment (e.g. when the
    # suite runs inside a slurm allocation), which would auto-detect a
    # restart command and take the command branch instead of raising
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    module = fr.modules.RestartModule(
        realtime_interval=np.timedelta64(0, "s"), file_path=file_path)
    module.setup(mset=mset)
    module.set_full_filename(3)

    with pytest.raises(ValueError, match="No restart command is set"):
        module.restart(make_model_mock())


def test_info(file_path):
    # a disabled module only shows the base info
    module = fr.modules.RestartModule(file_path=file_path)
    assert "File Path" not in module.info

    # an enabled module shows the restart configuration
    module = fr.modules.RestartModule(
        realtime_interval=np.timedelta64(1, "h"), file_path=file_path,
        restart_command="sbatch job.sh")
    info = module.info
    assert info["Realtime Restart Interval"] == np.timedelta64(1, "h")
    assert info["Restart Command"] == "sbatch job.sh"
    assert info["File Path"] == file_path

    module = fr.modules.RestartModule(
        clock_trigger=fr.ClockTrigger(step_size=10), file_path=file_path)
    assert "Clock Trigger" in module.info


def test_file_path_sanitization(directory):
    # a string path is cast to Path, and underscores are replaced
    module = fr.modules.RestartModule(
        file_path=str(Path(directory) / "my_restart_file.dill"))
    assert module.file_path == Path(directory) / "my-restart-file.dill"


def test_slurm_restart_command(file_path, monkeypatch):
    def make_module():
        return fr.modules.RestartModule(
            realtime_interval=np.timedelta64(1, "h"), file_path=file_path)

    # no job id in the environment
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert make_module().restart_command is None

    # invalid job id
    monkeypatch.setenv("SLURM_JOB_ID", "not-a-number")
    assert make_module().restart_command is None

    # valid job id with a command in the job info
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    run_mock = MagicMock(return_value=MagicMock(
        stdout="   JobId=123\n   Command=/path/to/job.sh\n"))
    monkeypatch.setattr(subprocess, "run", run_mock)
    assert make_module().restart_command == "sbatch /path/to/job.sh"

    # valid job id without a command in the job info
    run_mock = MagicMock(return_value=MagicMock(stdout="   JobId=123\n"))
    monkeypatch.setattr(subprocess, "run", run_mock)
    assert make_module().restart_command is None


def test_disabled_module_skips_slurm_lookup(file_path, monkeypatch):
    # a disabled module must not query the environment for a job id
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    run_mock = MagicMock()
    monkeypatch.setattr(subprocess, "run", run_mock)

    module = fr.modules.RestartModule(file_path=file_path)

    assert module.restart_command is None
    assert not run_mock.called


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
