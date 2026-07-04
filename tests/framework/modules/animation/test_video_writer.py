"""Tests for the video writer module."""
import queue
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

import fridom.framework as fr
import fridom.framework.modules.animation.video_writer as video_writer_module

mpl.use("Agg")


# ================================================================
#  Helpers
# ================================================================
class SimplePlotter(fr.modules.animation.ModelPlotter):

    """A minimal matplotlib based model plotter."""

    @staticmethod
    def create_figure():
        return plt.figure(figsize=(2, 2), dpi=32)

    @staticmethod
    def prepare_arguments(mz):
        return {"time": float(mz.clock.time)}

    @staticmethod
    def update_figure(fig, time):
        ax = fig.add_subplot()
        ax.plot([0, 1], [0, time])


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset(tmp_path, monkeypatch):
    # the video writer works in the current directory
    monkeypatch.chdir(tmp_path)
    grid = fr.grid.cartesian.Grid(shape=(4,), domain_size=(1.0,))
    mset = fr.ModelSettingsBase(grid=grid)

    def _state_constructor() -> fr.VectorField:
        var = fr.ScalarField(mset, name="var")
        return fr.VectorField(mset, field_list=[var])

    mset.state_constructor = _state_constructor
    return mset.setup()


def make_writer(mset, parallel, filename="output"):
    writer = fr.modules.animation.VideoWriter(
        SimplePlotter, model_time_per_second=1.0, fps=5,
        parallel=parallel, filename=filename)
    writer.setup(mset=mset)
    return writer


def count_frames(filename):
    reader = imageio.get_reader(filename)
    try:
        return reader.count_frames()
    finally:
        reader.close()


# ================================================================
#  Tests
# ================================================================
def test_initialization(mset):
    writer = make_writer(mset, parallel=False)
    assert writer.filename == str(Path("videos") / "output.mp4")
    assert writer.write_interval == pytest.approx(0.2)
    assert not writer.mpi_available
    assert writer.info["fps"] == 5


def test_timedelta_model_time():
    writer = fr.modules.animation.VideoWriter(
        SimplePlotter, model_time_per_second=np.timedelta64(2, "s"), fps=4)
    assert writer.write_interval == pytest.approx(0.5)
    assert writer.filename == str(Path("videos") / "output.mp4")


# imageio-ffmpeg leaves the subprocess pipes to the garbage collector,
# which raises ResourceWarnings at finalization time
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_serial_video(mset):
    # the writer produces one frame every write_interval model seconds
    writer = make_writer(mset, parallel=False)
    writer.start()

    mz = fr.ModelState(mset)
    for time in np.arange(0.0, 1.0, 0.05):
        mz.clock.time = float(time)
        mz = writer.update(mz=mz)
    writer.stop()

    assert Path(writer.filename).exists()
    assert count_frames(writer.filename) == 5


def test_setup_deletes_existing_file(mset):
    writer = make_writer(mset, parallel=False)
    Path(writer.filename).touch()
    writer.setup(mset=mset, setup_mode="forced")
    assert not Path(writer.filename).exists()


def test_start_with_open_writer_keeps_writer(mset):
    writer = make_writer(mset, parallel=False)
    writer.start()
    previous = writer.writer
    writer.start()
    assert writer.writer is previous
    writer.stop()


def test_stop_without_writer(mset):
    writer = make_writer(mset, parallel=False)
    writer.running_jobs = []
    writer.open_queues = []
    writer.stop()
    assert writer.writer is None


def test_start_after_stop_creates_new_writer(mset):
    writer = make_writer(mset, parallel=False)
    writer.start()
    previous = writer.writer
    writer.stop()
    writer.start()
    assert writer.writer is not previous
    writer.stop()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_parallel_video(mset, monkeypatch):
    # replace the process spawning with a synchronous fake
    class FakeProcess:
        def __init__(self, target, kwargs):
            self._target = target
            self._kwargs = kwargs

        def start(self):
            self._target(**self._kwargs)

        def join(self):
            pass

    monkeypatch.setattr(video_writer_module.mp, "Process", FakeProcess)

    writer = make_writer(mset, parallel=True)
    writer.maximum_jobs = max(writer.maximum_jobs, 1)
    writer.start()

    mz = fr.ModelState(mset)
    for time in (0.0, 0.5, 1.0):
        mz.clock.time = time
        mz = writer.update(mz=mz)
    writer.stop()

    assert Path(writer.filename).exists()
    assert count_frames(writer.filename) == 3


def test_parallel_update_waits_for_free_slot(mset, monkeypatch):
    # when the maximum number of jobs is reached, parallel_update waits
    # until a job finishes
    writer = make_writer(mset, parallel=True)
    writer.maximum_jobs = 1
    writer.writer = MagicMock()

    img = np.zeros((64, 64, 4), dtype=np.uint8)
    finished = queue.Queue()
    finished.put(img)
    writer.open_queues = [finished]
    writer.running_jobs = [MagicMock()]

    # the first collect call does nothing, so that the wait loop of
    # parallel_update has to collect the figures
    original_collect = writer.collect_figures
    calls = []

    def delayed_collect():
        if calls:
            original_collect()
        calls.append(1)

    monkeypatch.setattr(writer, "collect_figures", delayed_collect)
    monkeypatch.setattr(
        video_writer_module.mp, "Process",
        lambda target, kwargs: MagicMock())  # noqa: ARG005

    writer.parallel_update(fr.ModelState(mset))

    assert len(calls) >= 2
    assert writer.writer.append_data.called


def test_to_numpy_shares_writer(mset):
    writer = make_writer(mset, parallel=False)
    writer.writer = MagicMock()
    copy = writer.__to_numpy__({})
    assert copy.writer is writer.writer


def test_p_make_figure():
    output_queue = queue.Queue()
    fr.modules.animation.VideoWriter.p_make_figure(
        kwargs={"time": 0.5}, output_queue=output_queue,
        model_plotter=SimplePlotter)
    img = output_queue.get(timeout=1)
    assert img.shape == (64, 64, 4)
    assert img.dtype == np.uint8


def test_collect_figures(mset):
    writer = make_writer(mset, parallel=False)
    writer.start()

    # a queue with a finished figure is collected
    img = np.zeros((64, 64, 4), dtype=np.uint8)
    finished = queue.Queue()
    finished.put(img)
    job = MagicMock()
    writer.open_queues = [finished]
    writer.running_jobs = [job]
    writer.collect_figures()
    assert len(writer.running_jobs) == 0
    assert job.join.called

    # an empty queue leaves the job running
    writer.open_queues = [queue.Queue()]
    writer.running_jobs = [MagicMock()]
    writer.collect_figures()
    assert len(writer.running_jobs) == 1

    writer.running_jobs = []
    writer.open_queues = []
    writer.stop()


def test_show_video(mset):
    writer = make_writer(mset, parallel=False)
    writer.start()
    mz = fr.ModelState(mset)
    writer.update(mz=mz)
    writer.stop()

    video = writer.show_video(width=100)
    assert type(video).__name__ == "Video"


def test_deepcopy_shares_writer_and_figure(mset):
    writer = make_writer(mset, parallel=False)
    writer.start()

    copy = deepcopy(writer)

    assert copy.writer is writer.writer
    assert copy.fig is writer.fig
    assert copy.filename == writer.filename
    writer.stop()
