"""Tests for the figure saver module."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

import fridom.framework as fr

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
    # the figure saver works in the current directory
    monkeypatch.chdir(tmp_path)
    grid = fr.grid.cartesian.Grid(shape=(4,), domain_size=(1.0,))
    mset = fr.ModelSettingsBase(grid=grid)

    def _state_constructor() -> fr.VectorField:
        var = fr.ScalarField(mset, name="var")
        return fr.VectorField(mset, field_list=[var])

    mset.state_constructor = _state_constructor
    return mset.setup()


@pytest.fixture
def figure_saver(mset):
    saver = fr.modules.FigureSaver(
        filename="figures/test.png", model_time=1.0,
        plotter=SimplePlotter, dpi=32)
    return saver.setup(mset=mset)


# ================================================================
#  Tests
# ================================================================
def test_no_save_before_model_time(mset, figure_saver):
    mz = fr.ModelState(mset)
    mz.clock.time = 0.5

    mz = figure_saver.update(mz=mz)

    assert not figure_saver._created
    assert not Path("figures/test.png").exists()


def test_save_at_model_time(mset, figure_saver):
    mz = fr.ModelState(mset)
    mz.clock.time = 1.5

    mz = figure_saver.update(mz=mz)

    assert figure_saver._created
    assert Path("figures/test.png").exists()


def test_save_only_once(mset, figure_saver):
    mz = fr.ModelState(mset)
    mz.clock.time = 1.5
    mz = figure_saver.update(mz=mz)

    # remove the file and update again: the figure is not recreated
    Path("figures/test.png").unlink()
    mz = figure_saver.update(mz=mz)

    assert not Path("figures/test.png").exists()
