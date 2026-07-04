"""Tests for the model plotter base class."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import fridom.framework as fr

mpl.use("Agg")


# ================================================================
#  Helpers
# ================================================================
class SimplePlotter(fr.modules.animation.ModelPlotter):

    """A minimal matplotlib based model plotter."""

    @staticmethod
    def prepare_arguments(mz):
        return {"time": float(mz.clock.time)}

    @staticmethod
    def update_figure(fig, time):
        ax = fig.add_subplot()
        ax.plot([0, 1], [0, time])


class FakeModelState:

    """A stand-in model state that only provides a clock."""

    def __init__(self, time):
        self.clock = fr.Clock()
        self.clock.time = time


# ================================================================
#  Tests
# ================================================================
def test_create_figure():
    fig = fr.modules.animation.ModelPlotter.create_figure()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_new_returns_updated_figure():
    # instantiating a plotter with a model state directly returns the
    # updated figure
    fig = SimplePlotter(FakeModelState(time=0.5))
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_convert_to_img():
    fig = plt.figure(figsize=(2, 2), dpi=32)
    img = fr.modules.animation.ModelPlotter.convert_to_img(fig)
    assert isinstance(img, np.ndarray)
    assert img.shape == (64, 64, 4)
    plt.close(fig)
