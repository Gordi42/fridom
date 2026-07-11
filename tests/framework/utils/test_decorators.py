"""Tests for the decorator utilities."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import fridom.framework as fr

mpl.use("Agg")


# ================================================================
#  skip_on_doc_build
# ================================================================
def test_skip_on_doc_build_passthrough(monkeypatch):
    monkeypatch.delenv("FRIDOM_DOC_GENERATION", raising=False)

    def my_function():
        return 42

    assert fr.utils.skip_on_doc_build(my_function) is my_function


def test_skip_on_doc_build_skips(monkeypatch):
    monkeypatch.setenv("FRIDOM_DOC_GENERATION", "True")

    def my_function():
        return 42

    decorated = fr.utils.skip_on_doc_build(my_function)
    assert decorated is not my_function
    assert decorated() is None


# ================================================================
#  cache_figure
# ================================================================
def test_cache_figure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def make_figure():
        calls.append(1)
        fig = plt.figure(figsize=(1, 1), dpi=32)
        plt.close(fig)
        return fig

    cached = fr.utils.cache_figure(make_figure, name="my_figure.png", dpi=32)

    # the first call computes and caches the figure
    with cached() as image:
        assert isinstance(image, Image.Image)
    assert Path("figures/my_figure.png").exists()
    assert len(calls) == 1

    # the second call loads the cached figure from disk
    with cached() as image:
        assert isinstance(image, Image.Image)
    assert len(calls) == 1


def test_cache_figure_force_recompute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def make_figure():
        calls.append(1)
        fig = plt.figure(figsize=(1, 1), dpi=32)
        plt.close(fig)
        return fig

    cached = fr.utils.cache_figure(
        make_figure, name="my_figure", force_recompute=True, dpi=32)

    cached().close()
    cached().close()
    assert len(calls) == 2
