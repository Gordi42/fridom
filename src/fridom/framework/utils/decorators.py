"""decorators.py: Utilities for decorators."""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from PIL import Image


def skip_on_doc_build(func: callable) -> callable:
    """
    Skip a function when building the documentation.

    Description
    -----------
    This decorator skips a function when building the documentation. This is
    useful to avoid expensive computations during the documentation build.

    Parameters
    ----------
    `func` : `callable`
        The function to skip.

    Returns
    -------
    `callable`
        The function that is skipped when building the documentation.

    Examples
    --------
    >>> import fridom.framework as fr
    >>> @fr.utils.skip_on_doc_build
    ... def my_function():
    ...     return "This function is skipped when building the documentation."
    """
    # check if we are building the documentation
    if os.getenv("FRIDOM_DOC_GENERATION") == "True":
        def do_nothing(*args, **kwargs) -> None:  # pylint: disable=unused-argument
            return None
        return do_nothing
    return func

def cache_figure(
        func: Callable,
        name: str | None = None,
        force_recompute: bool = False,
        dpi: int = 200) -> callable:
    """
    Cache a figure to disk, if it exists return the image from disk.

    Description
    -----------
    This decorator caches a figure to disk. If the figure already exists on
    disk, the image is loaded from disk. If the figure does not exist on disk,
    the figure is computed and saved to disk. This is useful to avoid
    recomputing expensive figures.

    Parameters
    ----------
    `func` : `Callable`
        The function that computes the figure. This function must return a
        matplotlib figure.
    `name` : `str`
        The name of the figure file.
    `force_recompute` : `bool` (default=False)
        If True, the figure is recomputed even if it exists on disk.
    `dpi` : `int` (default=200)
        The DPI of the figure.

    Returns
    -------
    `Callable`
        The function that returns the image.
    """
    def wrapper():
        # Find out the main file name
        filename = f"figures/{name.split('.', maxsplit=1)[0]}.png"
        # Create the cache directory if it does not exist
        Path("figures").mkdir(parents=True, exist_ok=True)
        # Check if we need to compute the figure
        if force_recompute or not Path(filename).exists():
            fig = func()
            fig.savefig(filename, dpi=dpi)

        return Image.open(filename)
    return wrapper
