"""Scalar field class definition."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import fridom.framework as fr

if TYPE_CHECKING:
    from numpy import ndarray
    import xarray as xr

class ScalarField(fr.FieldBase):

    """
    A scalar mapping from grid space to real / complex numbers.

    Description
    -----------
    A scalar field is the most basic field in FRIDOM. It is a mapping from the
    grid space to real or complex numbers. It is used to represent scalar
    quantities like pressure, temperature, etc. Essentially, a scalar field is
    wrapper around a numpy-like array with additional metadata and methods.

    """

