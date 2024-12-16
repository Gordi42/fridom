"""Base class for all types of fields."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np

import fridom.framework as fr

if TYPE_CHECKING:
    import xarray as xr

class FieldBase:

    r"""
    Base class for all types of fields.

    Description
    -----------
    A field is a mathematical mapping from the grid space :math:`\Omega`
    to an abstract space :math:`\mathcal{F}`. This abstract space can for
    example be the real or complex numbers for scalar fields, or the
    space of vectors or tensors for vector or tensor fields.

    This base class defines the interface for all types of fields.

    Parameters
    ----------
    mset : fr.ModelSettingsBase
        The model settings.

    """
