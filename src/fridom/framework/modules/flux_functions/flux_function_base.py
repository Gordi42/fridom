"""Base class for flux functions."""
from __future__ import annotations

from abc import abstractmethod

import fridom.framework as fr


@fr.utils.jaxify
class FluxFunctionBase(fr.modules.Module):

    """
    Base class for flux functions.

    Description
    -----------
    This class implements the base interface for flux functions.

    """

    name = "Flux Function Base"

    @abstractmethod
    def compute(self,
                flux: fr.ScalarField,
                velocity: fr.ScalarField,
                axis: int) -> fr.ScalarField:
        """
        Compute the flux function.

        Parameters
        ----------
        flux : fr.ScalarField
            The flux to compute.
        velocity : fr.ScalarField
            The velocity field.
        axis : int
            The axis along which to compute the flux function.

        Returns
        -------
        fr.ScalarField
            The computed flux function.

        """
        msg = "Subclasses must implement the compute method."
        raise NotImplementedError(msg)
