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
                flux_left: fr.ScalarField,
                flux_right: fr.ScalarField,
                velocity: fr.ScalarField ) -> fr.ScalarField:
        """
        Compute the flux function.

        Parameters
        ----------
        flux_left : fr.ScalarField
            The flux field which is biased to the left.
        flux_right : fr.ScalarField
            The flux field which is biased to the right.
        velocity : fr.ScalarField
            The velocity field.

        Returns
        -------
        fr.ScalarField
            The computed flux function.

        """
        msg = "Subclasses must implement the compute method."
        raise NotImplementedError(msg)
