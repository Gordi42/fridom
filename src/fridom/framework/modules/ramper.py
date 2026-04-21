"""A module for ramping up and down parameters during a simulation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import fridom.framework as fr

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


class Ramper(fr.modules.Module):

    """
    A module for ramping up and down parameters during a simulation.

    Parameters
    ----------
    start_time : float
        The time at which the ramping should start.
    ramp_period : float
        The period over which the ramping should occur.
    update_parameters : Callable[[fr.ModelState, float], None], optional
        A method that updates the model parameters based on the ramped value.
        It should take the model state and the ramped value which is between 0 and 1.
    ramp_function : str or Callable[[float], float], optional
        The ramp function. It can be one of the following strings:
        "exponential", "power_3", "cosine", "linear" or a custom callable.
        The default is "exponential".

    Description
    -----------
    This module contains a list of ramp functions that can be used to ramp up and down
    parameters during a simulation. This can be useful for adiabatic processes or
    for slowly changing parameters. For example to ramp up the nonlinear term.

    """

    name = "Ramper"

    def __init__(self,
                 start_time: float,
                 ramp_period: float,
                 update_parameters: Callable[[fr.ModelState, float], None] = \
                     lambda mz, ramped_value: None,  # noqa: ARG005
                 ramp_function: ( Literal["exponential",
                                          "power_3",
                                          "cosine",
                                          "linear"]
                                | Callable[[float], float]
                                 ) = "exponential",
                 ) -> None:
        super().__init__()
        self.start_time = start_time
        self.ramp_period = ramp_period
        self.ramp_function = ramp_function
        self.update_parameters = update_parameters

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        time = float(mz.clock.time)
        # Check if the time is smaller than the start time
        if time < self.start_time:
            return mz
        # Check if the time is greater than the start time + ramp period
        if time > self.start_time + self.ramp_period:
            return mz
        # get the scaled time value
        theta = (time - self.start_time) / self.ramp_period
        # get the ramped value
        ramped_value = self.ramp_function(theta)
        # call the custom update parameters method
        self.update_parameters(mz, ramped_value)
        return mz

    # ================================================================
    #  The different ramp functions
    # ================================================================

    @staticmethod
    def exponential_ramp_func(theta: float) -> float:
        r"""
        Exponential ramp function.

        .. math::

            f(\theta) = \frac{e^{-\frac{1}{\theta}}}
                            {e^{-\frac{1}{\theta}} + e^{-\frac{1}{1-\theta}}}

        """
        t1 = 1. / max(1e-32, theta)
        t2 = 1. / max(1e-32, 1. - theta)
        return np.exp(-t1) / (np.exp(-t1) + np.exp(-t2))

    @staticmethod
    def power_3_ramp_func(theta: float) -> float:
        r"""
        Power 3 ramp function.

        .. math::

            f(\theta) = \frac{\theta^3}{\theta^3 + (1 - \theta)^3}

        """
        return theta**3 / (theta**3 + (1 - theta)**3)

    @staticmethod
    def cosine_ramp_func(theta: float) -> float:
        r"""
        Cosine ramp function.

        .. math::

            f(\theta) = \frac{1}{2} \left(1 - \cos(\pi \theta)\right)

        """
        return 0.5 * (1 - np.cos(np.pi * theta))

    @staticmethod
    def linear_ramp_func(theta: float) -> float:
        r"""
        Linear ramp function.

        .. math::

            f(\theta) = \theta

        """
        return theta

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def info(self) -> dict[str, str]:  # noqa: D102
        res = super().info
        res["start_time"] = str(self.start_time)
        res["ramp_period"] = str(self.ramp_period)
        res["ramp_function"] = self._ramp_name
        return res

    @property
    def ramp_function(self) -> Callable[[float], float]:
        """
        The ramp function.

        Parameters
        ----------
        theta : float
            The scaled time value (between 0 and 1).

        Returns
        -------
        float
            The ramped value (between 0 and 1).

        """
        return self._ramp_func

    @ramp_function.setter
    def ramp_function(self, value: Literal["exponential",
                                           "power_3",
                                           "cosine",
                                           "linear"] | Callable[[float], float],
                      ) -> None:
        # If the value is a callable, set the ramp function to that callable
        if callable(value):
            self._ramp_name = value.__name__
            self._ramp_func = value
            return
        # Else, set the ramp function to the corresponding method
        self._ramp_name = value
        self._ramp_func = {
            "exponential": self.exponential_ramp_func,
            "power_3": self.power_3_ramp_func,
            "cosine": self.cosine_ramp_func,
            "linear": self.linear_ramp_func,
        }[value]
