"""Runge Kutta time stepping methods."""
from __future__ import annotations

from copy import deepcopy
from enum import Enum

import numpy as np

import fridom.framework as fr


class ButcherTableau:

    """
    Butcher tableau for Runge-Kutta time stepping methods.

    Parameters
    ----------
    A : np.ndarray
        Matrix of coefficients.
    b : np.ndarray
        Vector of coefficients.
    c : np.ndarray
        Vector of coefficients.

    """

    def __init__(self,
                 A: np.ndarray,
                 b: np.ndarray,
                 c: np.ndarray,
                 b_error: np.ndarray | None = None) -> None:
        self.A = A
        self.b = b
        self.c = c
        self.b_error = b_error
        self.order = len(b)

class RKMethods(Enum):

    """Enumeration of Runge-Kutta methods."""

    # ----------------------------------------------------------------
    #  Explicit Rung-Kutta methods (fixed time step)
    # ----------------------------------------------------------------

    Euler = ButcherTableau(
        A = np.array([0]),
        b = np.array([1]),
        c = np.array([0]),
    )

    RK2 = ButcherTableau(
        A = np.array([[0,   0],
                      [1/2, 0]]),
        b = np.array([0,   1]),
        c = np.array([0, 1/2]),
    )

    RK3 = ButcherTableau(
        A = np.array([[  0, 0, 0],
                      [1/2, 0, 0],
                      [ -1, 2, 0]]),
        b = np.array([1/6, 2/3, 1/6]),
        c = np.array([  0, 1/2,   1]),
    )

    RK4 = ButcherTableau(
        A = np.array([[  0,   0, 0, 0],
                      [1/2,   0, 0, 0],
                      [  0, 1/2, 0, 0],
                      [  0,   0, 1, 0]]),
        b = np.array([1/6, 1/3, 1/3, 1/6]),
        c = np.array([  0, 1/2, 1/2,   1]),
    )

    RK4_38 = ButcherTableau(
        A = np.array([[   0,  0, 0, 0],
                      [ 1/3,  0, 0, 0],
                      [-1/3,  1, 0, 0],
                      [   1, -1, 1, 0]]),
        b = np.array([1/8, 3/8, 3/8, 1/8]),
        c = np.array([  0, 1/3, 2/3,   1]),
    )

    # ----------------------------------------------------------------
    #  Adaptive Runge-Kutta methods
    # ----------------------------------------------------------------

    HEUN_EULER = ButcherTableau(
        A = np.array([[0, 0],
                      [1, 0]]),
        b = np.array([1/2, 1/2]),
        c = np.array([  0,   1]),
        b_error=np.array([1/2, -1/2]),
    )

    BOGACKI_SHAMPINE = ButcherTableau(
        A = np.array([[  0,   0,   0, 0],
                      [1/2,   0,   0, 0],
                      [  0, 3/4,   0, 0],
                      [2/9, 1/3, 4/9, 0]]),
        b = np.array([7/24, 1/4, 1/3, 1/8]),
        c = np.array([   0, 1/2, 3/4,   1]),
        b_error=np.array([2/9-7/24, 1/3-1/4, 4/9-1/3, -1/8]),
    )

    RKF45 = ButcherTableau(
        A = np.array([[        0,          0,          0,         0,      0, 0],
                      [      1/4,          0,          0,         0,      0, 0],
                      [     3/32,       9/32,          0,         0,      0, 0],
                      [1932/2197, -7200/2197,  7296/2197,         0,      0, 0],
                      [  439/216,         -8,   3680/513, -845/4104,      0, 0],
                      [    -8/27,          2, -3544/2565, 1859/4104, -11/40, 0]]),
        b = np.array([16/135,   0, 6656/12825, 28561/56430, -9/50, 2/55]),
        c = np.array([     0, 1/4,        3/8,       12/13,     1,  1/2]),
        b_error = np.array([-1/360, 0, 128/4275,  2197/75240, -1/50, -2/55]),
    )

@fr.utils.jaxjit
def sum_product(coeefs, dt, k):
    return sum(coeefs[i] * dt * k[i] for i in range(len(k)))

@fr.utils.jaxjit
def _compute_tendency(
    tendency: fr.modules.Module,
    mz: fr.ModelState,
) -> fr.ModelState:
    return tendency.update(mz=mz)

#TODO(Silvano): Jaxify this class
class RungeKutta(fr.time_steppers.TimeStepper):

    #TODO(Silvano): Add documentation

    name = "Runge-Kutta"
    def __init__(self,
                 dt: np.timedelta64 | float = 1,
                 method: RKMethods = RKMethods.RK4,
                 max_dt: np.timedelta64 | float | None = None,
                 tol: float = 1e-6) -> None:
        super().__init__()
        self.method = method.value
        self.dt = dt
        self.max_dt = max_dt
        self.dz_list = None
        self.tol = tol

    def _on_setup(self) -> None:
        self.dz_list = [self.mset.state_constructor() for _ in range(self.method.order)]

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        """
        Update the model state to the next time level.

        Parameters
        ----------
        mz : ModelState
            Model state.

        """
        method = self.method
        order = method.order
        # clone the clock
        clock = deepcopy(mz.clock)
        mod_state = fr.ModelState(self.mset, clock=clock)
        error = 1
        while error > self.tol:
            k = []
            dt = self.dt
            for i in range(order):
                mod_state.clock.tick(method.c[i] * dt)
                mod_state.z = mz.z + sum_product(method.A[i], dt, k)
                mod_state.dz = self.dz_list[i]
                mod_state = _compute_tendency(self.mset.tendencies, mod_state)
                k.append(mod_state.dz)

            if method.b_error is not None:
                te = sum_product(method.b_error, dt, k)
                error = sum(f.norm_l2() for f in te.field_list)
                if self.max_dt is not None:
                    self.dt = min(float(0.9 * dt * (self.tol / error) ** (1 / order)),
                                  self.max_dt)
                else:
                    self.dt = float(0.9 * dt * (self.tol / error) ** (1 / order))
            else:
                error = 0

        mz.z += sum_product(method.b, dt, k)
        mz.clock.tick(dt)
        return mz

    @property
    def max_dt(self) -> np.timedelta64:
        """Time step size."""
        return self._max_dt

    @max_dt.setter
    def max_dt(self, value: np.timedelta64 | float | None) -> None:
        if value is None:
            self._max_dt = None
            return
        if isinstance(value, float | int):
            self._max_dt = value
        else:
            self._max_dt = fr.config.dtype_real(value / np.timedelta64(1, "s"))
        self.dt = self._max_dt
