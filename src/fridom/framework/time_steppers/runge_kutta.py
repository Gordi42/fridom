from copy import deepcopy
import fridom.framework as fr
import numpy as np
from enum import Enum
from typing import Union


class ButcherTableau:
    """
    Butcher tableau for Runge-Kutta time stepping methods.
    
    Parameters
    ----------
    `A` : `np.ndarray`
        Matrix of coefficients.
    `b` : `np.ndarray`
        Vector of coefficients.
    `c` : `np.ndarray`
        Vector of coefficients.
    
    """
    def __init__(self, A, b, c, b_error=None):
        self.A = A
        self.b = b
        self.c = c
        self.b_error = b_error
        self.order = len(b)
        return

class RKMethods(Enum):
    """
    Enumeration of Runge-Kutta methods.
    
    """
    Euler = ButcherTableau(
        A = np.array([0]),
        b = np.array([1]),
        c = np.array([0])
    )
    RK2 = ButcherTableau(
        A = np.array([[0, 0],
                      [1/2, 0]]),
        b = np.array([0, 1]),
        c = np.array([0, 1/2])
    )
    RK3 = ButcherTableau(
        A = np.array([[0, 0, 0],
                      [1/2, 0, 0],
                      [-1, 2, 0]]),
        b = np.array([1/6, 2/3, 1/6]),
        c = np.array([0, 1/2, 1])
    )
    RK4 = ButcherTableau(
        A = np.array([[0, 0, 0, 0],
                      [1/2, 0, 0, 0],
                      [0, 1/2, 0, 0],
                      [0, 0, 1, 0]]),
        b = np.array([1/6, 1/3, 1/3, 1/6]),
        c = np.array([0, 1/2, 1/2, 1])
    )
    RK4_38 = ButcherTableau(
        A = np.array([[0, 0, 0, 0],
                      [1/3, 0, 0, 0],
                      [-1/3, 1, 0, 0],
                      [1, -1, 1, 0]]),
        b = np.array([1/8, 3/8, 3/8, 1/8]),
        c = np.array([0, 1/3, 2/3, 1])
    )
    HEUN_EULER = ButcherTableau(
        A = np.array([[0, 0],
                        [1, 0]]),
        b = np.array([1/2, 1/2]),
        c = np.array([0, 1]),
        b_error=np.array([1/2, -1/2])
    )
    BOGACKI_SHAMPINE = ButcherTableau(
        A = np.array([[0, 0, 0, 0],
                        [1/2, 0, 0, 0],
                        [0, 3/4, 0, 0],
                        [2/9, 1/3, 4/9, 0]]),
        b = np.array([7/24, 1/4, 1/3, 1/8]),
        c = np.array([0, 1/2, 3/4, 1]),
        b_error=np.array([2/9-7/24, 1/3-1/4, 4/9-1/3, -1/8])
    )
    RKF45 = ButcherTableau(
        A = np.array([[0, 0, 0, 0, 0, 0],
                      [1/4, 0, 0, 0, 0, 0],
                      [3/32, 9/32, 0, 0, 0, 0],
                      [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
                      [439/216, -8, 3680/513, -845/4104, 0, 0],
                      [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]]),
        b = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]),
        c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2]),
        b_error = np.array([-1/360, 0, 128/4275,  2197/75240, -1/50, -2/55])
    )

@fr.utils.jaxjit
def sum_product(coeefs, dt, k):
    return sum(coeefs[i] * dt * k[i] for i in range(len(k)))

class RungeKutta(fr.time_steppers.TimeStepper):
    name = "Runge-Kutta"
    def __init__(self, 
                 dt: Union[np.timedelta64, float] = 1, 
                 method: RKMethods = RKMethods.RK4,
                 max_dt: Union[np.timedelta64, float, None] = None, 
                 tol=1e-6):
        super().__init__()
        self.method = method.value
        self.dt = dt
        self.max_dt = max_dt
        self.dz_list = None
        self.tol = tol
        return

    @fr.modules.module_method
    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        super().setup(mset)
        self.dz_list = [self.mset.state_constructor() for _ in range(self.method.order)]
        return

    def calculate_tendency(self, mz: 'fr.ModelState') -> 'fr.StateBase':
        return self.mset.tendencies.update(mz).dz

    @fr.modules.module_method
    def update(self, mz: 'fr.ModelState') -> None:
        """
        Update the model state to the next time level.
        
        Parameters
        ----------
        `mz` : `ModelState`
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
                dz = self.calculate_tendency(mod_state)
                k.append(dz)
            
            if method.b_error is not None:
                te = sum_product(method.b_error, dt, k)
                error = sum(f.norm_l2() for f in te.field_list)
                if self.max_dt is not None:
                    self.dt = min(float(0.9 * dt * (self.tol / error) ** (1 / order)), self.max_dt)
                else:
                    self.dt = float(0.9 * dt * (self.tol / error) ** (1 / order))
            else:
                error = 0

        mz.z += sum_product(method.b, dt, k)
        mz.clock.tick(dt)
        mz.it += 1
        return mz

    @property
    def max_dt(self) -> np.timedelta64:
        """
        Time step size.
        """
        return self._max_dt

    @max_dt.setter
    def max_dt(self, value: Union[np.timedelta64, float, None]) -> None:
        if value is None:
            self._max_dt = None
            return
        if isinstance(value, float) or isinstance(value, int):
            self._max_dt = value
        else:
            self._max_dt = fr.config.dtype_real(value / np.timedelta64(1, 's'))
        self.dt = self._max_dt
