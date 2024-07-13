# Import external modules
import numpy as np
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.time_steppers.time_stepper import TimeStepper
from fridom.framework.modules.module import setup_module, module_method
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState

# ================================================================
#  JIT COMPILATION
# ================================================================

@utils.jaxjit
def _time_step_jit(z: dict[str, np.ndarray], 
                  dz_list: list[dict[str, np.ndarray]], 
                  coeffs: np.ndarray) -> dict[str, np.ndarray]:
    """
    Jax jitted time stepping function for Adam-Bashforth.
    
    Parameters
    ----------
    `z` : `dict[str, np.ndarray]`
        The state at the current time level.
    `dz_list` : `list[dict[str, np.ndarray]]`
        List of tendency terms at previous time levels.
    `coeffs` : `np.ndarray`
        Coefficients for the Adam-Bashforth time stepping.
    """
    for key in z.keys():  # loop over all fields
        for i in range(len(dz_list)):  # loop over all time levels
            z[key] = z[key] + dz_list[i][key] * coeffs[i]
    return z

# ================================================================
#  ADAM BASHFORTH TIME STEPPING
# ================================================================
class AdamBashforth(TimeStepper):
    """
    Adam Bashforth time stepping up to 4th order.
    
    Parameters
    ----------
    `dt` : `float`
        Time step size. (default 0.01)
    `order` : `int`
        Order of the time stepping. (default 3, max 4)
    `eps` : `float`
        2nd order bashforth correction. (default 0.01)
    
    Attributes
    ----------
    `dz` : `State`
        Tendency term at the current time level.
    
    """
    def __init__(self, dt = np.timedelta64(1, 's'), order: int = 3, eps=0.01):
        # check that the order is not too high
        if order > 4:
            raise ValueError(
                "Adam Bashforth Time Stepping only supports orders up to 4.")
        
        super().__init__("Adam Bashforth", 
                         order=order, eps=eps)
        self.AB1 = [1]
        self.AB2 = [3/2 + eps, -1/2 - eps]
        self.AB3 = [23/12, -4/3, 5/12]
        self.AB4 = [55/24, -59/24, 37/24, -3/8]
        self.it_count = None
        self._dt_float = None
        self._dt_timedelta = None
        self.dt = dt
        return

    @setup_module
    def setup(self):
        ncp = config.ncp
        dtype = config.dtype_real

        # Adam Bashforth coefficients including time step size
        self.coeffs = [
            ncp.asarray(self.AB1, dtype=dtype) * self._dt_float, 
            ncp.asarray(self.AB2, dtype=dtype) * self._dt_float,
            ncp.asarray(self.AB3, dtype=dtype) * self._dt_float, 
            ncp.asarray(self.AB4, dtype=dtype) * self._dt_float
        ]

        self.coeff_AB = ncp.zeros(self.order, dtype=dtype)

        # pointers
        self.pointer = np.arange(self.order, dtype=ncp.int32)

        # tendencies
        self.dz_list = [self.mset.state_constructor() for _ in range(self.order)]
        self.it_count = 0
        return

    def reset(self):
        self.setup()
        return

    @module_method
    def update(self, mz: 'ModelState'):
        """
        Update the time stepper.
        """
        if config.backend_is_jax:
            dz_list = [self.dz_list[p].arr_dict for p in self.pointer]
            mz.z.arr_dict = _time_step_jit(mz.z.arr_dict, dz_list, self.coeff_AB)
        else:
            dz_list = [self.dz_list[p].arr_dict for p in self.pointer]
            mz.z.arr_dict = _time_step_jit(mz.z.arr_dict, dz_list, self.coeff_AB)
            # for i in range(self.order):
            #     mz.z += self.dz_list[self.pointer[i]] * self.coeff_AB[i]

        self.it_count += 1
        mz.it += 1
        mz.time += self.dt
        return

    @module_method
    def update_tendency(self):
        if self.it_count <= self.order+1:
            self.update_coeff_AB()
        self.update_pointer()
        return


    def update_pointer(self) -> None:
        """
        Update pointer for Adam-Bashforth time stepping.
        """
        self.pointer = np.roll(self.pointer, 1)
        return


    def update_coeff_AB(self) -> None:
        """
        Upward ramping of Adam-Bashforth coefficients after restart.
        """
        # current time level (ctl)
        # maximum ctl is the number of time levels - 1
        ncp = config.ncp
        ctl = min(self.it_count, self.order-1)

        # list of Adam-Bashforth coefficients
        coeffs = self.coeffs

        # choose Adam-Bashforth coefficients of current time level
        self.coeff_AB = utils.modify_array(self.coeff_AB, slice(None), 0)
        self.coeff_AB = utils.modify_array(self.coeff_AB, slice(ctl+1), coeffs[ctl])
        return

    @property
    def info(self) -> dict:
        res = super().info
        res["dt"] = f"{self.dt}"
        res["order"] = self.order
        if self.order == 2:
            res["eps"] = self.eps
        return res

    @property
    def dz(self):
        """
        Returns a pointer on the current tendency term.
        """
        return self.dz_list[self.pointer[0]]

    @dz.setter
    def dz(self, value):
        """
        Set the current tendency term.
        """
        self.dz_list[self.pointer[0]] = value
        return

    @property
    def dt(self) -> float:
        """
        Time step size.
        """
        return self._dt_timedelta

    @dt.setter
    def dt(self, value: np.timedelta64) -> None:
        """
        Set the time step size.
        """
        self._dt_timedelta = value
        self._dt_float = config.dtype_real(value / np.timedelta64(1, 's'))
        return
