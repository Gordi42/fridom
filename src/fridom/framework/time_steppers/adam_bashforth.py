import fridom.framework as fr
import numpy as np
from functools import partial


@partial(fr.utils.jaxify, dynamic=('dz_list', 'pointer', 'it_count', 'coeff_AB'))
class AdamBashforth(fr.time_steppers.TimeStepper):
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
    """
    def __init__(self, dt = 1, order: int = 3, eps=0.01):
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
        self.dt = dt
        return

    @fr.modules.module_method
    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        super().setup(mset)
        ncp = fr.config.ncp
        dtype = fr.config.dtype_real

        # Adam Bashforth coefficients including time step size
        self.coeffs = [
            ncp.asarray(self.AB1, dtype=dtype) * self.dt, 
            ncp.asarray(self.AB2, dtype=dtype) * self.dt,
            ncp.asarray(self.AB3, dtype=dtype) * self.dt, 
            ncp.asarray(self.AB4, dtype=dtype) * self.dt
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

    
    @fr.utils.jaxjit
    def _update_state(self, z: 'fr.StateBase', dz_list: 'list[fr.StateBase]'
                      ) -> 'fr.StateBase':
        """
        Jax jitted time stepping function for Adam-Bashforth.
    
        Parameters
        ----------
        `z` : `State`
            The state at the current time level.
        `dz_list` : `list[State]`
            List of tendency terms at previous time levels.
        
        Returns
        -------
        `State` : The updated state.
        """
        for i in range(len(dz_list)):  # loop over all time levels
            z += dz_list[i] * self.coeff_AB[i]
        return z

    @fr.modules.module_method
    def update(self, mz: 'fr.ModelState'):
        """
        Update the time stepper.
        """
        self.update_tendency()

        mz.dz = self.dz

        mz = self.mset.tendencies.update(mz)
        self.dz = mz.dz

        dz_list = [self.dz_list[p] for p in self.pointer]
        mz.z = self._update_state(mz.z, dz_list)

        self.it_count += 1
        mz.it += 1
        mz.time += self.dt
        return mz

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
        ctl = min(self.it_count, self.order-1)

        # list of Adam-Bashforth coefficients
        coeffs = self.coeffs

        # choose Adam-Bashforth coefficients of current time level
        self.coeff_AB = fr.utils.modify_array(self.coeff_AB, slice(None), 0)
        self.coeff_AB = fr.utils.modify_array(self.coeff_AB, slice(ctl+1), coeffs[ctl])
        return
    
    def time_discretization_effect(self, omega: np.ndarray) -> np.ndarray:
        # shorthand notation
        ncp = fr.config.ncp

        # cast omega to ndarray
        omega = ncp.array(omega)

        # get adam-bashforth coefficients
        ab_coefficients = [self.AB1, self.AB2, self.AB3, self.AB4]
        
        # get the coefficients for the current time level
        coeff = ncp.array(ab_coefficients[self.order-1])

        # construct polynomial coefficients for each grid point
        # tile the array such that coeff and omega have the same shape
        new_shape = tuple(list(omega.shape) + [1])
        coeff = ncp.tile(coeff, new_shape)
        omega = omega[..., ncp.newaxis]

        # calculate the polynomial coefficients
        coeff = ncp.multiply(omega, coeff) * 1j * self.dt

        # subtract 1 from the last coefficient
        last_col = (..., 0)
        coeff = fr.utils.modify_array(coeff, last_col, coeff[last_col] - 1)

        # leading coefficient is 1
        paddings = [(0,0)] * len(coeff.shape)
        paddings[-1] = (1,0)
        coeff = ncp.pad(coeff, paddings, 'constant', constant_values=(1,0))

        # reverse the order of the coefficients
        coeff = coeff[..., ::-1]

        def find_roots(c):
            """
            Find the last root of the polynomial.

            Parameters:
                c (1D array): Polynomial coefficients.

            Returns:
                root (complex): Last root of the polynomial.
            """
            return np.roots(c)[-1]

        # find the roots of the polynomial
        # root finding only works on the CPU
        coeff = fr.utils.to_numpy(coeff)
        roots = ncp.array(np.apply_along_axis(find_roots, -1, coeff))
    
        return -1j * ncp.log(roots) / self.dt

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def info(self) -> dict:
        res = super().info
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