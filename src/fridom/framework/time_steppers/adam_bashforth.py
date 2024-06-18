import numpy as np

from fridom.framework.modules.module import start_module, update_module
from fridom.framework.time_steppers.time_stepper import TimeStepper
from fridom.framework.model_state import ModelState

class AdamBashforth(TimeStepper):
    def __init__(self, dt: float = 0.01, order: int = 3, eps=0.01):
        """
        # Adam Bashforth Time Stepping.
        ## Args:
        - dt (float): Time step size. (default 0.01)
        - order (int): Order of the time stepping. (default 3, max 4)
        - eps (float): 2nd order bashforth correction. (default 0.01)
        """
        # check that the order is not too high
        if order > 4:
            raise ValueError(
                "Adam Bashforth Time Stepping only supports orders up to 4.")
        
        super().__init__("Adam Bashforth Time Stepping", 
                         dt=dt, order=order, eps=eps)
        self.AB1 = [1]
        self.AB2 = [3/2 + eps, -1/2 - eps]
        self.AB3 = [23/12, -4/3, 5/12]
        self.AB4 = [55/24, -59/24, 37/24, -3/8]
        self.it_count = None
        return

    @start_module
    def start(self):
        """
        # Start the time stepper.
        ## Args:
        - grid (GridBase): Grid object.
        - timer (TimerBase): Timer object.
        """
        cp = self.grid.cp
        mset = self.mset
        grid = self.grid

        # cast the time step to the correct data type
        self.dt = mset.dtype(self.dt)

        # Adam Bashforth coefficients
        self.coeffs = [
            cp.asarray(self.AB1, dtype=mset.dtype), 
            cp.asarray(self.AB2, dtype=mset.dtype),
            cp.asarray(self.AB3, dtype=mset.dtype), 
            cp.asarray(self.AB4, dtype=mset.dtype)
        ]
        self.coeff_AB = cp.zeros(self.order, dtype=mset.dtype)

        # pointers
        self.pointer = np.arange(self.order, dtype=cp.int32)

        # tendencies
        self.dz_list = [mset.state_constructor(grid) for _ in range(self.order)]
        self.it_count = 0
        return

    @update_module
    def update(self, mz: ModelState):
        """
        # Update the time stepper.
        ## Args:
        - mz (ModelStateBase): Model state.
        """
        dt = self.dt
        for i in range(self.order):
            mz.z += self.dz_list[self.pointer[i]] * dt * self.coeff_AB[i]

        self.it_count += 1
        mz.it += 1
        mz.time += dt
        return

    @update_module
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
        cp = self.grid.cp
        ctl = min(self.it_count, self.order-1)

        # list of Adam-Bashforth coefficients
        coeffs = self.coeffs

        # choose Adam-Bashforth coefficients of current time level
        self.coeff_AB[:]      = 0
        self.coeff_AB[:ctl+1] = cp.asarray(coeffs[ctl])
        return

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

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    dt = {self.dt}\n"
        res += f"    order = {self.order}\n"
        if self.order == 2:
            res += f"    eps = {self.eps}\n"
        return res

