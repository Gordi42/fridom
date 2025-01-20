"""Adam Bashforth time stepping up to 4th order."""
from __future__ import annotations

from functools import partial

import numpy as np

import fridom.framework as fr

MAX_ORDER = 4

@partial(fr.utils.jaxify,
         dynamic=("dz_list", "pointer", "it_count", "coeff_AB", "coeffs"))
class AdamBashforth(fr.time_steppers.TimeStepper):

    r"""
    Adam Bashforth time stepping up to 4th order.

    Parameters
    ----------
    dt : float
        Time step size. (default 0.01)
    order : int
        Order of the time stepping. (default 3, max 4)
    eps : float
        2nd order bashforth correction. (default 0.01)

    Description
    -----------
    The Adam Bashforth time stepping scheme is a multi-step explicit time stepping
    scheme. It solves a given PDE

    .. math::
        \partial_t \boldsymbol{z} = \boldsymbol{F}(\boldsymbol{z}, t)

    by using the following scheme of order :math:`n`

    .. math::
        \boldsymbol{z}^{n+1} = \boldsymbol{z}^n
            + \Delta t \sum_{j=0}^{n-1} \alpha_j
                \boldsymbol{F}(\boldsymbol{z}^{n-j}, t^{n-j})

    where :math:`\alpha_i` are the Adam Bashforth coefficients, :math:`\Delta t`
    is the time step size, :math:`\boldsymbol{z}^j` is the state at time
    :math:`t^j = t_0 + j \Delta t`. The coefficients for orders 1 to 4 are
    given in the table below.

    +-------+-------------------+-------------------+-------------------+-------------------+
    | Order | :math:`\alpha_1`  | :math:`\alpha_2`  | :math:`\alpha_3`  | :math:`\alpha_4`  |
    +=======+===================+===================+===================+===================+
    | 1     | 1                 |                   |                   |                   |
    +-------+-------------------+-------------------+-------------------+-------------------+
    | 2     | 3/2 + \epsilon    | -1/2 - \epsilon   |                   |                   |
    +-------+-------------------+-------------------+-------------------+-------------------+
    | 3     | 23/12             | -4/3              | 5/12              |                   |
    +-------+-------------------+-------------------+-------------------+-------------------+
    | 4     | 55/24             | -59/24            | 37/24             | -3/8              |
    +-------+-------------------+-------------------+-------------------+-------------------+

    Stability Analysis
    ******************
    Let :math:`\lambda` be the eigenvalues of the right-hand side
    of the PDE, e.g:

    .. math::
        \partial_t \boldsymbol{z} = \boldsymbol{F}(\boldsymbol{z}, t)
        = -i \lambda \boldsymbol{z}

    Inserting this into the Adam Bashforth scheme gives:

    .. math::
        \boldsymbol{z}^{n+1} = \sum_{j=0}^{n-1} c_j \boldsymbol{z}^{n-j}

    where

    .. math::
        c_j = \begin{cases}
            1 - i \Delta t \lambda & \text{if } j = 0 \\
            -i \Delta t \lambda & \text{if } j > 0
        \end{cases}

    We now insert the Ansatz:

    .. math::
        \boldsymbol{z}^n = \boldsymbol{z}_0 e^{-i \omega n \Delta t}
                         = \boldsymbol{z}_0 x^n

    with :math:`x = e^{-i \omega \Delta t}`. This yields a polynomial equation
    for :math:`x`:

    .. math::
        x^{n+1} = \sum_{j=0}^{n-1} c_j x^{n-j}

    Finally, we find the eigenvalues of the time stepping scheme by solving
    the polynomial equation for :math:`x` numerically and taking the logarithm:

    .. math::
        \omega = -i \log(x) / \Delta t

    """

    name = "Adam Bashforth"
    def __init__(self,
                 dt: float = 1,
                 order: int = 3,
                 eps: float=0.01) -> None:
        # check that the order is not too high
        if order > MAX_ORDER:
            msg = f"Only support orders up to {MAX_ORDER}."
            raise ValueError(msg)

        super().__init__()
        self.order = order
        self.eps = eps
        self.AB1 = [1]
        self.AB2 = [3/2 + eps, -1/2 - eps]
        self.AB3 = [23/12, -4/3, 5/12]
        self.AB4 = [55/24, -59/24, 37/24, -3/8]
        self.it_count = None
        self.dt = dt

    def _on_setup(self) -> None:
        ncp = fr.config.ncp
        dtype = fr.config.dtype_real

        # Adam Bashforth coefficients including time step size
        self.coeffs = [
            ncp.asarray(self.AB1, dtype=dtype) * self.dt,
            ncp.asarray(self.AB2, dtype=dtype) * self.dt,
            ncp.asarray(self.AB3, dtype=dtype) * self.dt,
            ncp.asarray(self.AB4, dtype=dtype) * self.dt,
        ]

        self.coeff_AB = ncp.zeros(self.order, dtype=dtype)

        # pointers
        self.pointer = np.arange(self.order, dtype=ncp.int32)

        # tendencies
        self.dz_list = [self.mset.state_constructor() for _ in range(self.order)]
        self.it_count = 0

    def _on_reset(self) -> None:
        self._on_setup()

    @fr.utils.jaxjit
    def _update_state(self,
                      z: fr.VectorField,
                      dz_list: list[fr.VectorField],
                      ) -> fr.VectorField:
        """
        Jax jitted time stepping function for Adam-Bashforth.

        Parameters
        ----------
        z : State
            The state at the current time level.
        dz_list : list[State]
            List of tendency terms at previous time levels.

        Returns
        -------
        State : The updated state.

        """
        for i in range(len(dz_list)):  # loop over all time levels
            z += dz_list[i] * self.coeff_AB[i]
        return z

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        """Update the time stepper."""
        self._update_tendency()

        mz.dz = self.dz

        mz = self.mset.tendencies.update(mz)
        self.dz = mz.dz

        dz_list = [self.dz_list[p] for p in self.pointer]
        mz.z = self._update_state(mz.z, dz_list)

        self.it_count += 1
        mz.clock.tick(self.dt)
        return mz

    def _update_tendency(self) -> None:
        if self.it_count <= self.order+1:
            self.update_coeff_AB()
        self.update_pointer()

    def update_pointer(self) -> None:
        """Update pointer for Adam-Bashforth time stepping."""
        self.pointer = np.roll(self.pointer, 1)

    def update_coeff_AB(self) -> None:
        """Upward ramping of Adam-Bashforth coefficients after restart."""
        # current time level (ctl)
        # maximum ctl is the number of time levels - 1
        ctl = min(self.it_count, self.order-1)

        # list of Adam-Bashforth coefficients
        coeffs = self.coeffs

        # choose Adam-Bashforth coefficients of current time level
        self.coeff_AB = fr.utils.modify_array(self.coeff_AB, slice(None), 0)
        self.coeff_AB = fr.utils.modify_array(self.coeff_AB, slice(ctl+1), coeffs[ctl])

    def time_discretization_effect(self, omega: np.ndarray) -> np.ndarray:  # noqa: D102
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
        new_shape = (*tuple(omega.shape), 1)
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
        coeff = ncp.pad(coeff, paddings, "constant", constant_values=(1,0))

        # reverse the order of the coefficients
        coeff = coeff[..., ::-1]

        def find_roots(c: np.ndarray) -> complex:
            """
            Find the last root of the polynomial.

            Parameters
            ----------
            c : ndarray
                Polynomial coefficients.

            Returns
            -------
            complex
                Last root of the polynomial.

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
    def _on_time_step_change(self) -> None:
        if not self.is_setup:
            return
        # we need to call the setup method again when the time step is changed
        self._on_setup()

    @property
    def info(self) -> dict:  # noqa: D102
        second_order = 2
        res = super().info
        res["order"] = self.order
        if self.order == second_order:
            res["eps"] = self.eps
        return res

    @property
    def dz(self) -> fr.VectorField:
        """Pointer on the current tendency term."""
        return self.dz_list[self.pointer[0]]

    @dz.setter
    def dz(self, value: fr.VectorField) -> None:
        self.dz_list[self.pointer[0]] = value
