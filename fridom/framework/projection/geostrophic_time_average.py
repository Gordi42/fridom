import numpy as np

from fridom.framework.grid_base import GridBase
from fridom.framework.state_base import StateBase
from fridom.framework.projection.projection import Projection

class GeostrophicTimeAverage(Projection):
    """
    Geostrophic projection using time-averaging.
    """
    def __init__(self, grid: GridBase, 
                 n_ave=4,
                 equidistant_chunks=True,
                 max_period=None,
                 backward_forward=False,
                 disable_diagnostic=True) -> None:
        """
        Geostrophic projection using time-averaging.

        Arguments:
            grid      (Grid)          : The grid.
            n_ave             (int)   : Number of averages to perform.
            equidistant_chunks(bool)  : Whether to split the averaging periods 
                                        into equidistant chunks.
            max_period        (float) : The maximum period of the time averages.
                                        if None, the maximum period is set to
                                        the inertial period.
            backward_forward  (bool)  : Whether to use backward-forward averaging.
        """

        # initialization
        super().__init__(grid)
        mset = self.mset
        self.n_ave = n_ave
        if max_period is None:
            max_period = np.abs(2 * np.pi / mset.f0)
        
        # construct the averaging periods
        if equidistant_chunks:
            self.periods = np.linspace(max_period/2, max_period, n_ave+1)[1:][::-1]
        else:
            self.periods = np.ones(n_ave) * max_period

        # calculate the number of time steps
        self.n_steps = np.ceil(self.periods / mset.time_stepper.dt).astype(int)
        self.backward_forward = backward_forward

        # initialize the model
        from fridom.framework.model import Model
        self.model = Model(grid)

        # disable advection
        self.model.tendencies.advection.disable()
        # disable diagnostics
        if disable_diagnostic:
            self.model.diagnostics.disable()
        return

    def __call__(self, z: StateBase) -> StateBase:
        """
        Project a state to the geostrophic subspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_ave  (State) : The geostrophic state.
        """
        verbose = self.mset.print_verbose
        z_ave = z.copy()
        model = self.model
        time_stepper = model.time_stepper
        
        verbose("Starting time averaging")
        for n_its in self.n_steps:
            # forward averaging
            time_stepper.dt = np.abs(time_stepper.dt)
            verbose(f"Averaging forward for {n_its*time_stepper.dt:.2f} seconds")
            model.reset()
            model.z = z_ave.copy()
            for _ in range(n_its):
                model.step()
                z_ave += model.z
            z_ave /= (n_its + 1)

            # backward averaging
            if self.backward_forward:
                verbose(f"Averaging backwards for {n_its*time_stepper.dt:.2f} seconds")
                time_stepper.dt = - np.abs(time_stepper.dt)
                model.reset()
                model.z = z_ave.copy()
                for _ in range(n_its):
                    model.step()
                    z_ave += model.z
                z_ave /= (n_its + 1)

        return z_ave


# remove symbols from namespace
del GridBase, StateBase