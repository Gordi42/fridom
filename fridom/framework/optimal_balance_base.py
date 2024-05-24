import numpy as np

from fridom.framework.grid_base import GridBase
from fridom.framework.state_base import StateBase
from fridom.framework.model_base import ModelBase
from fridom.framework.projection_base import Projection


class OptimalBalanceBase(Projection):
    """
    Nonlinear balancing using the optimal balance method.
    """
    def __init__(self, grid: GridBase,
                 Model: ModelBase,
                 base_proj:Projection,
                 ramp_period=1,
                 ramp_type="exp",
                 enable_forward_friction=False,
                 enable_backward_friction=False,
                 update_base_point=True,
                 max_it=3,
                 stop_criterion=1e-9,
                 return_details=False) -> None:
        """
        Nonlinear balancing using the optimal balance method.

        Arguments:
            grid      (Grid)          : The grid.
            base_proj  (Projection)   : The projection onto the base point.
            ramp_period (float)       : The ramping period (not scaled).
            ramp_type   (str)         : The ramping type. Choose from
                                        "exp", "pow", "cos", "lin".
            enable_forward_friction (bool) : Whether to enable forward friction.
            enable_backward_friction (bool): Whether to enable backward friction.
                                        (Backward friction has negative sign.)
            update_base_point (bool)  : Whether to update the base point 
                                        after each iteration. This has no effect
                                        on OB. But it matters for OBTA. Should 
                                        be True for OBTA.
            max_it     (int)          : Maximum number of iterations.
            stop_criterion (float)    : The stopping criterion.
        """
        mset = grid.mset
        super().__init__(mset.copy(), grid)

        # check the model settings
        if not mset.enable_nonlinear or mset.Ro == 0:
            print("WARNING: Model is linear.")
        # disable forcing
        mset.enable_source = False
        # disable diagnostics
        mset.enable_diag = False
        # disable snapshots
        mset.enable_snap = False

        self.base_proj = base_proj
        self.return_details = return_details
        
        # initialize the model
        self.model = Model(mset, grid)

        # save the parameters
        self.ramp_period    = ramp_period
        self.ramp_steps     = int(ramp_period / mset.dt)
        self.ramp_func      = OptimalBalanceBase.get_ramp_func(ramp_type)
        self.max_it         = max_it
        self.stop_criterion = stop_criterion
        self.enable_forward_friction  = enable_forward_friction
        self.enable_backward_friction = enable_backward_friction
        self.update_base_point = update_base_point

        # save the rossby number
        self.rossby = float(mset.Ro)

        # prepare the balancing
        self.z_base = None
        return

    def calc_base_coord(self, z: StateBase) -> None:
        self.z_base = self.base_proj(z)
        return

    def forward_to_nonlinear(self, z: StateBase) -> StateBase:
        """
        Perform forward ramping from linear model to nonlinear model.

        Arguments:
            z      (State) : The state to ramp.

        Returns:
            z_ramp (State) : The ramped state.
        """

        model = self.model
        model.reset()
        mset = model.mset

        # update model settings
        model.mset.enable_biharmonic = self.enable_forward_friction
        model.mset.enable_harmonic = self.enable_forward_friction
        # make sure that the parameters are positive
        for attr in ["ah", "kh", "ahbi", "khbi", "dt"]:
            setattr(mset, attr, np.abs(getattr(mset, attr)))

        # initialize the model
        model.z = z.copy()

        # perform the forward ramping
        for n in range(self.ramp_steps):
            mset.Ro = self.rossby * self.ramp_func(n / self.ramp_steps)
            model.step()
        return model.z
    
    def backward_to_linear(self, z: StateBase) -> StateBase:
        """
        Perform backward ramping from nonlinear model to linear model.

        Arguments:
            z      (State) : The state to ramp.

        Returns:
            z_ramp (State) : The ramped state.
        """
        model = self.model
        model.reset()
        mset = model.mset

        # update model settings
        model.mset.enable_biharmonic = self.enable_backward_friction
        model.mset.enable_harmonic = self.enable_backward_friction
        # make sure that the parameters are negative
        for attr in ["ah", "kh", "ahbi", "khbi", "dt"]:
            setattr(mset, attr, - np.abs(getattr(mset, attr)))

        # initialize the model
        model.z = z.copy()

        # perform the backward ramping
        for n in range(self.ramp_steps):
            mset.Ro = self.rossby * self.ramp_func(1 - n / self.ramp_steps)
            model.step()
        return model.z

    def forward_to_linear(self, z: StateBase) -> StateBase:
        """
        Perform forward ramping from nonlinear model to linear model.

        Arguments:
            z      (State) : The state to ramp.

        Returns:
            z_ramp (State) : The ramped state.
        """
        model = self.model
        model.reset()
        mset = model.mset

        # update model settings
        model.mset.enable_biharmonic = self.enable_forward_friction
        model.mset.enable_harmonic = self.enable_forward_friction
        # make sure that the parameters are positive
        for attr in ["ah", "kh", "ahbi", "khbi", "dt"]:
            setattr(mset, attr, np.abs(getattr(mset, attr)))

        # initialize the model
        model.z = z.copy()

        # perform the forward ramping
        for n in range(self.ramp_steps):
            mset.Ro = self.rossby * self.ramp_func(n / self.ramp_steps)
            model.step()
        return model.z

    def backward_to_nonlinear(self, z: StateBase) -> StateBase:
        """
        Perform backward ramping from linear model to nonlinear model.

        Arguments:
            z      (State) : The state to ramp.

        Returns:
            z_ramp (State) : The ramped state.
        """
        model = self.model
        model.reset()
        mset = model.mset

        # update model settings
        model.mset.enable_biharmonic = self.enable_backward_friction
        model.mset.enable_harmonic = self.enable_backward_friction
        # make sure that the parameters are negative
        for attr in ["ah", "kh", "ahbi", "khbi", "dt"]:
            setattr(mset, attr, -np.abs(getattr(mset, attr)))

        # initialize the model
        model.z = z.copy()

        # perform the backward ramping
        for n in range(self.ramp_steps):
            mset.Ro = self.rossby * self.ramp_func(1 - n / self.ramp_steps)
            model.step()
        return model.z

    def get_ramp_func(ramp_type):
        if ramp_type == "exp":
            def ramp_func(theta):
                t1 = 1./np.maximum(1e-32,theta )
                t2 = 1./np.maximum(1e-32,1.-theta )
                return np.exp(-t1)/(np.exp(-t1)+np.exp(-t2))  
        elif ramp_type == "pow":
            def ramp_func(theta):
                return theta**3/(theta**3+(1.-theta)**3)
        elif ramp_type == "cos":
            def ramp_func(theta):
                return 0.5*(1.-np.cos(np.pi*theta))
        elif ramp_type == "lin":
            def ramp_func(theta):
                return theta
        else:
            raise ValueError(
                "Invalid ramp type. Choose from 'exp', 'pow', 'cos', 'lin'.")
        return ramp_func
        

    def __call__(self, z: StateBase) -> StateBase:
        """
        Project a state to the geostrophic subspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_bal  (State) : The balanced state.
        """
        verbose = self.mset.print_verbose

        iterations = np.arange(self.max_it)
        errors     = np.ones(self.max_it)

        # save the base coordinate
        self.calc_base_coord(z)

        z_res = z.copy()

        # start the iterations
        verbose("Starting optimal balance iterations")

        for it in iterations:
            verbose(f"Starting iteration {it}")
            # backward ramping
            verbose("Performing backward ramping")
            z_lin = self.backward_to_linear(z_res)
            # project to the base point
            z_lin = self.base_proj(z_lin)
            # forward ramping
            verbose("Performing forward ramping")
            z_bal = self.forward_to_nonlinear(z_lin)
            # exchange base point coordinate
            z_new = z_bal - self.base_proj(z_bal) + self.z_base

            # calculate the error
            errors[it] = error = z_new.norm_of_diff(z_res)

            verbose(f"Difference to previous iteration: {error:.2e}")

            # update the state
            z_res = z_new

            # check the stopping criterion
            if error < self.stop_criterion:
                verbose("Stopping criterion reached.")
                break

            # check if the error is increasing
            if it > 0 and error > errors[it-1]:
                verbose("WARNING: Error is increasing. Stopping iterations.")
                break

            # recalculate the base coordinate if needed
            if self.update_base_point:
                # check if it is not the last iteration
                if it < self.max_it - 1:
                    self.calc_base_coord(z_res)

        if self.return_details:
            return z_res, (iterations, errors)
        else:
            return z_res

# remove symbols from the namespace
del GridBase, StateBase, ModelBase, Projection