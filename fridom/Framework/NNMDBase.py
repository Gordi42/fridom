import numpy as np

from fridom.Framework.ModelSettingsBase import ModelSettingsBase
from fridom.Framework.GridBase import GridBase
from fridom.Framework.StateBase import StateBase
from fridom.Framework.ModelBase import ModelBase
from fridom.Framework.ProjectionBase import Projection

class NNMDBase(Projection):
    """
    Nonlinear normal mode decomposition.
    """
    def __init__(self, mset: ModelSettingsBase, grid: GridBase,
                 Model: ModelBase, State: StateBase,
                 VecQ, VecP,
                 order=3,
                 enable_dealiasing=True) -> None:
        """
        Nonlinear balacing using Nonlinear Normal Mode Decomposition.

        Arguments:
            mset      (ModelSettings) : Model settings.
            grid      (Grid)          : The grid.
            order     (int)           : The order up to which to perform the
                                        decomposition. Implemented are up to
                                        order 4.
            enable_dealiasing (bool)  : Whether to enable dealiasing.
        """
        super().__init__(mset, grid)

        # check if model is nonlinear
        if not mset.enable_nonlinear or mset.Ro == 0:
            print("WARNING: Model is linear.")
        # check if N and f are constant
        if mset.enable_varying_f:
            print("WARNING: f is varying. NNMD may not work properly.")
        if mset.enable_varying_N:
            print("WARNING: N is varying. NNMD may not work properly.")
        if mset.enable_harmonic:
            print("WARNING: Harmonic friction is not included in Eigenvectors.")
        if mset.enable_biharmonic:
            print("WARNING: Biharmonic friction is not included in Eigenvectors.")
        if order > 4:
            raise ValueError("Order must be <= 4.")
        self.order = order

        # create the eigenvectors
        self.q0 = VecQ(0, mset, grid)
        self.p0 = VecP(0, mset, grid)
        self.qp = VecQ(1, mset, grid)
        self.pp = VecP(1, mset, grid)
        self.qm = VecQ(-1, mset, grid)
        self.pm = VecP(-1, mset, grid)

        self.one_over_omega = 1 / self.grid.omega_space_discrete
        # set inf to zero
        self.one_over_omega[np.isinf(self.one_over_omega)] = 0

        # initialize the model
        self.model = Model(mset, grid)
        self.State = State
        return
    
    def __call__(self, z: StateBase) -> StateBase:
        """
        Project a state to the balanced subspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_bal  (State) : The balanced state.
        """
        if self.order == 0:
            return self.balance_zero_order(z)
        elif self.order == 1:
            return self.balance_first_order(z)
        elif self.order == 2:
            return self.balance_second_order(z)
        else:
            raise ValueError("Order must be <= 2.")


    def balance_zero_order(self, z: StateBase) -> StateBase:
        """
        Balance zero order term.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_bal  (State) : The balanced state.
        """
        return z.project(self.p0, self.q0)
    
    def balance_first_order(self, z: StateBase) -> StateBase:
        """
        Balance up to first-order.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_bal  (State) : The balanced state.
        """
        was_spectral = z.is_spectral

        # transform to spectral space if necessary
        z_hat = z if z.is_spectral else z.fft()

        # calculate the zero-order component
        z0_hat = self.balance_zero_order(z_hat)
        # calculate the first-order component
        z1_hat = self.first_order(z0_hat)

        # power series
        Ro = self.mset.Ro
        z_bal_hat = z0_hat + Ro * z1_hat

        return z_bal_hat if was_spectral else z_bal_hat.fft()

    def balance_second_order(self, z: StateBase) -> StateBase:
        """
        Balance up to second order.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_bal  (State) : The balanced state.
        """
        was_spectral = z.is_spectral
        # transform to spectral space if necessary
        z_hat = z if z.is_spectral else z.fft()

        # calculate the zero-order component
        z0_hat = self.balance_zero_order(z_hat)
        # calculate the first-order component
        z1_hat = self.first_order(z0_hat)
        # calculate the second-order component
        z2_hat = self.second_order(z0_hat, z1_hat)

        # power series
        Ro = self.mset.Ro
        z_bal_hat = z0_hat + Ro * z1_hat + Ro**2 * z2_hat

        return z_bal_hat if was_spectral else z_bal_hat.fft()



    
    def first_order(self, z0_hat: StateBase) -> StateBase:
        """
        Calculate the unscaled first order term of the power series.

        Arguments:
            z0_hat (State) : The zero-order term in spectral space.
        """
        # check if z0_hat is in spectral space
        if not z0_hat.is_spectral:
            raise ValueError("Input must be in spectral space.")

        # short notation
        proj_p = self.proj_p; proj_m = self.proj_m; proj_0 = self.proj_0

        # calculate the nonlinear tendency of z0
        non_z0_hat = self.non_linear(z0_hat)

        # initialize the first-order term
        z1_hat = self.State(self.mset, self.grid, is_spectral=True)

        # calculate each mode separately
        for proj, sign in zip([proj_p, proj_m], [1, -1]):
            factor = sign * 1j * self.one_over_omega
            z1_hat -= proj(non_z0_hat) * factor

        return z1_hat

    def second_order(self, z0_hat: StateBase, z1_hat: StateBase) -> StateBase:
        """
        Calculate the unscaled second order term of the power series.

        Arguments:
            z0_hat (State) : The zero-order term in spectral space.
            z1_hat (State) : The first-order term in spectral space.
        """
        # check if input is in spectral space
        if not z0_hat.is_spectral:
            raise ValueError("Input must be in spectral space.")
        if not z1_hat.is_spectral:
            raise ValueError("Input must be in spectral space.")

        # short notation
        proj_p = self.proj_p; proj_m = self.proj_m; proj_0 = self.proj_0

        # slow tendency of z0
        dz0_hat = proj_0(self.non_linear(z0_hat))
        dt_non_z0 = self.non_linear_inter(z0_hat, dz0_hat)*2

        # initialize the second-order term
        z2_hat = self.State(self.mset, self.grid, is_spectral=True)

        # calculate each mode separately
        for proj, sign in zip([proj_p, proj_m], [1, -1]):
            factor = sign * 1j * self.one_over_omega
            # calculate the slow tendency of z1 projected to the subspace
            dt_z1 = proj(dt_non_z0) * factor
            # calculate the nonlinear interaction between z0 and z1
            non_z0_z1 = proj(self.non_linear_inter(z0_hat, proj(z1_hat)))
            # calculate the second-order term
            z2_hat = (dt_z1 - 2*non_z0_z1) * factor

        return z2_hat

    def proj_p(self, z: StateBase) -> StateBase:
        """
        Project a state to the positive eigenspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The projected state.
        """
        return z.project(self.pp, self.qp)
    
    def proj_m(self, z: StateBase) -> StateBase:
        """
        Project a state to the negative eigenspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The projected state.
        """
        return z.project(self.pm, self.qm)
    
    def proj_0(self, z: StateBase) -> StateBase:
        """
        Project a state to the zero eigenspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The projected state.
        """
        return z.project(self.p0, self.q0)







    def non_linear(self, z: StateBase) -> StateBase:
        """
        Calculate the nonlinear term.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_nl  (State) : The nonlinear term.
        """
        # todo dealiasing (mask after 2/3 rule)
        spectral_transform = z.is_spectral
        # transform to physical space if necessary
        z_phys = z.fft() if spectral_transform else z

        # initialize the model
        model = self.model
        model.reset()
        model.z = z_phys
        z_nl = model.nonlinear_dz()

        # transform back to spectral space if it was spectral
        z_nl = z_nl.fft() if spectral_transform else z_nl
        return z_nl
    

    def non_linear_inter(self, z1: StateBase, z2: StateBase) -> StateBase:
        """
        Calculate the nonlinear term between two states.

        Arguments:
            z1      (State) : The first state .
            z2      (State) : The second state .

        Returns:
            z_nl  (State) : The nonlinear term .
        """
        # check if both states are in same space
        if z1.is_spectral != z2.is_spectral:
            raise ValueError("Both states must be in same space.")

        # todo dealiasing (mask after 2/3 rule)
        spectral_transform = z1.is_spectral
        # transform to physical space if necessary
        z1_phys = z1.fft() if spectral_transform else z1
        z2_phys = z2.fft() if spectral_transform else z2

        z_nl = 0.5 * (self.non_linear(z1_phys + z2_phys) - 
                      self.non_linear(z1_phys) - self.non_linear(z2_phys))

        # transform back to spectral space if it was spectral
        z_nl = z_nl.fft() if spectral_transform else z_nl
        return z_nl

    def time_tendency(self, z: StateBase) -> StateBase:
        """
        Calculate the time tendency.

        Arguments:
            z      (State) : The state to project.

        Returns:
            dz  (State)    : The tendency term.
        """
        spectral_transform = z.is_spectral
        # transform to physical space if necessary
        z_phys = z.fft() if spectral_transform else z

        # initialize the model
        model = self.model
        model.reset()
        model.z = z_phys.copy()
        z_next = model.step()

        dz = (z_next - z_phys) / model.mset.dt

        return dz.fft() if spectral_transform else dz


        