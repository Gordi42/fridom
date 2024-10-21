# Import external modules
import numpy as np
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.model import Model
from fridom.framework.projection.projection import Projection
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase
    from fridom.framework.state_base import StateBase

class NNMDBase(Projection):
    """
    Nonlinear normal mode decomposition.
    """
    def __init__(self, mset: 'ModelSettingsBase',
                 VecQ, VecP,
                 order=3,
                 enable_dealiasing=True) -> None:
        """
        Nonlinear balacing using Nonlinear Normal Mode Decomposition.

        Arguments:
            grid      (Grid)          : The grid.
            order     (int)           : The order up to which to perform the
                                        decomposition. Implemented are up to
                                        order 4.
            enable_dealiasing (bool)  : Whether to enable dealiasing.
        """
        mset = mset
        grid = mset.grid
        super().__init__(mset)

        # check if model is nonlinear
        if not mset.enable_nonlinear or mset.Ro == 0:
            print("WARNING: Model is linear.")
        # check if N and f are constant
        if hasattr(mset, "enable_varying_f"):
            if mset.enable_varying_f:
                print("WARNING: f is varying. NNMD may not work properly.")
        if hasattr(mset, "enable_varying_N"):
            if mset.enable_varying_N:
                print("WARNING: N is varying. NNMD may not work properly.")
        if hasattr(mset, "enable_harmonic"):
            if mset.enable_harmonic:
                print("WARNING: Harmonic friction is not included in Eigenvectors.")
        if hasattr(mset, "enable_biharmonic"):
            if mset.enable_biharmonic:
                print("WARNING: Biharmonic friction is not included in Eigenvectors.")
        if order > 4:
            raise ValueError("Order must be <= 4.")
        self.order = order

        # create the eigenvectors
        self.q0 = VecQ(0, mset)
        self.p0 = VecP(0, mset)
        self.qp = VecQ(1, mset)
        self.pp = VecP(1, mset)
        self.qm = VecQ(-1, mset)
        self.pm = VecP(-1, mset)

        self.one_over_omega = 1 / self.grid.omega_space_discrete
        # set inf to zero
        self.one_over_omega[np.isinf(self.one_over_omega)] = 0

        # initialize the model
        self.model = Model(mset)
        self.State = mset.state_constructor
        return
    
    def __call__(self, z: 'StateBase') -> 'StateBase':
        """
        Project a state to the balanced subspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_bal  (State) : The balanced state.
        """
        Ro = self.mset.Ro

        was_spectral = z.is_spectral
        # transform to spectral space if necessary
        z_hat = z if z.is_spectral else z.fft()

        # calculate the ZERO-order component
        z0_hat = self.zero_order(z_hat)
        if self.order == 0:
            return z0_hat if was_spectral else z0_hat.fft()

        # calculate the FIRST-order component
        dT_z0_hat = self.derivative_zero_order(z0_hat)
        z1_hat = self.first_order(z0_hat)
        if self.order == 1:
            z_bal_hat = z0_hat + Ro * z1_hat
            return z_bal_hat if was_spectral else z_bal_hat.fft()

        # calculate the SECOND-order component
        dT_z1_hat = self.derivative_first_order(z0_hat, dT_z0_hat)
        z2_hat = self.second_order(z0_hat, z1_hat, dT_z1_hat)
        if self.order == 2:
            z_bal_hat = z0_hat + Ro * z1_hat + Ro**2 * z2_hat
            return z_bal_hat if was_spectral else z_bal_hat.fft()

        # calculate the THIRD-order component
        dT_z2_hat = self.derivative_second_order(z0_hat, z1_hat, 
                                                 dT_z0_hat, dT_z1_hat)
        z3_hat = self.third_order(z0_hat, z1_hat, z2_hat, dT_z2_hat)
        if self.order == 3:
            z_bal_hat = z0_hat + Ro * z1_hat + Ro**2 * z2_hat + Ro**3 * z3_hat
            return z_bal_hat if was_spectral else z_bal_hat.fft()


        raise ValueError("Order must be <= 2.")
    
    # ========================================================================
    #  Calculation of the n-order term in the power series
    # ========================================================================

    def zero_order(self, z_hat: 'StateBase') -> 'StateBase':
        """
        Calculate the zero-order term of the power series.

        Arguments:
            z_hat (State) : The zero-order term in spectral space.
        """
        return self.proj_0(z_hat)

    
    def first_order(self, z0_hat: 'StateBase') -> 'StateBase':
        """
        Calculate the unscaled first order term of the power series.

        Arguments:
            z0_hat (State) : The zero-order term in spectral space.
        """
        self.check_if_input_is_spectral(z0_hat)

        # calculate the nonlinear tendency of z0
        interaction = self.non_linear(z0_hat)

        return self.solve_for_z(interaction)

    def second_order(self, z0_hat: 'StateBase', z1_hat: 'StateBase',
                     dT_z1_hat: 'StateBase') -> 'StateBase':
        """
        Calculate the unscaled second order term of the power series.

        Arguments:
            z0_hat (State) : The zero-order term in spectral space.
            z1_hat (State) : The first-order term in spectral space.
            dT_z1_hat (State) : The slow derivative of the first-order term in spectral space.
        """
        self.check_if_input_is_spectral(z0_hat, z1_hat, dT_z1_hat)

        # calculate the nonlinear interaction terms
        interaction = self.non_linear_inter(z0_hat, z1_hat)

        right_hand_side = 2 * interaction - dT_z1_hat

        return self.solve_for_z(right_hand_side)

    def third_order(self, z0_hat: 'StateBase', z1_hat: 'StateBase',
                    z2_hat: 'StateBase', dT_z2_hat: 'StateBase') -> 'StateBase':
        """
        Calculate the unscaled third order term of the power series.
        """
        self.check_if_input_is_spectral(z0_hat, z1_hat, z2_hat, dT_z2_hat)

        # calculate the nonlinear interaction terms
        interaction = 2*self.non_linear_inter(z0_hat, z2_hat) \
                      + self.non_linear(z1_hat)
        right_hand_side = interaction - dT_z2_hat
        return self.solve_for_z(right_hand_side)

    def solve_for_z(self, right_hand_side: 'StateBase') -> 'StateBase':
        """
        Solves the linear equation for z.
        $$ 
        i \\omega^\\pm \\hat{z}^\\pm = \\hat{f}^\\pm
        $$

        Arguments:
            right_hand_side (State) : The right hand side of the equation (f).
        """
        # initialize the solution
        z_hat = self.State(self.grid, is_spectral=True)

        # calculate each mode separately
        for proj, sign in zip([self.proj_p, self.proj_m], [-1, 1]):
            factor = sign * 1j * self.one_over_omega
            z_hat += proj(right_hand_side) * factor
        return z_hat

    # ========================================================================
    #  Slow derivative of the n-order term
    # ========================================================================

    def derivative_zero_order(self, z0_hat: 'StateBase') -> 'StateBase':
        """
        Calculates the slow derivative of the zero-order term.
        """
        self.check_if_input_is_spectral(z0_hat)

        # calculate the slow tendency of z0
        dT_z0_hat = self.proj_0(self.non_linear(z0_hat))

        return dT_z0_hat

    def derivative_first_order(self, z0_hat: 'StateBase', 
                               dT_z0_hat: 'StateBase') -> 'StateBase':
        """
        Calculates the slow derivative of the first-order term.
        """
        self.check_if_input_is_spectral(z0_hat, dT_z0_hat)

        # calculate the nonlinear interaction between z0 and dT_z0
        interaction = self.non_linear_inter(z0_hat, dT_z0_hat) * 2

        return self.solve_for_z(interaction)

    def derivative_second_order(self, z0_hat: 'StateBase', z1_hat: 'StateBase',
                                dT_z0_hat: 'StateBase', dT_z1_hat: 'StateBase') -> 'StateBase':
        """
        Calculates the slow derivative of the second-order term.
        """
        self.check_if_input_is_spectral(z0_hat, z1_hat, dT_z0_hat, dT_z1_hat)

        dT_inter = self.derivative_inter(z0_hat, z1_hat, dT_z0_hat, dT_z1_hat)

        # calculate second-order derivative of z0
        dT_dT_z0_hat = 2 * self.proj_0(self.non_linear_inter(z0_hat, dT_z0_hat))

        # calculate second-order derivative of z1
        dT_dT_z1_hat_inter = self.derivative_inter(
            z0_hat, dT_z0_hat, dT_z0_hat, dT_dT_z0_hat)

        dT_dT_z1_hat = self.solve_for_z(dT_dT_z1_hat_inter)
        return self.solve_for_z(2*dT_inter - dT_dT_z1_hat)


    def proj_p(self, z: 'StateBase') -> 'StateBase':
        """
        Project a state to the positive eigenspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The projected state.
        """
        return z.project(self.pp, self.qp)
    
    def proj_m(self, z: 'StateBase') -> 'StateBase':
        """
        Project a state to the negative eigenspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The projected state.
        """
        return z.project(self.pm, self.qm)
    
    def proj_0(self, z: 'StateBase') -> 'StateBase':
        """
        Project a state to the zero eigenspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The projected state.
        """
        return z.project(self.p0, self.q0)

    def check_if_input_is_spectral(self, *args):
        """
        Check if all input arguments are in spectral space.
        """
        for arg in args:
            if not arg.is_spectral:
                raise ValueError("Input must be in spectral space.")
        return






    def non_linear(self, z: 'StateBase') -> 'StateBase':
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
        z_nl = z_phys * 0
        model.nonlinear_tendency(z_phys, z_nl)

        # divide by Ro
        z_nl /= self.mset.Ro

        # transform back to spectral space if it was spectral
        z_nl = z_nl.fft() if spectral_transform else z_nl
        return z_nl

    def derivative_inter(self, z1_hat: 'StateBase', z2_hat: 'StateBase',
                         dT_z1_hat: 'StateBase', dT_z2_hat: 'StateBase') -> 'StateBase':
        """
        Calculate the slow derivative of the nonlinear term between two states.
        """
        # check if all states are in spectral space
        self.check_if_input_is_spectral(z1_hat, z2_hat, dT_z1_hat, dT_z2_hat)

        dT_inter = self.non_linear_inter(z1_hat + z2_hat, dT_z1_hat + dT_z2_hat) \
                   - self.non_linear_inter(z1_hat, dT_z1_hat) \
                   - self.non_linear_inter(z2_hat, dT_z2_hat)

        return dT_inter

    

    def non_linear_inter(self, z1: 'StateBase', z2: 'StateBase') -> 'StateBase':
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

    def time_tendency(self, z: 'StateBase') -> 'StateBase':
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