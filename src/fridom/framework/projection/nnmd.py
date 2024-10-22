import fridom.framework as fr
from scipy.special import comb
import numpy as np

class NNMD(fr.projection.Projection):
    r"""
    Nonlinear normal mode decomposition

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `order` : `int`
        The order of the balanced state.
    `epsilon` : `float` (default: None)
        The small parameter epsilon.
    `use_discrete` : `bool` (default: True)
        Whether to use the discrete eigenvectors vectors.
    `enable_dealiasing` : `bool` (default: True)
        Whether to enable dealiasing when computing the nonlinear term.

    Description
    -----------
    Nonlinear normal mode decomposition (NNMD) is a method to filter out all fast waves from a given state :math:`\boldsymbol z`. The obtained state :math:`\boldsymbol z_b` is called a balanced state. For a linear system, such a balancing operation would be the projection onto the eigenspace with the smallest eigenvalue, e.g. the geostrophic mode for ocean models. In contrast to the simple projection onto the linear geostrophic mode, NNMD takes the nonlinear terms into account. 

    The key concept of NNMD emerged independently through the works of Machenhauer (1977) [1]_ and Bear & Tribbia (1977) [2]_ . Building on this foundation, Warn et al. (1995) [3]_ further developed this approach by expanding the fast linear normal modes in a power series expansion in terms of a small parameter, the Rossby number. Here we use a small modification of the Warn et al. (1995) [3]_ method, that is described in more detail by Eden et al. (2019) [4]_.

    Requirements
    ------------

    Form of the System
    ~~~~~~~~~~~~~~~~~~
    To apply nonlinear normal mode decomposition, we require a system of equations that can be written in spectral space as:

    .. math::
        \partial_t \boldsymbol z = -i \mathbf A \cdot \boldsymbol z + \epsilon \boldsymbol N (\boldsymbol z)

    where :math:`\boldsymbol z(\boldsymbol k, t)` is the state vector, :math:`\boldsymbol k` is the wave vector, :math:`t` is the time, :math:`\mathbf A` is a matrix, :math:`\epsilon` is a small parameter and :math:`\boldsymbol N(\boldsymbol z)`  is a nonlinear term. The :math:`j`-th. component of the nonlinear term is given by:
    
    .. math::
        N_j(\boldsymbol z) = \boldsymbol z * (\mathbf G_j \cdot \boldsymbol z)
    
    where the star :math:`*` denotes a convolution, and :math:`\mathbf G_j` is a matrix. The nonlinear term :math:`\boldsymbol N` typically corresponds to the advection term, and :math:`\mathbf G_j \cdot \boldsymbol z` to the gradient of the :math:`j`-th component of :math:`\boldsymbol z`. 

    Scaling of the system
    ~~~~~~~~~~~~~~~~~~~~~
    The system must be scaled, e.g. the magnitude of :math:`\mathbf A`, and of the nonlinear term :math:`\boldsymbol N` should be of order one. 

    Eigenvectors and Eigenvalues
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The eigenvectors :math:`\boldsymbol q_j` and eigenvalues :math:`\lambda_j` of the linear system matrix :math:`\mathbf A` should be known. Further, projection vectors :math:`\boldsymbol p_j:math:` are required, such that
    
    .. math::
        \boldsymbol p_i \cdot \boldsymbol q_j = \delta_{i,j}
    
    with the Kronecker-Delta symbol :math:`\delta_{i,j}`. The first eigenvalue :math:`\lambda_0` must be zero.


    Calculation of the balanced state
    ---------------------------------
    In this section we present the equations from which the balance state is computed. For more details on how these equations are derived and why the obtained state is balanced see the derivation section.

    Nonlinear normal mode decomposition is based around the idea on expanding a state around a small parameter :math:`\epsilon` and balancing the system of equations in each order of :math:`\epsilon`. The full balanced state :math:`\boldsymbol z_b` up to order :math:`N` is then given by:
    
    .. math::
        \boldsymbol z_b = z_{0,0,0}~\boldsymbol q_0 + \sum_{n=1}^N \epsilon^n \sum_{j={1,2}} z_{j,n,0}~\boldsymbol q_j 
    
    where :math:`N` is the maximum order to which the state is balanced, and :math:`z_{j,n,k}` are amplitudes, where
    - :math:`j` correspond to the :math:`j`-th linear normal mode
    - :math:`n` correspond to the :math:`n`-th order of the power series expansion
    - :math:`k` correspond to the :math:`k`-th partial time derivative
    For the slow linear normal modes (:math:`j=0`), these amplitudes are given by:
    
    .. math::
        z_{0,0,k} = \begin{cases}
        \boldsymbol p_0 \cdot \boldsymbol z & \text{for} & k=0\\ 
        \boldsymbol p_0 \cdot \boldsymbol I_0^{(k-1)} &\text{else}
        \end{cases}
    
    where :math:`\boldsymbol z` is the state to be balanced, and :math:`\boldsymbol I_n^{(k)}` are interaction terms that are given below. For the two fast linear normal modes :math:`j=1,2`, the amplitudes are given by:
    
    .. math::
        z_{j,n,k} = \begin{cases}
        0 & \text{for} & n=0\\ 
        \frac{i}{\lambda_j} \left( z_{j,n-1,k+1} - \boldsymbol p_j \cdot \boldsymbol I_{n-1}^{(k)}\right) &\text{else}
        \end{cases}
    
    For :math:`n=0`, the interaction terms :math:`\boldsymbol I_n^{(k)}` are given by:
    
    .. math::
        \boldsymbol I_0^{(k)} = 
        \sum_{m=0}^{k} \binom{k}{m} \boldsymbol S \left(
        z_{0,0,k-m} ~ \boldsymbol q_0 , 
        z_{0,0,m} ~ \boldsymbol q_0 \right)
   
    and for :math:`n>0`:
    
    .. math::
        \begin{eqnarray}
            \boldsymbol I_n^k = 
            \sum_{m=0}^k \begin{pmatrix}k \\ m \end{pmatrix} \left [
            2 \boldsymbol S \left(
            z_{0,0,k-m} ~ \boldsymbol q_0,
            \sum_{j=1,2} z_{j,n,m} ~ \boldsymbol q_j \right) \right.
            \\ \left. 
            + \sum_{i=1}^{n} \boldsymbol S \left(
            \sum_{j=1,2} z_{j,i,k-m} ~ \boldsymbol q_j ,
            \sum_{j=1,2} z_{j,n-i,m} ~ \boldsymbol q_j \right)
            \right]
        \end{eqnarray}
    
    where :math:`\binom{k}{m}` are binomial coefficients and :math:`\boldsymbol S` is a symmetrical bilinear form, defined as:
    
    .. math::
        \boldsymbol S(\boldsymbol z_1 , \boldsymbol z_2) = 
        \frac{1}{2} \left [
        \boldsymbol N (\boldsymbol z_1 + \boldsymbol z_2) -
        \boldsymbol N (\boldsymbol z_1) 
        - \boldsymbol N(\boldsymbol z_2)
        \right]
    
    Note that the amplitudes :math:`z_{j,n,k}` can be straightforward calculated from these equations, since :math:`z_{j,n,k}` only depends on terms of order :math:`n-1` and smaller, and :math:`z_{j,0,k}` can be calculated from terms with time derivative order :math:`k'` smaller than :math:`k`. 


    Derivation
    ----------
    We start with the spectral system of equations and multiply it with the projection vector :math:`\boldsymbol p_j` from the left:
    
    .. math::
        \partial_t z_0 = \epsilon ~\boldsymbol p_0 \cdot \boldsymbol N(\boldsymbol z)
        \quad , \quad 
        \partial_t z_j = -i \lambda_j z_j + \epsilon ~\boldsymbol p_j \cdot \boldsymbol N(\boldsymbol z)
   
    where :math:`z_j` is amplitude of the :math:`j`-th normal mode:
    
    .. math::
        z_j \equiv \boldsymbol p_j \cdot \boldsymbol z
    
    Note that we consider the zero mode :math:`j=0` separately, since the eigenvalue is zero. For a balanced state :math:`\boldsymbol z_b`, the tendency of the state should be small:
    
    .. math::
        \frac{\mathcal O(\partial_t \boldsymbol z_b)}{\mathcal O(\boldsymbol z_b)} = \epsilon 
    
    Introducing a slow time scale :math:`T=\epsilon t`, the balance condition becomes:
    
    .. math::
        \partial_t = \epsilon \partial_T
        \quad \implies \quad 
        \frac{\mathcal O(\partial_T \boldsymbol z)}{\mathcal O(\boldsymbol z)} = 1
    
    which is easier to handle. In the slow time scale, the system of equations becomes:
    
    .. math::
        \partial_T z_0 = \boldsymbol N (\boldsymbol z)
        \quad , \quad 
        \epsilon \partial_T z_j = -i \lambda_j z_j + \epsilon \boldsymbol N(\boldsymbol z)
    
    these equations will hereafter be referred to as the slow time scale tendency equations. While these equations still represent the full system of equations and thus, the solution space is not yet limited to balanced states. However, together with the balance condition above, the solution space is restricted to slowly evolving states, i.e. balanced states. However to find the balanced state :math:`\boldsymbol z_b` of a given state :math:`\boldsymbol z`, we need a boundary condition for solving these equations. Such a boundary condition could for example be :math:`\boldsymbol z - \boldsymbol z_b` is minimal. However, this boundary condition would be hard to work with. A simpler boundary condition is to take the linear slow mode as a base point coordinate, i.e.:
    
    .. math::
        \boldsymbol p_0 \cdot \boldsymbol z_b = \boldsymbol p_0 \cdot \boldsymbol z = z_0
    
    With this boundary condition, :math:`z_0` is given, and we can solve the above balance equations for :math:`z_j`. To do so, we express the fast modes as a power series expansion:
    
    .. math::
        z_j = \sum_{n=0}^\infty \epsilon^n z_{j,n}
   
    Such that the full balanced state is given by:
    
    .. math::
        \boldsymbol z_b = z_{0}~\boldsymbol q_0 + \sum_{n=0}^\infty \epsilon^n \sum_{j={1,2}} z_{j,n}~\boldsymbol q_j 
    
    We now insert this power series expansion into the slow time scale tendency equations, and sort the terms to the order of :math:`\epsilon`. We then combine the obtained equations with the balance condition by balancing the tendency equation in every order. We will get separate equations for the terms of order :math:`\epsilon^0`, of order :math:`\epsilon^1`, etc. By definition, the balance condition is satisfied when the terms balance in each order. However, before we insert the power series expansion, we need to find a way to deal with the nonlinear term.

    Expansion of the Nonlinear Term
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Using that the :math:`j`-th component of the nonlinear term is given by :math:`\boldsymbol z * (\mathbf G_j \cdot \boldsymbol z)` , we find:
    
    .. math::
        \begin{eqnarray}
	        N_j(\boldsymbol z_1 + \boldsymbol z_2) &=& 
	        \boldsymbol z_1 * (\mathbf G_j \cdot \boldsymbol z_1) + 
	        \boldsymbol z_1 * (\mathbf G_j \cdot \boldsymbol z_2) +
	        \boldsymbol z_2 * (\mathbf G_j \cdot \boldsymbol z_1) +
	        \boldsymbol z_2 * (\mathbf G_j \cdot \boldsymbol z_2)
	        \\
	        &=& 
	        N_j(\boldsymbol z_1) + N_j(\boldsymbol z_2)
	        + 
	        \boldsymbol z_1 * (\mathbf G_j \cdot \boldsymbol z_2) +
	        \boldsymbol z_2 * (\mathbf G_j \cdot \boldsymbol z_1)
	        \\
	        &\equiv& 
	        N_j(\boldsymbol z_1) + N_j(\boldsymbol z_2) +
	        2 S_j (z_1, z_2)
        \end{eqnarray}
   
    with the symmetrical bilinear form :math:`S_j`:
    
    .. math::
        \begin{eqnarray}
	        S_j (z_1, z_2) &=&
	        \frac{1}{2} \left [
	        \boldsymbol z_1 * (\mathbf G_j \cdot \boldsymbol z_2) +
	        \boldsymbol z_2 * (\mathbf G_j \cdot \boldsymbol z_1)
	        \right]
	        \\
	        &=& 
	        \frac{1}{2} \left [
	        N_j(\boldsymbol z_1 + \boldsymbol z_2) -
	        N_j(\boldsymbol z_1) - N_j(\boldsymbol z_2)
	        \right]
        \end{eqnarray}
    
    The first line shows that :math:`S_j` is symmetric and linear in both arguments. The second line shows how to compute :math:`S_j` from :math:`N_j`. Note that the nonlinear term :math:`\boldsymbol N` can be expressed with the symmetrical bilinear form:
    
    .. math::
        \boldsymbol N(\boldsymbol z) = \boldsymbol S(\boldsymbol z, \boldsymbol z)
   
    Inserting the full balanced state and it's power series expansion yields:
    
    .. math::
        \begin{eqnarray}
	        \boldsymbol S(\boldsymbol z_b, \boldsymbol z_b) &=&
	        \boldsymbol S \left( \boldsymbol z_0 + \sum_{n=0}^\infty \epsilon^n \boldsymbol z_{f,n} ~,~
	        \boldsymbol z_0 + \sum_{i=0}^\infty \epsilon^i \boldsymbol z_{f,i}
	        \right)
	        \\ &=&
	        \mathcal S(\boldsymbol z_0, \boldsymbol z_0) 
	        + \sum_i \epsilon^i \boldsymbol S(\boldsymbol z_0, \boldsymbol z_{f,i})
	        + \sum_n \epsilon^n \boldsymbol S(\boldsymbol z_{f,n}, \boldsymbol z_0)
	        + \sum_{n,i} \epsilon^{n+i} \boldsymbol S(\boldsymbol z_{f,n}, \boldsymbol z_{f,i})
	        \\ &=&
	        \mathcal S(\boldsymbol z_0, \boldsymbol z_0) 
	        + \sum_{n=0}^\infty \epsilon^n \left( 2 \boldsymbol S(\boldsymbol z_0 , \boldsymbol z_{f,n}) + \sum_{i=0}^n \boldsymbol S(\boldsymbol z_{f,i}, \boldsymbol z_{f,n-i}) \right)
	        \\ &\equiv&
	        \sum_{n=0}^\infty \epsilon^n \boldsymbol I_n
        \end{eqnarray}
   
    with 
    
    .. math::
        \boldsymbol z_0 \equiv z_0 ~ \boldsymbol q_0
        \quad , \quad
        \boldsymbol z_{f,n} \equiv \sum_{j=1,2} z_{j,n} ~ \boldsymbol q_j
   
    and the interaction term :math:`\boldsymbol I_n`:
    
    .. math::
        \begin{eqnarray}
	        \boldsymbol I_0 &=& \boldsymbol S(\boldsymbol z_0 + \boldsymbol z_{f,0} ~,~ \boldsymbol z_0 + \boldsymbol z_{f,0})
	        \\
	        \boldsymbol I_n &=& 2 \boldsymbol S(\boldsymbol z_0 , \boldsymbol z_{f,n}) + \sum_{i=0}^n \boldsymbol S(\boldsymbol z_{f,i}, \boldsymbol z_{f,n-i})
	        \quad \quad \text{for} \quad n>0
        \end{eqnarray}
    
    Balance in Every Order
    ~~~~~~~~~~~~~~~~~~~~~~
    Inserting the power series in the slow time scale tendency equation of the fast wave modes :math:`j=1,2` yields:
    
    .. math::
        \begin{eqnarray}
	        0 &=& \epsilon ~ \partial_T z_j + i\lambda_j z_j - \epsilon ~ \boldsymbol p_j \cdot \boldsymbol S(\boldsymbol z_b, \boldsymbol z_b)
	        \\ &=&
	        \sum_{n=0}^\infty \epsilon^{n+1} \partial_T z_{j,n} + i \epsilon^n \lambda_j z_{j,n} - \epsilon^{n+1} \boldsymbol p_j \cdot \boldsymbol I_n
	        \\ &=&
	        i\lambda_j z_{j,0} + 
	        \sum_{n=1}^\infty \epsilon^{n} \left( \partial_T z_{j,n-1} + i \lambda_j z_{j,n} - \boldsymbol p_j \cdot \boldsymbol I_{n-1} \right)
        \end{eqnarray}
   
    balancing this in every order yields
    
    .. math::
        \begin{eqnarray}
	        z_{j,0} &=& 0 \\
	        z_{j,n} &=& \frac{i}{\lambda_j} \left( \partial_T z_{j,n-1} - \boldsymbol p_j \cdot \boldsymbol I_{n-1} \right)
	        \quad \quad \text{for} \quad n>0
        \end{eqnarray}
    
    The slow time derivative
    ~~~~~~~~~~~~~~~~~~~~~~~~
    For the calculation of the term :math:`z_{j,n}` we require the slow time derivative of the term :math:`z_{j,n-1}`. We obtain an analytical expression for this term by taking the :math:`k`-th slow time derivative of :math:`z_{j,n}`:
    
    .. math::
        \begin{eqnarray}
	        \partial_T^k z_{j,n} &=& \frac{i}{\lambda_j}(\partial_T^{k+1} z_{j,n-1} - \boldsymbol p_j \cdot \partial_T^k \boldsymbol I_{n-1})
	        \\
	        \Leftrightarrow
	        z_{j,n,k} &=& \frac{i}{\lambda_j}(z_{j,n-1,k+1} - \boldsymbol p_j \cdot \boldsymbol I_{n-1}^{(k)})
        \end{eqnarray}
    
    where the third index denotes the order of the derivative. To calculate the :math:`k`-th derivative of the interaction term :math:`\boldsymbol I_n`, we need to take the derivative of the symmetrical bilinear form:
    
    .. math::
        \begin{eqnarray}
	        \partial_T S_j (\boldsymbol z',\boldsymbol z'') &=& \frac{1}{2}\partial_T \left( \boldsymbol z' * (\mathbf G_j \cdot \boldsymbol z'') + \boldsymbol z'' * (\mathbf G_j \cdot \boldsymbol z') \right) \\
	        &=&
	        \frac{1}{2}\left( 
	        (\partial_T \boldsymbol z') * (\mathbf G_j \cdot \boldsymbol z'') 
	        + \boldsymbol z' * (\mathbf G_j \cdot \partial_T \boldsymbol z'')  
	        + (\partial_T \boldsymbol z'') * (\mathbf G_j \cdot \boldsymbol z')  
	        + \boldsymbol z'' * (\mathbf G_j \cdot \partial_T \boldsymbol z') \right)
	        \\ &=&
	        S_j(\partial_T \boldsymbol z', \boldsymbol z'') + S_j(\boldsymbol z', \partial_T \boldsymbol z'')
        \end{eqnarray}
    
    We search for a formula for the derivative of order :math:`k`. For this we take a look at the second order derivative:
    
    .. math::
        \begin{eqnarray}
	        \partial_T^2 \boldsymbol S(\boldsymbol z', \boldsymbol z'') &=& \partial_T \left[ \boldsymbol S(\partial_T \boldsymbol z', \boldsymbol z'') + \boldsymbol S(\boldsymbol z', \partial_T \boldsymbol z'') \right]
	        \\ &=&
	        \boldsymbol S(\partial_T^2 \boldsymbol z', \boldsymbol z'') + 2\boldsymbol S(\partial_T \boldsymbol z' , \partial_T \boldsymbol z'') + \boldsymbol S(\boldsymbol z', \partial_T^2 \boldsymbol z'')
        \end{eqnarray}
    
    This is the third row of pascal's triangle. For the :math:`k`-th. derivative, we find:
    
    .. math::
        \partial_T^k \boldsymbol S(\boldsymbol z', \boldsymbol z'') = \sum_{m=0}^k \binom{k}{m}
        \boldsymbol S(\partial_T^{k-m} \boldsymbol z', \partial_T^m \boldsymbol z'')
    
    where :math:`\binom{k}{m}` is the binomial coefficient. Using this identity, the :math:`k`-th derivative of the interaction term is given by:
    
    .. math::
        \begin{eqnarray}
	        \boldsymbol I_0^{(k)} &=&
	        \sum_{m=0}^k \binom{k}{m}
	        \boldsymbol S(\boldsymbol z_{0,0,k-m} , \boldsymbol z_{0,0,m})
	        \\
	        \boldsymbol I_n^{(k)} &=& 
	        \sum_{m=0}^k \binom{k}{m} \left[
	        2 \boldsymbol S(\boldsymbol z_{0,0,k-m} , \boldsymbol z_{f,n,m}) + \sum_{i=0}^n \boldsymbol S(\boldsymbol z_{f,i,k-m}, \boldsymbol z_{f,n-i,m})
	        \right]
	        \quad \quad \text{for} \quad n>0
        \end{eqnarray}
    
    with :math:`\boldsymbol z_{0,0,k} = \partial_T^k \boldsymbol z_0`, which we can find an analytical expression for, by evaluating the leading order term of the slow time tendency equation for the slow mode :math:`z_0`:
    
    .. math::
        \partial_T{z_0} =\boldsymbol p_0 \cdot \boldsymbol S(\boldsymbol z_0, \boldsymbol z_0)
        \quad \Rightarrow \quad
        z_{0,0,1} = \boldsymbol p_0 \cdot \boldsymbol I_{0}

    taking the :math:`k`-th derivative yields:
    
    .. math::
        z_{0,0,k} = \boldsymbol p_0 \cdot \boldsymbol I_0^{(k-1)}


    References
    ----------
    .. [1] Machenhauer, B. (1977). On the dynamics of gravity oscillations in a shallow water model, with applications to normal mode initialization. Beitr. Phys. Atmos, 50(1).
    .. [2] Baer, F., & Tribbia, J. J. (1977). On complete filtering of gravity modes through nonlinear initialization. Monthly Weather Review, 105(12), 1536-1539.
    .. [3] Warn, T., Bokhove, O., Shepherd, T.G. and Vallis, G.K. (1995), Rossby number expansions, slaving principles, and balance dynamics. Q.J.R. Meteorol. Soc., 121: 723-739. https://doi.org/10.1002/qj.49712152313
    .. [4] Eden, C., Chouksey, M., & Olbers, D. (2019). Gravity wave emission by shear instability. Journal of Physical Oceanography, 49(9), 2393-2406.
    """
    def __init__(self, 
                 mset: fr.ModelSettingsBase,
                 order=3,
                 epsilon=None,
                 use_model=True,
                 time_step_factor=0.01,
                 use_discrete=True,
                 enable_dealiasing=True) -> None:
        super().__init__(mset)

        ncp = fr.config.ncp

        # compute the eigenvectors
        modes = [0, 1, -1]
        self.q = [mset.grid.vec_q(s=mode, use_discrete=use_discrete) for mode in modes]
        self.p = [mset.grid.vec_p(s=mode, use_discrete=use_discrete) for mode in modes]

        # compute the eigenvalues
        omega = mset.grid.omega(
            k=self.mset.grid.get_mesh(spectral=True),
            use_discrete=use_discrete)
        self.one_over_omega = ncp.where(omega == 0, 0, 1 / omega)

        # set the advection module
        self.advection: fr.modules.advection.AdvectionBase = mset.tendencies.advection

        # set the epsilon
        if epsilon is None:
            # try to get the scaling factor from the advection module and 
            # use it as epsilon
            try:
                epsilon = self.advection.scaling
            except AttributeError:
                raise ValueError("Can't find the scaling factor in the advection module. Please provide epsilon with the keyword arguments.")
        self.epsilon = epsilon

        # set the model
        self.use_model = use_model
        self.model_state = None
        self.subnnmd = None
        self.time_step_factor = time_step_factor
        if use_model and order > 0:
            self.model_state = fr.ModelState(mset)
            self.model_state.dz = mset.state_constructor()
            self.subnnmd = NNMD(
                mset, 
                order=order-1, 
                epsilon=epsilon, 
                use_model=self.use_model,
                use_discrete=use_discrete,
                enable_dealiasing=enable_dealiasing)

        # set other parameters
        self.order = order
        self.enable_dealiasing = enable_dealiasing
        return
    
    def __call__(self, z: fr.StateBase) -> fr.StateBase:
        """
        Project a state to the balanced subspace.
        """
        epsilon = self.epsilon

        # reset the fields
        self.reset_fields()

        # transform to spectral space if necessary
        was_spectral = z.is_spectral
        if not was_spectral:
            z = z.fft()

        # compute the geostrophic mode
        z0 = z @ self.p[0]
        self.fields[0,0,0] = z0

        return self.create_state(self.order, spectral=was_spectral)

    def create_state(self, order, spectral=False):
        """
        Create a state from the balanced state up to a given order.

        Parameters
        ----------
        `order` : `int`
            The order up to which the state is created.
        `spectral` : `bool` (default: False)
            Whether the returned state should be in spectral space.
        """
        epsilon = self.epsilon
        # compute the two wave modes:
        zw1 = sum(epsilon**n * self[1,n,0] for n in range(1, order+1))
        zw2 = sum(epsilon**n * self[2,n,0] for n in range(1, order+1))

        # compute the balanced state
        z_bal = self.fields[0,0,0] * self.q[0] + zw1 * self.q[1] + zw2 * self.q[2]

        # return the balanced state
        return z_bal if spectral else z_bal.ifft()
    
    # ================================================================
    #  The nonlinear interaction terms
    # ================================================================
    def _advect_state(self, z: fr.StateBase) -> fr.StateBase:
        # we need to disable the scaling factor here. We simply do it by dividing by the scaling factor
        dz = self.advection.advect_state(z, self.mset.state_constructor())
        try:
            dz /= self.advection.scaling
        except AttributeError:
            pass # no scaling factor => do nothing
        return dz

    def bilinear_form(self, 
                   z1: fr.StateBase, 
                   z2: fr.StateBase) -> fr.StateBase:
        r"""
        Calculate the symmetrical bilinear form.

        Description
        -----------
        The symmetrical bilinear form is defined as:
        
        .. math::
            \boldsymbol S = \frac{1}{2} \left( \boldsymbol N(\boldsymbol z_1 + \boldsymbol z_2) - \boldsymbol N(\boldsymbol z_1) - \boldsymbol N(\boldsymbol z_2) \right)

        where :math:`\boldsymbol N` is the nonlinear advection term. The interaction term is a symmetric bilinear form.

        Parameters
        ----------
        `z1` : `State`
            The first state (spectral space).
        `z2` : `State`
            The second state (spectral space).

        Returns
        -------
        `S` : `State`
            The interaction term (spectral space).
        """
        # TODO: add dealiasing
        ncp = fr.config.ncp

        # if z1 is z2, we can simplify the calculation
        same_state = True
        for f in z1.fields.keys():
            if not ncp.allclose(z1.fields[f].arr, z2.fields[f].arr):
                same_state = False
                break

        if same_state:
            z1 = z1.ifft()
            bilinear = self._advect_state(z1)
            return bilinear.fft()

        z1 = z1.ifft()
        z2 = z2.ifft()
        bilinear = 0.5 * (self._advect_state(z1 + z2)
                            - self._advect_state(z1)
                            - self._advect_state(z2))
        return bilinear.fft()

    def interaction(self, order_series: int, order_derivative: int) -> fr.StateBase:
        n = order_series
        k = order_derivative

        # create a vector for the interactions
        interactions = self.q[0] * 0

        # the zero-order term
        if n == 0:
            for m in range(k+1):
                coeff = comb(k, m)
                z1 = self[0,0,k-m] * self.q[0]
                z2 = self[0,0,m] * self.q[0]
                interactions += coeff * (self.bilinear_form(z1, z2))

        # the higher-order terms
        else:
            for m in range(k+1):
                coeff = comb(k, m)
                z1 = self[0,0,k-m] * self.q[0]
                z2 = sum(self[j,n,m] * self.q[j] for j in [1,2])
                interactions += 2 * coeff * (self.bilinear_form(z1, z2))

                for i in range(1,n):
                    z1 = sum(self[j,i,k-m] * self.q[j] for j in [1,2])
                    z2 = sum(self[j,n-i,m] * self.q[j] for j in [1,2])
                    interactions += coeff * (self.bilinear_form(z1, z2))

        return interactions

    def reset_fields(self):
        """
        Reset the fields.
        """
        ncp = fr.config.ncp
        self.fields = np.full((3, self.order+1, self.order+1), None, dtype=object)
        # the zero-order terms of the wave modes are zero
        self.fields[1:,0,:] = 0
        return

    # ================================================================
    #  The n-th order terms of k-th order derivative
    # ================================================================
    def __getitem__(self, key):
        # transform the key to mode, order_series, order_derivative
        if not isinstance(key, tuple):
            raise ValueError("Key must be a tuple. Use (mode, order, derivative).")
        mode, order_series, order_derivative = key

        # check if the mode is valid
        if not mode in [0, 1, 2]:
            raise ValueError("Mode must be 0, 1, or 2.")

        # check if the order is zero for mode 0
        if mode == 0 and order_series != 0:
            raise ValueError("Order must be 0 for mode 0.")

        # check if this field has already been computed
        f_ind = (mode, order_series, order_derivative)
        if self.fields[f_ind] is not None:
            return self.fields[f_ind]
        
        # compute the mode 0:
        if mode == 0:
            interaction = self.interaction(order_series=0, order_derivative=order_derivative-1)
            self.fields[f_ind] = interaction @ self.p[0]
            return self.fields[f_ind]
        
        # compute the first derivative with the model
        if self.use_model:
            if order_derivative == 1 and order_series > 1:
                self._derivative_with_model(order_series)
                return self.fields[f_ind]


        interaction = self.interaction(order_series=order_series-1, order_derivative=order_derivative)
        for j, sign in zip([1, 2], [-1, 1]):
            z_prev = self[j, order_series-1, order_derivative+1]
            z_new = 1j * sign * self.one_over_omega * (
                z_prev - interaction @ self.p[j] )
            self.fields[j, order_series, order_derivative] = z_new
        return self.fields[f_ind]

    def _derivative_with_model(self, order_series):
        # construct the state up to the order_series
        z = self.create_state(order_series, spectral=False)

        # calculate the time derivative
        self.model_state.z = z
        dz = self.mset.tendencies.update(self.model_state).dz

        time_step = self.time_step_factor / self.epsilon
        z_next = z + dz * time_step

        # balance the next state
        nnmd = self.subnnmd
        nnmd(z_next)
        # nnmd.reset_fields()
        # nnmd.fields[0,0,0] = z_next.fft() @ self.p[0]
        for j in [1, 2]:
            df = self.fields[j, order_series, 0] - nnmd.fields[j, order_series, 0]
            df *= self.epsilon / time_step
            self.fields[j, order_series, 1] = df
