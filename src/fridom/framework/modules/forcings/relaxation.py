import fridom.framework as fr
from functools import partial


@partial(fr.utils.jaxify, dynamic=("target", "domain"))
class Relaxation(fr.modules.Module):
    r"""
    Biharmonic diffusion module

    Description
    -----------
    This module implements the relaxation operator :math:`\mathcal{R}(\phi)`
    for a scalar field :math:`\phi`. The relaxation operator is defined as:
    
    .. math::
        \mathcal{R}(\phi) = \frac{\phi^* - \phi}{\tau} \delta_\Omega

    where :math:`\phi^*` is the target value of the field, :math:`\tau` is the
    relaxation time scale, and :math:`\delta_\Omega` is one on the domain
    :math:`\Omega` and zero elsewhere. At each time step, :math:`\mathcal{R}(\phi)`
    is added to the tendency of the field :math:`\phi`. The analytical solution
    of the relaxation operator with no other forcing terms is:

    .. math::
        \partial_t \phi = \frac{\phi^* - \phi}{\tau}

        \Rightarrow \phi(t) = \phi^* + C e^{-t/\tau}

    with a constant :math:`C`. 

    The relaxation operator can be used to add heating or cooling to
    a fluid, but also for example for wind stress forcing.

    Parameters
    ----------
    `tau` : `float`
        The relaxation time scale :math:`\tau`.
    `field_name` : `str`
        The name of the field that should be relaxed.
    `target` : `float | fr.FieldVariable`
        The target value of the field.
    `domain_function` : `callable`
        A function that takes the mesh as input and returns a boolean array
        that indicates the domain where the relaxation should be applied.
    """
    name = "Relaxation"
    def __init__(self, 
                 tau: float, 
                 field_name: str,
                 target: float | fr.FieldVariable,
                 domain_function: callable):
        super().__init__()
        self.tau = tau
        self.field_name = field_name
        if type(target) is fr.FieldVariable:
            target = target.arr
        self.target = target
        self.domain_function = domain_function
        self.domain = None

    @fr.modules.module_method
    def setup(self, mset: fr.ModelSettingsBase):
        super().setup(mset)
        z = mset.state_constructor()
        mesh = z[self.field_name].get_mesh()
        self.domain = self.domain_function(mesh)
        del z

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.dz = self.relax(mz.z, mz.dz)
        return mz

    @fr.utils.jaxjit
    def relax(self, z: fr.StateBase, dz: fr.StateBase) -> fr.StateBase:
        ncp = fr.config.ncp
        delta = (self.target - z[self.field_name].arr) / self.tau
        dz[self.field_name].arr += ncp.where(self.domain, delta, 0)
        return dz
    