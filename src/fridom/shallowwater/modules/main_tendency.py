import fridom.framework as fr
import fridom.shallowwater as sw


class MainTendency(fr.modules.ModuleContainer):
    r"""
    Container for the main tendency modules of the shallow water model.

    The main tendency of the shallow water model computes the tendency terms
    in the following order:

    .. math::
        \partial_t \boldsymbol{u} = \text{Linear} + \text{Advection} + \text{Additional}
    
    with the default modules being:
    - `linear_tendency`: :py:class:`LinearTendency <fridom.shallowwater.modules.LinearTendency>`
    - `advection`: :py:class:`SadournyAdvection <fridom.shallowwater.modules.advection.SadournyAdvection>`
    """
    name = "Main Tendencies: Shallow Water Model"
    def __init__(self):
        mods = sw.modules
        self._reset_tendency = mods.ResetTendency()
        self._linear_tendency = mods.LinearTendency()
        self._advection = mods.SadournyAdvection()
        self._additional_modules = []
        self._set_module_list()

        super().__init__(module_list=self.module_list)
        return

    def add_module(self, module):
        self._additional_modules.append(module)
        self._set_module_list()
        return

    def _set_module_list(self):
        """
        Set the module list.
        
        Description
        -----------
        This function make sure that the pressure solver and the pressure
        gradient tendency are always in the last two positions.
        """
        module_list = []
        module_list.append(self._reset_tendency)
        module_list.append(self.linear_tendency)
        module_list.append(self.advection)
        module_list += self._additional_modules
        self.module_list = module_list
        return

    # ============================================================
    #   PROPERTIES
    # ============================================================

    @property
    def linear_tendency(self):
        """The core linear momentum tendency module."""
        return self._linear_tendency
    
    @linear_tendency.setter
    def linear_tendency(self, value):
        self._linear_tendency = value
        return
    
    @property
    def advection(self):
        """The advection module (nonlinear + linear by backgound state)."""
        return self._advection
    
    @advection.setter
    def advection(self, value):
        self._advection = value
        return
