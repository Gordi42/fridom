"""The main tendency module for the shallow water model."""
from __future__ import annotations

import fridom.framework as fr
import fridom.shallowwater as sw


@fr.utils.jaxify
class MainTendency(fr.modules.ModuleContainer):

    r"""
    Container for the main tendency modules of the shallow water model.

    The main tendency of the shallow water model computes the tendency terms
    in the following order:

    .. math::
        \partial_t \boldsymbol{u} =
            \text{Linear} + \text{Advection} + \text{Additional}

    with the default modules being:
    - `linear_tendency`: :py:class:`LinearTendency
      <fridom.shallowwater.modules.LinearTendency>`
    - `advection`: :py:class:`SadournyAdvection
      <fridom.shallowwater.modules.advection.SadournyAdvection>`
    """

    name = "Main Tendencies: Shallow Water Model"
    def __init__(self) -> None:
        mods = sw.modules
        self._sync_module = mods.SyncModule()
        self._reset_tendency = mods.ResetTendency()
        self._linear_tendency = mods.LinearTendency()
        self._advection = mods.advection.SadournyAdvection()
        self._additional_modules = []
        self._set_module_list()

        super().__init__(module_list=self.module_list)

    def add_module(self, module: fr.modules.Module) -> None:  # noqa: D102
        self._additional_modules.append(module)
        self._set_module_list()
        self._setup_new_module(module)

    def _set_module_list(self) -> None:
        """
        Set the module list.

        Description
        -----------
        This function make sure that the pressure solver and the pressure
        gradient tendency are always in the last two positions.
        """
        module_list = []
        module_list.append(self._sync_module)
        module_list.append(self._reset_tendency)
        module_list.append(self.linear_tendency)
        module_list.append(self.advection)
        module_list += self._additional_modules
        self.module_list = module_list

    # ============================================================
    #   PROPERTIES
    # ============================================================

    @property
    def linear_tendency(self) -> sw.modules.LinearTendency:
        """The core linear momentum tendency module."""
        return self._linear_tendency

    @linear_tendency.setter
    def linear_tendency(self, value: sw.modules.Module) -> None:
        self._linear_tendency = value
        if self.is_setup:
            value.setup(mset=self.mset)
        self._set_module_list()

    @property
    def advection(self) -> fr.modules.advection.AdvectionBase:
        """The advection module (nonlinear + linear by backgound state)."""
        return self._advection

    @advection.setter
    def advection(self, value: fr.modules.Module) -> None:
        self._advection = value
        if self.is_setup:
            value.setup(mset=self.mset)
        self._set_module_list()
