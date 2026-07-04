"""Main Tendency module for the nonhydrostatic model."""
from __future__ import annotations

import fridom.framework as fr
import fridom.nonhydro as nh


@fr.utils.jaxify
class MainTendency(fr.modules.ModuleContainer):
    #TODO(Silvano): Add a description of the module.

    """Main Tendency module for the nonhydrostatic model."""

    name = "Main Tendencies: Nonhydrostatic Model"
    def __init__(self) -> None:
        mods = nh.modules
        self._sync_module = mods.SyncModule()
        self._reset_tendency = mods.ResetTendency()
        self._linear_tendency = mods.LinearTendency()
        self._tendency_divergence = mods.TendencyDivergence()
        self._advection = mods.advection.CenteredAdvection()
        self._pressure_solver = mods.pressure_solvers.RFFTPressureSolver()
        self._pressure_gradient_tendency = mods.PressureGradientTendency()
        self._additional_modules = []
        self._set_module_list()

        super().__init__(module_list=self.module_list)

    def _on_setup(self) -> None:
        # update the advection module if the grid is spectral
        if type(self.mset.grid) is nh.grid.spectral.Grid:
            self.advection = nh.modules.advection.SpectralAdvection()

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
        module_list.append(self.tendency_divergence)
        module_list.append(self.pressure_solver)
        module_list.append(self.pressure_gradient_tendency)
        self.module_list = module_list

    # ============================================================
    #   PROPERTIES
    # ============================================================

    @property
    def linear_tendency(self) -> nh.modules.LinearTendency:
        """The core linear momentum tendency module."""
        return self._linear_tendency

    @linear_tendency.setter
    def linear_tendency(self, value: nh.modules.Module) -> None:
        self._linear_tendency = value
        self._set_module_list()
        if self.is_setup:
            value.setup(mset=self.mset)

    @property
    def advection(self) -> nh.modules.advection.AdvectionBase:
        """The advection module (nonlinear + linear by backgound state)."""
        if self._advection is None:
            msg = "The advection module is available after the setup."
            raise ValueError(msg)
        return self._advection

    @advection.setter
    def advection(self, value: nh.modules.Module) -> None:
        self._advection = value
        self._set_module_list()
        if self.is_setup:
            value.setup(mset=self.mset)

    @property
    def tendency_divergence(self) -> nh.modules.TendencyDivergence:
        """The divergence of the momentum tendency, for the pressure solver."""
        return self._tendency_divergence

    @tendency_divergence.setter
    def tendency_divergence(self, value: nh.modules.Module) -> None:
        self._tendency_divergence = value
        self._set_module_list()
        if self.is_setup:
            value.setup(mset=self.mset)

    @property
    def pressure_solver(self) -> nh.modules.Module:
        """The pressure solver module."""
        return self._pressure_solver

    @pressure_solver.setter
    def pressure_solver(self, value: nh.modules.Module) -> None:
        self._pressure_solver = value
        self._set_module_list()
        if self.is_setup:
            value.setup(mset=self.mset)

    @property
    def pressure_gradient_tendency(
            self) -> nh.modules.PressureGradientTendency:
        """The pressure gradient tendency module."""
        return self._pressure_gradient_tendency

    @pressure_gradient_tendency.setter
    def pressure_gradient_tendency(self, value: nh.modules.Module) -> None:
        self._pressure_gradient_tendency = value
        self._set_module_list()
        if self.is_setup:
            value.setup(mset=self.mset)
