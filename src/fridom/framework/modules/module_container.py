"""A module container that can hold multiple modules."""
from __future__ import annotations

from functools import partial
from typing import Iterator, Literal

import fridom.framework as fr


@partial(fr.utils.jaxify, dynamic=("module_list", ))
class ModuleContainer(fr.modules.Module):

    """
    A module container that can hold multiple modules.

    Description
    -----------
    A module container holds a list of modules and is a module itself. It can
    start, stop, and update all the modules it contains.

    Parameters
    ----------
    name : str
        Name of the module container.
    module_list : list
        A list of modules to be added to the container.

    """

    def __init__(self,
                 name: str = "Module Container",
                 module_list: list | None = None) -> None:
        super().__init__()
        self.name = name
        self.module_list = module_list or []

    def __iter__(self) -> Iterator[fr.modules.Module]:
        """Iterate over the modules in the container."""
        return iter(self.module_list)

    @fr.modules.module_method
    def setup(self,
              mset: fr.ModelSettingsBase,
              setup_mode: Literal["default", "forced"] = "default",
              ) -> None:
        """Set all modules up."""
        super().setup(mset, setup_mode=setup_mode)
        _ = [module.setup(mset=mset, setup_mode=setup_mode) for module in self]

    @fr.modules.module_method
    def start(self) -> None:
        """Start all modules."""
        _ = [module.start() for module in self]

    @fr.modules.module_method
    def stop(self) -> None:
        """Stop all modules."""
        _ = [module.stop() for module in self]

    def reset(self) -> None:
        """Reset all modules."""
        _ = [module.reset() for module in self]

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        """Update all modules."""
        for module in self:
            mz = module.update(mz=mz)
        return mz

    def add_module(self, module: fr.modules.Module) -> None:
        """
        Add a module to the end of the module list.

        Parameters
        ----------
        module : Module
            The module to be added to the list.

        """
        self.module_list.append(module)
        self._setup_new_module(module)

    def _setup_new_module(self, module: fr.modules.Module) -> None:
        if self.is_setup and module.required_halo > self.grid.halo:
            # force a new setup of the model settings
            self.mset.setup(setup_mode="forced")
            return
        if self.is_setup:
            # Otherwise we only need to setup the individual module
            module.setup(mset=self.mset)

    def get(self, name: str) -> list[fr.modules.Module]:
        """
        Get a module by name.

        Parameters
        ----------
        name : str
            Name of the module.

        Returns
        -------
        list[Module]
            List of modules with the given name. If no module is found, an
            empty list is returned. If multiple modules are found, all of them
            are returned.

        """
        return [module for module in self.module_list if module.name == name]

    @property
    def required_halo(self) -> int:
        """The maximum required halo points of all modules."""
        if self._required_halo is not None:
            return self._required_halo
        return max([module.required_halo for module in self] + [0])

    @property
    def mpi_available(self) -> bool:
        """Whether all modules are available in MPI mode."""
        return all(module.mpi_available for module in self)

    @mpi_available.setter
    def mpi_available(self, value: bool) -> None:
        pass  # do nothing

    def __repr__(self) -> str:
        """Format the module container as a string."""
        # format the title into a 48 character string of format
        res = f"{self.name}"
        for module in self:
            res += f"\n## {module}"
        return res
