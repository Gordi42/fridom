# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from .module import Module, start_module, stop_module, update_module
# Import type information
if TYPE_CHECKING:
    from fridom.framework.state_base import StateBase
    from fridom.framework.model_state import ModelState

class ModuleContainer(Module):
    """
    A module container that can hold multiple modules.
    
    Description
    -----------
    A module container holds a list of modules and is a module itself. It can
    start, stop, and update all the modules it contains.
    
    Parameters
    ----------
    `name` : `str`
        The name of the module container.
    `module_list` : `list`
        A list of modules to be added to the container.

    Flags
    -----
    `mpi_available` : `bool`
        If True, the module is available in MPI mode.
    `required_halo` : `int`
        The number of halo points required by the module.
    
    Methods
    -------
    `start()`
        Start all modules.
    `stop()`
        Stop all modules.
    `update()`
        Update all modules.
    `add_module()`
        Append a module to the list.
    `get()`
        Get a module by name.
    """
    def __init__(self, name="Module Container", module_list: list = None):
        if module_list is None:
            module_list = []
        super().__init__(name=name, module_list=module_list)

    @start_module
    def start(self) -> None:
        """
        Start all modules.
        """
        for module in self.module_list:
            module.start(mset=self.mset, timer=self.timer)
        return

    @stop_module
    def stop(self) -> None:
        """
        Stop all modules.
        """
        for module in self.module_list:
            module.stop()
        return

    @update_module
    def update(self, mz: 'ModelState', dz: 'StateBase') -> None:
        """
        Update all modules.
        """
        for module in self.module_list:
            module.update(mz=mz, dz=dz)
        return

    def add_module(self, module) -> None:
        """
        Add a module to the end of the module list.
        
        Parameters
        ----------
        `module` : `Module`
            The module to be added to the list.
        """
        self.module_list.append(module)
        return

    def get(self, name) -> list:
        """
        Get a module by name.
        
        Parameters
        ----------
        `name` : `str`
            Name of the module.
        
        Returns
        -------
        `list[Module]`
            List of modules with the given name. If no module is found, an
            empty list is returned. If multiple modules are found, all of them
            are returned.
        """
        matches = []
        for module in self.module_list:
            if module.name == name:
                matches.append(module)
        return matches

    @property
    def required_halo(self) -> int:
        """
        The maximum required halo points of all modules.
        """
        # check if module_list is empty
        if not self.module_list:
            return 0
        return max([module.required_halo for module in self.module_list])
    
    @required_halo.setter
    def required_halo(self, value: int) -> None:
        pass  # do nothing
    
    @property
    def mpi_available(self) -> bool:
        """
        Whether all modules are available in MPI mode.
        """
        return all([module.mpi_available for module in self.module_list])

    @mpi_available.setter
    def mpi_available(self, value: bool) -> None:
        pass  # do nothing

    def __repr__(self) -> str:
        """
        String representation of the module.
        """
        # format the title into a 48 character string of format
        # "==================== TITLE ===================="
        title = self.name.upper()
        title = title.center(len(title)+2).center(48, "=")
        res = f"{title}\n"
        for module in self.module_list:
            res += f"{module}"
        return res