import fridom.framework as fr

class ModuleContainer(fr.modules.Module):
    """
    A module container that can hold multiple modules.
    
    Description
    -----------
    A module container holds a list of modules and is a module itself. It can
    start, stop, and update all the modules it contains.
    
    Parameters
    ----------
    `name` : `str`
        Name of the module container.
    `module_list` : `list`
        A list of modules to be added to the container.
    """
    def __init__(self, 
                 name: str = "Module Container",
                 module_list: list | None = None):
        super().__init__()
        self.name = name
        self.module_list = module_list or []

    @fr.modules.module_method
    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        """
        Setup all modules.
        """
        super().setup(mset)
        for module in self.module_list:
            module.setup(mset=mset)
        return

    @fr.modules.module_method
    def start(self) -> None:
        """
        Start all modules.
        """
        for module in self.module_list:
            module.start()
        return

    @fr.modules.module_method
    def stop(self) -> None:
        """
        Stop all modules.
        """
        for module in self.module_list:
            module.stop()
        return

    def reset(self) -> None:
        """
        Reset all modules.
        """
        for module in self.module_list:
            module.reset()
        return

    @fr.modules.module_method
    def update(self, mz: 'fr.ModelState') -> 'fr.ModelState':
        """
        Update all modules.
        """
        for module in self.module_list:
            mz = module.update(mz=mz)
        return mz

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
        if self._required_halo is not None:
            return self._required_halo
        # check if module_list is empty
        if not self.module_list:
            return 0
        return max([module.required_halo for module in self.module_list])
    
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
        # title = self.name.upper()
        # title = title.center(len(title)+2).center(48, "=")
        res = f"{self.name}"
        for module in self.module_list:
            res += f"\n## {module}"
        return res