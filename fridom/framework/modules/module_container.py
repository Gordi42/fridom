from .module import Module, start_module, stop_module, update_module
from fridom.framework.state_base import StateBase
from fridom.framework.model_state import ModelStateBase

class ModuleContainer(Module):
    def __init__(self, name="Module Container", module_list: list = None):
        if module_list is None:
            module_list = []
        super().__init__(name=name, module_list=module_list)

    @start_module
    def start(self) -> None:
        """
        # Start all modules.
        ## Args:
        - grid (GridBase): Grid object.
        - timer (TimerBase): Timer object.
        """
        for module in self.module_list:
            module.start(grid=self.grid, timer=self.timer)
        return

    @stop_module
    def stop(self) -> None:
        """
        # Stop all modules.
        """
        for module in self.module_list:
            module.stop()
        return

    @update_module
    def update(self, mz: ModelStateBase, dz: StateBase) -> None:
        """
        # Update all modules.
        ## Args:
        - mz (ModelStateBase): Model state.
        - dz (StateBase): Tendency state.
        """
        for module in self.module_list:
            module.update(mz=mz, dz=dz)
        return

    def add_module(self, module) -> None:
        """
        # Add a module to the list.
        ## Args:
        - module (Module): Module to be added.
        """
        self.module_list.append(module)
        return

    def get(self, name) -> list:
        """
        # Get a module by name.
        ## Args:
        - name (str): Name of the module.
        ## Returns:
        - matches (list): List of modules with the given name.
        """
        matches = []
        for module in self.module_list:
            if module.name == name:
                matches.append(module)
        return matches

    def __repr__(self) -> str:
        """
        # String representation of the module.
        """
        # format the title into a 48 character string of format
        # "==================== TITLE ===================="
        title = self.name.upper()
        title = title.center(len(title)+2).center(48, "=")
        res = f"{title}\n"
        for module in self.module_list:
            res += f"{module}"
        return res