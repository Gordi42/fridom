# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.modules.module import \
    Module, start_module, update_module
# Import type information
if TYPE_CHECKING:
    import numpy as np
    from fridom.framework.field_variable import FieldVariable
    from fridom.framework.model_state import ModelState
    from fridom.framework.state_base import StateBase

class BoundaryConditions(Module):
    """
    Set boundary conditions to the model state.
    
    Description
    -----------
    By default, all boundary conditions are set to zero. The user can set 
    custom boundary conditions for each field, at each axis and side. The
    boundary condition can be a constant value or a FieldVariable itself.
    
    Parameters
    ----------
    `field_names` : `list[str]`
        List of field names for which the boundary conditions should be set.
    `boundary_conditions` : `list[dict] | None`
        List of dictionaries with the following keys:
        - "field": name of the field
        - "axis": axis along which the boundary condition should be applied
        - "side": side of the axis along which the boundary condition should be
          applied
        - "value": value of the boundary condition
    `name` : `str` (default="BoundaryConditions")
        Name of the module.
    
    Methods
    -------
    `set_boundary_condition(field, axis, side, value)`
        Set a boundary condition for a field.
    
    Examples
    --------
    >>> # import modules
    >>> import fridom.nonhydro as nh
    >>> ncp = nh.config.ncp
    >>> # Create a grid and model settings
    >>> grid = nh.grid.CartesianGrid(N=[64]*3, L=[2*ncp.pi]*3)
    >>> mset = nh.ModelSettings(grid)
    >>> mset.setup()
    >>> # Set constant buoyancy boundary condition at the top
    >>> mset.bc.set_boundary_condition("b", 1, "left", 1.0)
    >>> # Set cosine velocity boundary condition at the right side in x-direction
    >>> u_bc = nh.FieldVariable(mset, topo=[False, True, False])
    >>> y = grid.X[1][0, :, 0]
    >>> u_bc[:] = ncp.cos(y)
    >>> mset.bc.set_boundary_condition("u", 0, "right", u_bc)
    """
    def __init__(self, 
                 field_names: list[str],
                 boundary_conditions: list[dict] | None = None,
                 name="BoundaryConditions"):
        super().__init__(name)
        # ----------------------------------------------------------------
        #  Sett attributes
        # ----------------------------------------------------------------
        self._fields = field_names
        self._all_boundary_conditions = {}
        self._initial_boundary_conditions = boundary_conditions or []
        return

    @start_module
    def start(self):
        all_boundary_conditions = {}
        for field in self._fields:
            all_boundary_conditions[field] = {}
            for axis in range(getattr(self.grid, "n_dims")):
                all_boundary_conditions[field][axis] = {}
                for side in ["left", "right"]:
                    # by default set all boundary conditions to zero
                    all_boundary_conditions[field][axis][side] = 0

        for bc in self._initial_boundary_conditions:
            field = bc["field"]
            axis  = bc["axis"]
            side  = bc["side"]
            value = bc["value"]
            all_boundary_conditions[field][axis][side] = value
        self._all_boundary_conditions = all_boundary_conditions
        return

    @update_module
    def update(self, mz: 'ModelState', dz: 'StateBase'):
        # first loop over all fields
        for field in self._fields:
            f = getattr(mz.z, field)  # get the field
            field_bcs = self._all_boundary_conditions[field]
            for axis in range(getattr(self.grid, "n_dims")):
                if getattr(self.grid, "periodic_bounds")[axis]:
                    continue  # periodic boundaries do not need to be set
                axis_bcs = field_bcs[axis]
                for side in ["left", "right"]:
                    f.apply_boundary_conditions(axis, side, axis_bcs[side])
        return
        
    def set_boundary_condition(self, field: 'FieldVariable', 
                               axis: int, side: str, 
                               value: 'float | np.ndarray | FieldVariable'):
        """
        Set a boundary condition for a field.
        
        Parameters
        ----------
        `field` : `str`
            The name of the field. The boundary condition will be applied to
            z.field.
        `axis` : `int`
            The axis along which the boundary condition should be applied.
        `side` : `str`
            The side of the axis along which the boundary condition should be
            applied. Can be either "left" or "right".
        `value` : `float | np.ndarray | FieldVariable`
            The value of the boundary condition. If a float is provided, the
            boundary condition will be set to a constant value. If an array is
            provided, the boundary condition will be set to the array.
        
        Examples
        --------
        >>> imoprt fridom.nonhydro as nh
        >>> # Create a grid and model settings
        >>> grid = nh.grid.CartesianGrid(N=[64]*3, L=[1.0]*3)
        >>> mset = nh.ModelSettings(grid)
        >>> # Add boundary conditions for the buoyancy at the top
        >>> mset.bc.set_boundary_condition("b", 1, "left", 1.0)

        """
        # first check if the grid is already set and if so, set the boundary
        # condition directly, otherwise store it in the initial bc list
        if getattr(self, "grid") is None:
            bc = {"field": field, "axis": axis, "side": side, "value": value}
            self._initial_boundary_conditions.append(bc)
        else:
            self._all_boundary_conditions[field][axis][side] = value
        return