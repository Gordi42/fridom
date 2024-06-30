# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.modules.boundary_conditions import \
    BoundaryConditions as BoundaryConditionsBase

class BoundaryConditions(BoundaryConditionsBase):
    """
    Set boundary conditions to the model state.
    
    Description
    -----------
    By default, all boundary conditions are set to zero. The user can set 
    custom boundary conditions for each field, at each axis and side. The
    boundary condition can be a constant value or a FieldVariable itself.
    
    Parameters
    ----------
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
    def __init__(self, boundary_conditions: list[dict] = None,
                 name: str = "BoundaryConditions"):
        super().__init__(["u", "v", "w", "b"], boundary_conditions, name)