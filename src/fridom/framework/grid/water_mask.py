import fridom.framework as fr
from numpy import ndarray


class WaterMask:
    """
    Water mask for the grid cells (for boundary conditions).
    
    Description
    -----------
    Let's consider the following staggered grid with periodic boundaries:

    ::

        -----e---------e---------e-----
        |         |         |         |
        |    x    o    x    o    x    o
        |         |         |         |
        -----e---------e---------e-----
        |         |         |         |
        |   (x)   o   (x)   o    x    o
        |         |         |         |
        -----e---------e---------e-----
        |         |         |         |
        |   (x)   o    x    o    x    o
        |         |         |         |
        -------------------------------

    where `x` represents the cell center and `o` and `e` represent the cell
    faces. Grid cells denoted with `(x)` are the cells with land. The water
    mask is a boolean array that indicates whether a cell is water (1) or
    land (0). For the above grid, the water masks would be:

    ::

            [1, 1, 1]           [1, 1, 1]           [0, 1, 1]
        x = [0, 0, 1]       o = [0, 0, 0]       e = [0, 0, 1]
            [0, 1, 1]           [0, 1, 0]           [0, 0, 1]
    
    """
    def __init__(self):
        self.name = "Water Mask"
        self._water_mask = None
        self._masks_at_faces = {}
        return

    def setup(self, mset: fr.ModelSettingsBase) -> None:
        self.mset = mset
        self.grid = mset.grid
        self.water_mask = fr.config.ncp.ones(self.grid.X[0].shape, dtype=bool)
        return

    def get_mask(self, position: fr.grid.Position) -> ndarray:
        """
        Get the water mask at the given position.
        """
        id = hash(position)
        if id not in self._masks_at_faces:
            self._masks_at_faces[id] = self.create_mask_at_position(position)
        return self._masks_at_faces[id]

    def create_mask_at_position(self, position: fr.grid.Position) -> ndarray:
        """
        Create a water mask at the given position.
        """
        new_mask = self._water_mask
        for axis, axpos in enumerate(position.positions):
            new_mask = self.shift_mask_along_axis(new_mask, axis, axpos)
        return new_mask

    def shift_mask_along_axis(
            self, 
            mask: ndarray,
            axis: int, 
            axpos: fr.grid.AxisPosition) -> ndarray:
        """
        Shift the mask along the given axis to the new position.

        Description
        -----------
        Let's say we have a mask given at the cell centers, denoted with `x`
        below. And we want to shift the mask to the right, denoted with `|`
        below. The overline represents the position of land in the mask.

        ::

            _______           _______
               x  |  x  |  x  |  x  |  x  |  x  |
        
        Hence the water mask would be:

        ::
        
            [0, 1, 1, 0, 1, 1]

        The new mask at the right position is only water if both neighboring
        cells are water. Hence the new mask at the right cell faces would be:

        ::

            [0, 1, 0, 0, 1, ?]

        where `?` depends on the neighboring cells. To find the new mask 
        algorithmically, we first determine the left and right cell centers
        of the corresponding cell face. And then we check if both neighboring
        cells are water by multiplying the left and right cell centers.
        Finally we synchronize the mask across the processors and fill the
        halo cells with land.
        
        Parameters
        ----------
        mask : ndarray
            The mask to shift (located at the center at the given axis)
        axis : int
            The axis along which the mask should be shifted
        axpos : fr.grid.AxisPosition
            The new position of the mask along the axis
        """
        match axpos:
            case fr.grid.AxisPosition.LEFT:
                # find out left and right side of the mask
                right_side = mask
                left_side = fr.config.ncp.roll(right_side, 1, axis)
                # both sides must be water (True) for the new mask to be water
                new_mask = right_side * left_side
            case fr.grid.AxisPosition.CENTER:
                # nothing to do
                new_mask = mask
            case fr.grid.AxisPosition.RIGHT:
                # find out left and right side of the mask
                left_side = mask
                right_side = fr.config.ncp.roll(left_side, -1, axis)
                # both sides must be water (True) for the new mask to be water
                new_mask = right_side * left_side
        new_mask = self.grid.sync(new_mask)
        new_mask = self.fill_halo(new_mask)
        return new_mask

    def fill_halo(self, mask: ndarray) -> ndarray:
        """
        Fill the halo cells with land (0) if the boundary is not periodic.
        """
        grid = self.grid
        subdomain = self.grid.get_subdomain(spectral=False)
        left = slice(0, subdomain.halo)
        right = slice(-subdomain.halo, None)
        for axis in range(grid.n_dims):
            # skip periodic boundaries
            if grid.periodic_bounds[axis]:
                continue
            if subdomain.is_left_edge[axis]:
                left_side = [slice(None)] * grid.n_dims
                left_side[axis] = left
                left_side = tuple(left_side)
                mask = fr.utils.modify_array(mask, left_side, False)
            if subdomain.is_right_edge[axis]:
                right_side = [slice(None)] * grid.n_dims
                right_side[axis] = right
                right_side = tuple(right_side)
                mask = fr.utils.modify_array(mask, right_side, False)
        return mask

    @property
    def water_mask(self) -> ndarray:
        """
        Get the water mask.
        """
        return self._water_mask

    @water_mask.setter
    def water_mask(self, mask: ndarray) -> None:
        mask = self.grid.get_domain_decomposition(spectral=False).sync(mask)
        mask = self.fill_halo(mask)
        self._water_mask = mask
        # clear the cache
        self._masks_at_faces = {}
        return