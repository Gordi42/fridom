import fridom.framework as fr
from numpy import ndarray
import itertools
from functools import partial


@partial(fr.utils.jaxify, dynamic=('_water_mask', '_cache'))
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
        self._cache = {}
        self._domain_decomposition: fr.domain_decomposition.DomainDecomposition = None
        self._periodic_bounds = None
        return

    def setup(self, mset: fr.ModelSettingsBase) -> None:
        # we can't set mset or grid as attributes due to recursion issues
        # with jaxjit, so we only set the attributes we need
        self._domain_decomposition = mset.grid.domain_decomp
        self._periodic_bounds = mset.grid.periodic_bounds
        self.water_mask = (mset.grid.domain_decomp.create_array(pad=True)+1).astype(bool)
        # self.water_mask = fr.config.ncp.ones(mset.grid.X[0].shape, dtype=bool)
        return

    def get_mask(self, position: fr.grid.Position) -> ndarray:
        """
        Get the water mask at the given position.
        """
        id = hash(position)
        if id not in self._cache:
            self._cache[id] = self.create_mask_at_position(position)
        return self._cache[id]

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
            case fr.grid.AxisPosition.CENTER:
                # nothing to do
                new_mask = mask
            case fr.grid.AxisPosition.FACE:
                # find out left and right side of the mask

                @self._domain_decomposition.shard_map
                def roll(arr):
                    left_side = arr
                    right_side = fr.config.ncp.roll(arr, -1, axis)
                    # both sides must be water (True) for the new mask to be water
                    return right_side * left_side

                new_mask = roll(mask)
        new_mask = self._sync_mask(new_mask)
        return new_mask

    def _sync_mask(self, mask: ndarray) -> ndarray:
        """
        Synchronize the mask across the processors.
        """
        # for some reason, sync does not work on boolean arrays
        # so we convert the mask to integers, sync it and then convert it back
        mask = mask.astype(int)
        mask = self._domain_decomposition.sync(mask)
        mask = mask.astype(bool)
        return mask

    @property
    def water_mask(self) -> ndarray:
        """
        Get the water mask.
        """
        return self._water_mask

    @water_mask.setter
    def water_mask(self, mask: ndarray) -> None:
        mask = self._sync_mask(mask)
        self._water_mask = mask
        # clear the cache
        self._cache = {}
        # construct all possible masks
        ndim = mask.ndim
        CENTER = fr.grid.AxisPosition.CENTER; FACE = fr.grid.AxisPosition.FACE
        for position in itertools.product([CENTER, FACE], repeat=ndim):
            self.get_mask(fr.grid.Position(position))
        return
