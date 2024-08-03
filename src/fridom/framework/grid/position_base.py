class PositionBase:
    """
    Base class for field positions on a grid.
    """

    def shift(self, axis: int, direction: str) -> 'PositionBase':
        """
        Shift the position of the field along an axis.
        
        Parameters
        ----------
        `axis` : `int`
            The axis along which to shift the field.
        `direction` : `str`
            The direction in which to shift the field. The direction can be
            either `forward` or `backward`.
        
        Returns
        -------
        `PositionBase`
            The new position of the field.
        """
    