from enum import Enum, auto
import fridom.framework as fr

class AxisPosition(Enum):
    """
    The position of a field along an axis relative to the grid cell.

    Options
    -------
    `LEFT` : 
        The field is located at the left edge of the grid cell.
    `CENTER` :
        The field is located at the center of the grid cell.
    `RIGHT` :
        The field is located at the right edge of the grid cell.
    """
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()

    def shift(self, direction: str) -> 'AxisPosition':
        """
        Shift the position of the field
        
        Parameters
        ----------
        `direction` : `str`
            The direction in which to shift the field. The direction can be
            either `forward` or `backward`.
        
        Returns
        -------
        `AxisPosition`
            The new axis position of the field.
        """
        match self:
            case AxisPosition.LEFT:
                if direction == "forward":
                    return AxisPosition.CENTER
                else:
                    raise ValueError("Cannot shift field to the left.")
            case AxisPosition.CENTER:
                if direction == "forward":
                    return AxisPosition.RIGHT
                else:
                    return AxisPosition.LEFT
            case AxisPosition.RIGHT:
                if direction == "forward":
                    raise ValueError("Cannot shift field to the right.")
                else:
                    return AxisPosition.CENTER


class Position:
    """
    The position of a field on a staggered grid.
    
    Description
    -----------
    This class represents the position of a field on a staggered grid. The
    position is defined by the offset of the field along each axis relative to
    the grid cell. The offset can be one of three values: `LEFT`, `CENTER`, or
    `RIGHT`.
    
    Parameters
    ----------
    `positions` : `tuple[AxisPosition]`
        The offset of the field along each axis.
    """
    _dynamic_attributes = set([])
    def __init__(self, positions: tuple[AxisPosition]) -> None:
        self.positions = positions
        return

    def shift(self, axis: int, direction: str) -> 'Position':
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
        `Position`
            The new position of the field.
        """
        if direction not in ["forward", "backward"]:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'forward' or 'backward'.")

        new_positions = list(self.positions)
        new_positions[axis] = self[axis].shift(direction)
        return Position(tuple(new_positions))

    def __getitem__(self, key: int) -> AxisPosition:
        """
        Get the position of the field along an axis.
        
        Parameters
        ----------
        `key` : `int`
            The axis along which to get the position of the field.
        
        Returns
        -------
        `AxisPosition`
            The position of the field along the axis.
        """
        return self.positions[key]
    
    def __setitem__(self, key: int, value: AxisPosition) -> None:
        positions = list(self.positions)
        positions[key] = value
        self.positions = tuple(positions)
        return

    def __repr__(self) -> str:
        return f"Position({self.positions})"

fr.utils.jaxify_class(Position)