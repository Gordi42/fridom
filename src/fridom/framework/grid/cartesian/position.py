from enum import Enum, auto
from fridom.framework.grid.position_base import PositionBase
from fridom.framework import utils

class AxisOffset(Enum):
    """
    The offset of a field along an axis relative to the grid cell.
    
    Description
    -----------
    - `LEFT` : The field is located at the left edge of the grid cell.
    - `CENTER` : The field is located at the center of the grid cell.
    - `RIGHT` : The field is located at the right edge of the grid cell.
    """
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


class Position(PositionBase):
    """
    The position of a field on a Cartesian grid.
    
    Description
    -----------
    This class represents the position of a field on a Cartesian grid. The
    position is defined by the offset of the field along each axis relative to
    the grid cell. The offset can be one of three values: `LEFT`, `CENTER`, or
    `RIGHT`.
    
    Parameters
    ----------
    `positions` : `tuple[Offset]`
        The offset of the field along each axis.
    """
    _dynamic_attributes = set([])
    def __init__(self, positions: tuple[AxisOffset]) -> None:
        self.positions = positions
        return

    def shift(self, axis: int, direction: str) -> 'Position':
        if direction not in ["forward", "backward"]:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'forward' or 'backward'.")

        positions = list(self.positions)
        old_offset = positions[axis]
        match old_offset:
            case AxisOffset.LEFT:
                if direction == "forward":
                    offset = AxisOffset.CENTER
                else:
                    raise ValueError("Cannot shift field to the left.")
            case AxisOffset.CENTER:
                if direction == "forward":
                    offset = AxisOffset.RIGHT
                else:
                    offset = AxisOffset.LEFT
            case AxisOffset.RIGHT:
                if direction == "forward":
                    raise ValueError("Cannot shift field to the right.")
                else:
                    offset = AxisOffset.CENTER

        positions[axis] = offset
        return Position(tuple(positions))

    def __repr__(self) -> str:
        return f"Position({self.positions})"

utils.jaxify_class(Position)