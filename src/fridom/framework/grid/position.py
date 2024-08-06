from enum import Enum, auto
import fridom.framework as fr


class AxisPosition(Enum):
    """
    The position of a field along an axis on the staggered grid.

    Options
    -------
    `CENTER` :
        Center of the grid cell.
    `FACE` :
        Face of the grid cell (right edge of the cell).

    ::

          CENTER
             ↓
        |    x    |    x    |    x    |
                  ↑
                FACE
        → positive direction
    """
    CENTER = auto()
    FACE = auto()

    def shift(self) -> 'AxisPosition':
        """
        Shift the position of the field. Center -> Face and vice versa

        Returns
        -------
        `AxisPositionNew`
            The new axis position of the field.
        """
        match self:
            case AxisPosition.CENTER:
                return AxisPosition.FACE
            case AxisPosition.FACE:
                return AxisPosition.CENTER


@fr.utils.jaxify
class Position:
    """
    The position of a field on a staggered grid.
    
    Parameters
    ----------
    `positions` : `tuple[AxisPosition]`
        The positions of the field along each axis.
    """
    def __init__(self, positions: tuple[AxisPosition]) -> None:
        self._positions = positions
        return

    def shift(self, axis: int) -> 'Position':
        """
        Shift the position of the field along an axis.

        The position of the field along the specified axis is shifted from
        center to face or vice versa.
        
        Parameters
        ----------
        `axis` : `int`
            The axis along which to shift the field.
        
        Returns
        -------
        `Position`
            The new position of the field.
        """
        new_positions = list(self._positions)
        new_positions[axis] = new_positions[axis].shift()
        return Position(tuple(new_positions))

    # ----------------------------------------------------------------
    #  Overloaded operators
    # ----------------------------------------------------------------

    def __hash__(self) -> int:
        return hash(self._positions)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Position):
            return False

        for my_pos, other_pos in zip(self.positions, value.positions):
            if my_pos != other_pos:
                return False
        return True

    def __getitem__(self, key: int) -> AxisPosition:
        return self.positions[key]

    def __setitem__(self, key: int, value: AxisPosition) -> None:
        positions = list(self._positions)
        positions[key] = value
        self._positions = tuple(positions)
        return

    def __repr__(self) -> str:
        return f"Position: {self._positions}"

    # ----------------------------------------------------------------
    #  Properties
    # ----------------------------------------------------------------
    @property
    def positions(self) -> tuple[AxisPosition]:
        """
        The positions of the field along each axis.
        """
        return self._positions
