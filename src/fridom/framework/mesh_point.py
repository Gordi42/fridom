# Import external modules
from enum import Enum, auto

class MeshPoint(Enum):
    """
    Enum class for the different types of mesh points.
    
    Description
    -----------
    In staggered grid methods, variables are stored at different locations in the grid. In particular, a mesh point in a certain direction can be either at the center of the cell or at the edge of the cell. This class provides an enumeration of the different types of mesh points.
    
    Attributes
    ----------
    CENTER : MeshPoint
        Mesh point at the center of the cell.
    EDGE : MeshPoint
        Mesh point at the edge of the cell (between this cell and the next one)
    """
    CENTER = auto()
    EDGE = auto()