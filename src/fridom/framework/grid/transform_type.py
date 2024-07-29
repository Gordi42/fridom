from enum import Enum, auto

class TransformType(Enum):
    """
    Enum class for the type of transform that should be applied to non-periodic
    axes.

    DCT2: Discrete Cosine Transform of type 2.
    DST1: Discrete Sine Transform of type 1.
    DST2: Discrete Sine Transform of type 2.
    """
    DCT2 = auto()
    DST1 = auto()
    DST2 = auto()