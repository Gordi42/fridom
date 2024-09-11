# Import external modules
import numpy as np
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.grid.transform_type import TransformType

@utils.jaxify
class FFT:
    ...