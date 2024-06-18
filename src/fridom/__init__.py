"""
# Framework for Idealized Ocean Models (FRIDOM)

FRIDOM is a modeling framework designed with a singular goal in mind: 
to provide a high-level interface for the development of idealized ocean models. 
FRIDOM leverages the power of CUDA arrays on GPU through CuPy, enabling the 
execution of models at medium resolutions, constrained only by your hardware 
capabilities, right within Jupyter Notebook.

For more information, please visit the project's GitHub repository:
https://github.com/Gordi42/FRIDOM
"""

from . import framework
from . import nonhydro
from . import shallowwater