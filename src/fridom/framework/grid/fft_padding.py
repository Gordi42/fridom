from enum import Enum, auto

class FFTPadding(Enum):
    r"""
    Zero padding options for the FFT.

    Options:
    --------

    - NOPADDING: No padding.
    - TRIM: Also known as the 2/3 rule.
    - EXTEND: Also known as the 3/2 rule.

    Description:
    ------------
    Let :math:`k_{\text{max}}` be the maximum wavenumber in the original grid and :math:`u(k)` be the field to be fourier transformed. The FFTPadding options modifies the field as follows: 

    Spectral -> Physical:
    - NOPADDING: no modification 
    - TRIM: all wavenumbers :math:`k > 2/3 k_{\text{max}}` are set to zero.
    - EXTEND: extend the field to include all wavenumbers :math:`k < 3/2 k_{\text{max}}` by adding zeros.

    Physical -> Spectral:
    - NOPADDING: no modification
    - TRIM: no modification
    - EXTEND: remove all frequencies :math:`k > k_{\text{max}}` to restore the original shape.
    """
    NOPADDING = auto()
    TRIM = auto()
    EXTEND = auto()