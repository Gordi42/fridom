"""A WENO advection scheme."""
from __future__ import annotations

import fridom.framework as fr


@fr.utils.jaxify
class WENO(fr.modules.advection.UpwindAdvection):

    r"""
    Weighted Essentially Non-Oscillatory (WENO) advection scheme.

    References
    ----------
    .. [1] S. Mishra, C. Pares-Pulido, and K. G. Pressel, "Arbitrarily
    high-order (weighted) essentially non-oscillatory finite difference
    schemes for anelastic flows on staggered meshes" *Communications in
    Computational Physics*, 2021.

    """

    name = "WENO Advection"

    def __init__(self,
                 order: int = 3,
                 symmetric_inter: fr.grid.InterpolationModule = None,
                 biased_inter: fr.grid.UpwindInterpolation = None,
                 ) -> None:

        biased_inter = biased_inter or fr.grid.cartesian.InterWENO(order=order)
        super().__init__(
            order, symmetric_inter=symmetric_inter, biased_inter=biased_inter)
