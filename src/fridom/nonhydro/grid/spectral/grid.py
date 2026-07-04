from __future__ import annotations

from typing import TYPE_CHECKING

import fridom.framework as fr
import fridom.nonhydro as nh

if TYPE_CHECKING:
    from numpy import ndarray


@fr.utils.jaxify
class Grid(fr.grid.spectral.Grid):

    def __init__(self, shape: list[int], domain_size: list[int], *args: any, **kwargs: dict) -> None:  # noqa: ARG002 (interface conformity)

        if "periodic_bounds" in kwargs and not all(kwargs["periodic_bounds"]):
            msg = "Only periodic boundaries are supported in the spectral grid."
            raise ValueError(msg)

        super().__init__(shape=shape, domain_size=domain_size, periodic_bounds=[True, True, True])

    def omega(self,
              k: tuple[float] | tuple[ndarray],
              use_discrete: bool = False,  # noqa: ARG002 (interface conformity)
              ) -> ndarray:
        # always use non-discrete eigenvalues even if use_discrete is True
        return nh.grid.cartesian.eigenvectors.omega(
            mset=self.mset, s=1, k=k, use_discrete=False)

    def vec_q(self, s: int, use_discrete=False) -> nh.State:  # noqa: ARG002 (interface conformity)
        # always use non-discrete eigenvectors even if use_discrete is True
        return nh.grid.cartesian.eigenvectors.vec_q(
            mset=self.mset, s=s, use_discrete=False)

    def vec_p(self, s: int, use_discrete=False) -> nh.State:  # noqa: ARG002 (interface conformity)
        # always use non-discrete eigenvectors even if use_discrete is True
        return nh.grid.cartesian.eigenvectors.vec_p(
            mset=self.mset, s=s, use_discrete=False)
