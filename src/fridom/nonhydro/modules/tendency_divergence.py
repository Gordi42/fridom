"""Module that computes the divergence of the tendency of the model."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr


@partial(fr.utils.jaxify, dynamic=("_water_mask",))
class TendencyDivergence(fr.modules.Module):

    """
    Compute the divergence of the tendency of the model.

    Description
    -----------
    The divergence of the tendency is required for the pressure solver.
    """

    name = "Tendency Divergence"

    def _on_setup(self) -> None:
        super()._on_setup()
        self._water_mask = self.mset.grid.water_mask

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        # we have to apply the water_mask to u, v, and w
        for f in mz.dz.velocity:
            mz.dz[f.name] = self._water_mask.apply_mask(f.sync())
        mz.z_diag.div.arr = self.diff_module.div(mz.dz.velocity).arr
        return mz
