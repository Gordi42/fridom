"""Module that computes the divergence of the tendency of the model."""
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

    @fr.utils.jaxjit
    def compute_divergence(self, dz: fr.VectorField) -> fr.ScalarField:
        """Compute the divergence of the tendency."""
        # we have to apply the water_mask to u, v, and w
        for f in (dz.u, dz.v, dz.w):
            mask = self._water_mask.get_mask(f.position)
            f.arr = f.arr * mask

        return self.diff_module.div((dz.u, dz.v, dz.w))


    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.z_diag.div.arr = self.compute_divergence(mz.dz).arr
        return mz
