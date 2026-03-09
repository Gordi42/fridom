"""Module that computes the divergence of the tendency of the model."""
import fridom.framework as fr


@fr.utils.jaxify
class TendencyDivergence(fr.modules.Module):

    """
    Compute the divergence of the tendency of the model.

    Description
    -----------
    The divergence of the tendency is required for the pressure solver.
    """

    name = "Tendency Divergence"

    @fr.utils.jaxjit
    def compute_divergence(self, dz: fr.VectorField) -> fr.ScalarField:
        """Compute the divergence of the tendency."""
        return self.diff_module.div((dz.u, dz.v, dz.w))

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.dz = mz.dz.sync()
        mz.z_diag.div.arr = self.compute_divergence(mz.dz).arr
        return mz
