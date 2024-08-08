import fridom.framework as fr


@fr.utils.jaxify
class TendencyDivergence(fr.modules.Module):
    """
    This class computes the divergence of the tendency of the model.
    The divergence of the tendency is required for the pressure solver.
    """
    name = "Tendency Divergence"

    @fr.utils.jaxjit
    def compute_divergence(self, dz: fr.StateBase) -> fr.FieldVariable:
        """
        Compute the divergence of the tendency.
        """
        dz.sync()
        div = self.diff_module.div((dz.u, dz.v, dz.w))
        return div

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.z_diag.div.arr = self.compute_divergence(mz.dz).arr
        return mz