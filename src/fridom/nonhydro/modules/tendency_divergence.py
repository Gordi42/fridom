import fridom.framework as fr


class TendencyDivergence(fr.modules.Module):
    """
    This class computes the divergence of the tendency of the model.
    The divergence of the tendency is required for the pressure solver.
    """
    def __init__(self):
        super().__init__(name="Tendency Divergence")
        self.required_halo = 1

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        dz = mz.dz
        dz.sync()
        mz.z_diag.div = self.grid.diff_mod.div((dz.u, dz.v, dz.w))
        return mz