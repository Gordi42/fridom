import fridom.framework as fr


@fr.utils.jaxify
class SpectralPressureSolver(fr.modules.Module):
    """
    This class solves the pressure field with a spectral solver.
    """
    name = "Spectral Pressure Solver"

    @fr.utils.jaxjit
    def solve_for_pressure(self, div: fr.FieldVariable) -> fr.FieldVariable:
        ps = div.fft() / (-self.grid.k2_hat)
        ps.arr = fr.utils.modify_array(ps.arr, self.grid.k2_hat_zero, 0)
        return ps.fft()


    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.z_diag.p.arr = self.solve_for_pressure(mz.z_diag.div).arr
        return mz

    @property
    def info(self) -> dict:
        res = super().info
        res["Solver"] = "Spectral"
        return res
