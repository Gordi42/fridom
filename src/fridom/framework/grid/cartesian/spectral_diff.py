import fridom.framework as fr
from functools import partial

@fr.utils.jaxify
class SpectralDiff(fr.grid.DiffModule):
    r"""
    Differentiation module in spectral space.
    
    Description
    -----------
    In spectral space, the differentiation of a field is equivalent to a multiplication by the wavenumber and the imaginary unit:

    .. math::
        u = U e^{ikx} \Rightarrow \partial_x u = ik u
    """
    name = "Spectral Difference"

    @fr.modules.module_method
    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        super().setup(mset)
        # check if the grid is either a cartesian grid or a spectral grid
        if not isinstance(self.mset.grid, 
                          (fr.grid.spectral.Grid, fr.grid.cartesian.Grid)):
            raise ValueError("SpectralDiff requires a spectral or cartesian grid")
        return

    @partial(fr.utils.jaxjit, static_argnames=('axis', 'order'))
    def diff(self, 
             f: fr.FieldVariable,
             axis: int,
             order: int = 1) -> fr.FieldVariable:

        # ----------------------------------------------------------------
        #  Transform to spectral space if necessary
        # ----------------------------------------------------------------
        transformed = False
        if not f.is_spectral:
            fr.log.warning("Called diff on a non-spectral field.")
            fr.log.warning("Fourier transforming the field to spectral space, differentiating, and transforming back.")
            f = f.fft()
            transformed = True

        # ----------------------------------------------------------------
        #  Update the type of the boundary conditions
        # ----------------------------------------------------------------
        bc_types = list(f.bc_types)
        if order % 2 == 1:
            match f.bc_types[axis]:
                case fr.grid.BCType.DIRICHLET:
                    bc_types[axis] = fr.grid.BCType.NEUMANN
                case fr.grid.BCType.NEUMANN:
                    bc_types[axis] = fr.grid.BCType.DIRICHLET

        # ----------------------------------------------------------------
        #  Compute the derivative
        # ----------------------------------------------------------------
        res = fr.FieldVariable(**f.get_kw())
        res.bc_types = tuple(bc_types)
        k = self.grid.get_mesh(spectral=True)[axis]
        res.arr = f.arr * (1j * k) ** order

        # ----------------------------------------------------------------
        #  Transform back to physical space if necessary
        # ----------------------------------------------------------------
        if transformed:
            res = res.fft()

        return res