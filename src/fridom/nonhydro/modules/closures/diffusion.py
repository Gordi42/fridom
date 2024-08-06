import fridom.framework as fr

class HarmonicMixing(fr.modules.closures.HarmonicDiffusion):
    r"""
    Harmonic mixing module

    Description
    -----------
    Applies the harmonic diffusion operator :math:`\mathcal{H}` 
    (see :py:class:`fridom.framework.modules.closures.HarmonicDiffusion`)
    to all fields with the flag "ENABLE_MIXING".

    Parameters
    ----------
    `kh` : `float | fr.FieldVariable`
        Horizontal harmonic mixing coefficient.
    `kv` : `float | fr.FieldVariable`
        Vertical harmonic mixing coefficient.
    `diff` : `fr.grid.DiffModule | None`, (default=None)
        Differentiation module to use. If None, the differentiation module of
        the grid is used.
    `interp` : `fr.grid.InterpolationModule | None`, (default=None)
        Interpolation module to use. If None, the interpolation module of
        the grid is used.
    """
    def __init__(self, 
                 kh: float | fr.FieldVariable,
                 kv: float | fr.FieldVariable,
                 diff_module: fr.grid.DiffModule | None = None,
                 interp_module: fr.grid.InterpolationModule | None = None):
        diffusion_coefficients = [kh, kh, kv]
        super().__init__(field_flags=["ENABLE_MIXING"],
                         diffusion_coefficients=diffusion_coefficients,
                         diff_module=diff_module,
                         interp_module=interp_module,
                         name="Harmonic Mixing")
        return

    @property
    def kh(self) -> float | fr.FieldVariable:
        """The horizontal diffusion coefficient."""
        return self.diffusion_coefficients[0]
    
    @kh.setter
    def kh(self, value: float | fr.FieldVariable):
        coeffs = [value, value, self.diffusion_coefficients[2]]
        self.diffusion_coefficients = coeffs
        return

    @property
    def kv(self) -> float | fr.FieldVariable:
        """The vertical diffusion coefficient."""
        return self.diffusion_coefficients[2]
    
    @kv.setter
    def kv(self, value: float | fr.FieldVariable):
        coeffs = [self.diffusion_coefficients[0], self.diffusion_coefficients[1], value]
        self.diffusion_coefficients = coeffs
        return

fr.utils.jaxify_class(HarmonicMixing)