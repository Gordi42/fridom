import fridom.framework as fr


@fr.utils.jaxify
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
        Harmonic mixing coefficient.
    """
    name = "Harmonic Mixing"
    def __init__(self, 
                 kh: float | fr.FieldVariable):
        diffusion_coefficients = [kh, kh]
        super().__init__(field_flags=["ENABLE_MIXING"],
                         diffusion_coefficients=diffusion_coefficients)
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


@fr.utils.jaxify
class HarmonicFriction(fr.modules.closures.HarmonicDiffusion):
    r"""
    Harmonic friction module

    Description
    -----------
    Applies the harmonic diffusion operator :math:`\mathcal{H}` 
    (see :py:class:`fridom.framework.modules.closures.HarmonicDiffusion`)
    to all fields with the flag "ENABLE_FRICTION" (typically the velocity field).

    Parameters
    ----------
    `ah` : `float | fr.FieldVariable`
        Harmonic friction coefficient (viscosity).
    """
    name = "Harmonic Friction"
    def __init__(self, 
                 ah: float | fr.FieldVariable):
        diffusion_coefficients = [ah, ah]
        super().__init__(field_flags=["ENABLE_FRICTION"],
                         diffusion_coefficients=diffusion_coefficients)
        return

    @property
    def ah(self) -> float | fr.FieldVariable:
        """The horizontal diffusion coefficient."""
        return self.diffusion_coefficients[0]
    
    @ah.setter
    def ah(self, value: float | fr.FieldVariable):
        coeffs = [value, value, self.diffusion_coefficients[2]]
        self.diffusion_coefficients = coeffs
        return


@fr.utils.jaxify
class BiharmonicMixing(fr.modules.closures.BiharmonicDiffusion):
    r"""
    Biharmonic mixing module

    Description
    -----------
    Applies the biharmonic diffusion operator :math:`\mathcal{B}` 
    (see :py:class:`fridom.framework.modules.closures.BiharmonicDiffusion`)
    to all fields with the flag "ENABLE_MIXING".

    Parameters
    ----------
    `kh` : `float | fr.FieldVariable`
        Horizontal mixing coefficient.
    """
    name = "Biharmonic Mixing"
    def __init__(self, 
                 kh: float | fr.FieldVariable):
        diffusion_coefficients = [kh, kh]
        super().__init__(field_flags=["ENABLE_MIXING"],
                         diffusion_coefficients=diffusion_coefficients)
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


@fr.utils.jaxify
class BiharmonicFriction(fr.modules.closures.BiharmonicDiffusion):
    r"""
    Biharmonic friction module

    Description
    -----------
    Applies the harmonic diffusion operator :math:`\mathcal{B}` 
    (see :py:class:`fridom.framework.modules.closures.BiharmonicDiffusion`)
    to all fields with the flag "ENABLE_FRICTION" (typically the velocity field).

    Parameters
    ----------
    `ah` : `float | fr.FieldVariable`
        Horizontal friction coefficient.
    """
    name = "Biharmonic Friction"
    def __init__(self, 
                 ah: float | fr.FieldVariable):
        diffusion_coefficients = [ah, ah]
        super().__init__(field_flags=["ENABLE_FRICTION"],
                         diffusion_coefficients=diffusion_coefficients)
        return

    @property
    def ah(self) -> float | fr.FieldVariable:
        """The horizontal diffusion coefficient."""
        return self.diffusion_coefficients[0]
    
    @ah.setter
    def ah(self, value: float | fr.FieldVariable):
        coeffs = [value, value, self.diffusion_coefficients[2]]
        self.diffusion_coefficients = coeffs
        return
