import fridom.framework as fr


@fr.utils.jaxify
class DummyInterpolation(fr.grid.InterpolationModule):
    r"""
    Dummy interpolation, where all interpolation methods are just the identity.
    """
    name = "Dummy Interpolation"

    @fr.utils.jaxjit
    def interpolate(self, 
                    f: fr.FieldVariable,
                    destination: fr.grid.Position) -> fr.FieldVariable:
        return f