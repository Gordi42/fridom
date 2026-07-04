import fridom.framework as fr


@fr.utils.jaxify
class DummyInterpolation(fr.grid.InterpolationModule):

    r"""Dummy interpolation, where all interpolation methods are just the identity."""

    name = "Dummy Interpolation"

    def interpolate(self,
                    f: fr.ScalarField,
                    destination: fr.grid.Position) -> fr.ScalarField:
        return f
