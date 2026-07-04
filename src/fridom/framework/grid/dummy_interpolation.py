"""Dummy interpolation module (identity interpolation)."""
from __future__ import annotations

import fridom.framework as fr


@fr.utils.jaxify
class DummyInterpolation(fr.grid.InterpolationModule):

    r"""Dummy interpolation, where all methods are just the identity."""

    name = "Dummy Interpolation"

    def interpolate(self,
                    f: fr.ScalarField,
                    destination: fr.grid.Position) -> fr.ScalarField:  # noqa: ARG002 (interface conformity)
        """Return the field unchanged (identity interpolation)."""
        return f
