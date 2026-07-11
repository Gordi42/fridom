"""A module that resets the tendency of a model state."""
from __future__ import annotations

import fridom.framework as fr


@fr.utils.jaxify
class ResetTendency(fr.modules.Module):

    """
    A module that resets the tendency of a model state.

    Description
    -----------
    Time steppers may reuse tendency states to avoid unnecessary memory
    deallocation and reallocation. For this reason, it is important to reset
    the tendency state before updating it. It should always be the first module
    of the tendencies list.
    """

    name = "Reset Tendency"

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.dz = mz.dz.set_zero()
        return mz
