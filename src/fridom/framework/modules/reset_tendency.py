import fridom.framework as fr


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
    def __init__(self, name="Reset Tendency"):
        super().__init__(name=name)

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.dz *= 0
        return mz