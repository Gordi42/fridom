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
    _dynamic_attributes = set(["mset"])
    def __init__(self, name="Reset Tendency"):
        super().__init__(name=name)

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.dz = self.set_state_to_zero(mz.dz)
        return mz

    @fr.utils.jaxjit
    def set_state_to_zero(self, dz):
        """
        Set the state to zero.
        """
        dz *= 0
        return dz

fr.utils.jaxify_class(ResetTendency)