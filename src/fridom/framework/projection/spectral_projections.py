import fridom.framework as fr

class GeostrophicSpectral(fr.projection.Projection):
    """
    Projection onto the geostrophic subspace using spectral eigenvectors.

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `use_discrete` : `bool` (default: `True`)
        Whether to use discrete eigenvectors.
    """
    def __init__(self, 
                 mset: fr.ModelSettingsBase,
                 use_discrete: bool = True) -> None:
        super().__init__(mset)
        # Construct the eigenvectors
        self.q = mset.grid.vec_q(s=0, use_discrete=use_discrete)
        self.p = mset.grid.vec_p(s=0, use_discrete=use_discrete)
        return

    def __call__(self, z: fr.StateBase) -> fr.StateBase:
        return z.project(self.p, self.q)

class WaveSpectral(fr.projection.Projection):
    """
    Projection onto the inertia-gravity wave subspace using spectral eigenvectors.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `use_discrete` : `bool` (default: `True`)
        Whether to use discrete eigenvectors.
    
    Methods
    -------
    `__call__(z: State) -> State`
        Project a state to the inertia-gravity wave subspace.
    """
    def __init__(self,
                 mset: fr.ModelSettingsBase,
                 use_discrete: bool = True) -> None:
        super().__init__(mset)
        # use that the projection on the positive and negative eigenspaces
        # are the same. Hence, we only need to construct one of them.
        self.q = mset.grid.vec_q(s=1, use_discrete=use_discrete)
        self.p = mset.grid.vec_p(s=1, use_discrete=use_discrete)
        return

    def __call__(self, z: fr.StateBase) -> fr.StateBase:
        return z.project(self.p, self.q) * 2


class DivergenceSpectral(fr.projection.Projection):
    """
    Projection onto the divergence subspace using spectral eigenvectors.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `use_discrete` : `bool` (default: `True`)
        Whether to use discrete eigenvectors.
    """
    def __init__(self, 
                 mset: fr.ModelSettingsBase,
                 use_discrete: bool = True) -> None:
        super().__init__(mset)
        self.q = mset.grid.vec_q(s="d", use_discrete=use_discrete)
        self.p = mset.grid.vec_p(s="d", use_discrete=use_discrete)
        return

    def __call__(self, z: fr.StateBase) -> fr.StateBase:
        return z.project(self.p, self.q)
