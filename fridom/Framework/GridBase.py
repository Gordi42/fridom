from fridom.Framework.ModelSettingsBase import ModelSettingsBase

class GridBase:
    """
    Grid container with the meshgrid of the physical and spectral domain.

    Attributes:
        mset (ModelSettings)           : Model settings.
        x (list of 1D arrays)          : Physical domain.
        X (list of 3D arrays)          : Physical domain (meshgrid).
        k (list of 1D arrays)          : Spectral domain.
        K (list of 3D arrays)          : Spectral domain (meshgrid).
    """

    def __init__(self, mset:ModelSettingsBase) -> None:
        """
        Constructor.

        Parameters:
            mset (ModelSettings): Model settings.
        """
        self.mset = mset

        # numpy or cupy
        import numpy
        try:
            import cupy
            self.cp = cupy if mset.gpu else numpy
        except ImportError:
            self.cp = numpy

        # shorthand notation
        n_dims = mset.n_dims
        cp = self.cp; dtype = mset.dtype
        pi2 = 2 * cp.pi
        L = mset.L; N = mset.N; dg = mset.dg

        # physical domain
        self.x = [cp.arange(0, L[i], dg[i], dtype=dtype) for i in range(n_dims)]
        self.X = list(cp.meshgrid(*self.x, indexing='ij'))


        # spectral domain
        k_freq = lambda n, d: cp.fft.fftfreq(n, d/pi2).astype(dtype)
        pad = [1 if mset.periodic_bounds[i] else 2 for i in range(n_dims)]
        self.k = [k_freq(n*p, d) for n, p, d in zip(N, pad, dg)]
        self.K = list(cp.meshgrid(*self.k, indexing='ij'))

        self._cpu = None  # CPU copy of the grid (created on demand)
        return

    @property
    def cpu(self) -> "GridBase":
        """
        Create a copy of the grid on the CPU.
        """
        if self._cpu is None:
            if not self.mset.gpu:
                self._cpu = self
            else:
                mset_cpu = self.mset.copy()
                mset_cpu.gpu = False
                self._cpu = GridBase(mset_cpu)
        return self._cpu

# remove symbols from namespace
del ModelSettingsBase