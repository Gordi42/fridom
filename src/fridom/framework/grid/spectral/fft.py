import fridom.framework as fr
# Import external modules
import numpy as np
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.grid.transform_type import TransformType

@utils.jaxify
class FFT(fr.grid.cartesian.FFT):
    """
    Class for performing fourier transforms on a spectral grid with
    options for zero padding.
        
    Parameters
    ----------
    `periodic` : `tuple[bool]`
        A list of booleans that indicate whether the axis is periodic.
        If True, the axis is periodic, if False, the axis is non-periodic.
    
    Examples
    --------
    .. code-block:: python

        import numpy as np
        from fridom.framework.grid.cartesian import FFT
        fft = FFT(periodic=(True, True, False))
        u = np.random.rand(*(32, 32, 8))
        v = fft.forward(u)
        w = fft.backward(v).real
        assert np.allclose(u, w)
    """
    @partial(utils.jaxjit, static_argnames=['axes', 'transform_types'])
    def forward(self, 
                u: np.ndarray, 
                axes: list[int] | None = None,
                transform_types: tuple[TransformType] | None = None,
                padding = fr.grid.FFTPadding.NOPADDING,
                ) -> np.ndarray:
        """
        Forward transform from physical space to spectral space.
        
        Parameters
        ----------
        `u` : `np.ndarray`
            The array to transform from physical space to spectral space.
        `axes` : `list[int] | None`
            The axes to transform. If None, all axes are transformed.
        `transform_types` : `tuple[TransformType] | None`
            The type of transform to apply for each axis which is not periodic.
        
        Returns
        -------
        `np.ndarray`
            The transformed array in spectral space. If all dimensions are
            periodic, the obtained array is real, else it is complex.
        """
        ncp = config.ncp; scp = config.scp
        # Get the axes to apply fft, dct
        if axes is None:
            fft_axes = self._fft_axes
            dct_axes = self._dct_axes
        else:
            fft_axes = list(set(axes) & set(self._fft_axes))
            dct_axes = list(set(axes) & set(self._dct_axes))

        u_hat = u
        if transform_types is None:
            transform_types = tuple(TransformType.DCT2 for _ in range(u.ndim))

        # discrete cosine transform
        for axis in dct_axes:
            match transform_types[axis]:
                case TransformType.DCT2:
                    if config.backend_is_jax:
                        u_hat = dct_type2(u_hat, axis, u_hat.shape[axis])
                    else:
                        u_hat = scp.fft.dct(u_hat, axis=axis)
                case TransformType.DST1:
                    u_hat = dst_type1(u_hat, axis, u_hat.shape[axis])
                case TransformType.DST2:
                    u_hat = dst_type2(u_hat, axis, u_hat.shape[axis])

        # fourier transform for periodic boundary conditions
        u_hat = ncp.fft.fftn(u_hat, axes=fft_axes)


        return u_hat

    