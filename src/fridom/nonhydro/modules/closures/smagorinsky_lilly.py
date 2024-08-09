import fridom.framework as fr
import fridom.nonhydro as nh
import numpy as np


@fr.utils.jaxify
class SmagorinskyLilly(fr.modules.Module):
    r"""
    A Smagorinsky-Lilly closure model for the non-hydrostatic model.

    Description
    -----------
    The Smagorinsky-Lilly model is a subgrid-scale turbulence model proposed
    by Smagorinsky (1963) and Lilly (1962). This implementation is based on
    the implementation in Oceananigans (https://clima.github.io/OceananigansDocumentation/v0.91.5/appendix/library/#Oceananigans.TurbulenceClosures.SmagorinskyLilly-Union{Tuple{},%20Tuple{TD},%20Tuple{TD,%20Any}}%20where%20TD).

    The closure for the velocity field :math:`\boldsymbol{u}` and a scalar
    tracer :math:`\phi` is given by:

    .. math::
        \Delta \boldsymbol{u} = \nabla \cdot \mathbf{\tau}

    .. math::
        \Delta \phi = \nabla \cdot \left( \kappa_t \nabla \phi \right)

    with the stress tensor :math:`\boldsymbol{\tau}` and turbulent diffusivity
    :math:`\kappa_t` given by:

    .. math::
        \mathbf{\tau} = \nu_t \mathbf{\Sigma}

    .. math::
        \nu_t = \nu_s + \nu_{\text{background}}

    .. math::
        \kappa_t = \frac{\nu_s}{\text{Pr}} + \nu_{\text{background}}

    where :math:`\mathbf{\Sigma}` is the strain rate tensor given by:

    .. math::
        \mathbf{\Sigma} = \frac{1}{2} \left( 
            \nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T \right)

    and :math:`\nu_s` is the Smagorinsky viscosity given by:

    .. math::
        \nu_s = \left( C_s \sqrt[3]{\Delta V} \right)^2 
                |\mathbf{\Sigma}| \Gamma(\text{Ri})

    where :math:`C_s` is the Smagorinsky constant, :math:`\Delta V` is the
    grid cell volume, :math:`|\mathbf{\Sigma}|` is the magnitude of the strain
    rate tensor, and :math:`\Gamma(\text{Ri})` is the stratification damping
    factor given by:

    .. math::
        \Gamma(\text{Ri}) = \sqrt{1 - \min(\beta \text{Ri}, 1)}

    where :math:`\beta` is the buoyancy multiplier and :math:`\text{Ri}` is the
    resolved Richardson number given by:

    .. math::
        \text{Ri} = \frac{N^2}{|\mathbf{\Sigma}|}

    where :math:`N^2` is the buoyancy frequency:

    .. math::
        N^2 = \partial_z b + N^2_{\text{background}}

    where :math:`b` is the buoyancy field and :math:`N^2_{\text{background}}`
    is the background buoyancy frequency. The magnitude of the strain rate
    tensor is given by:

    .. math::
        |\mathbf{\Sigma}|^2 = \sum_{i=1}^3 \sum_{j=1}^3 \Sigma_{ij}^2
    
    Parameters
    ----------
    `background_viscosity` : `float`, (default=1.05e-6)
        The background viscosity for velocity fields.
    `background_diffusivity` : `float`, (default=1.46e-7)
        The background diffusivity for tracer fields.
    `turbulent_prandtl_number` : `float`, (default=1.0)
        The turbulent Prandtl number.
    `smagorinsky_constant` : `float`, (default=0.16)
        The Smagorinsky constant.
    `buoyancy_multiplier` : `float | None`, (default=None)
        The buoyancy multiplier. If None, the buoyancy multiplier is set to
        :math:`1 / \text{turbulent_prandtl_number}`.
    """
    name = "Smagorinsky-Lilly"
    def __init__(self, 
                 background_viscosity: float = 1.05e-6,
                 background_diffusivity: float = 1.46e-7,
                 turbulent_prandtl_number: float = 1.0,
                 smagorinsky_constant: float = 0.16,
                 buoyancy_multiplier: float | None = None
                 ):
        super().__init__()
        self.background_viscosity = background_viscosity
        self.background_diffusivity = background_diffusivity
        self.turbulent_prandtl_number = turbulent_prandtl_number
        self.smagorinsky_constant = smagorinsky_constant
        self.buoyancy_multiplier = buoyancy_multiplier or 1 / turbulent_prandtl_number
        return

    @fr.modules.module_method
    def setup(self, mset: 'nh.ModelSettings') -> None:
        super().setup(mset)
        self.filter_width = self.grid.dV**(1/3)
        return

    @fr.utils.jaxjit
    def smagorinsky_lilly_operator(self, z: nh.State, dz: nh.State) -> nh.State:
        self.mset

        diff_mod = self.diff_module
        ncp = fr.config.ncp

        # Compute the velocity gradients
        du = diff_mod.grad(z.u)
        dv = diff_mod.grad(z.v)
        dw = diff_mod.grad(z.w)

        # Compute the strain rate tensor
        s_11 = du[0]; s_12 = 0.5 * (du[1] + dv[0]); s_13 = 0.5 * (du[2] + dw[0])
        s_21 = s_12 ; s_22 = dv[1]                ; s_23 = 0.5 * (dv[2] + dw[1])
        s_31 = s_13 ; s_32 = s_23                 ; s_33 = dw[2]

        # Compute the squared magnitude of the strain rate tensor
        # ignore the different grid positions of each component of s here
        sigma2 = (        (s_11**2 + s_22**2 + s_33**2)
                    + 2 * (s_12**2 + s_13**2 + s_23**2)  ).arr

        # Compute the buoyancy frequency (also ignoring the grid position)
        N2 = (diff_mod.diff(z.b, axis=2) + self.mset.N2).arr

        # Set the buoyancy frequency to zero where it is negative
        N2 = ncp.maximum(N2, 0.0)

        # Compute the resolved Richardson number
        with np.errstate(divide='ignore', invalid='ignore'):
            Ri = N2 / sigma2

        # Compute the stratification damping factor
        gamma = ncp.sqrt(1 - ncp.minimum(self.buoyancy_multiplier * Ri, 1.0))

        # set nan values to 0
        gamma = ncp.nan_to_num(gamma, nan=0)

        # Compute the smagorinsky viscosity
        nu_s = ( (self.smagorinsky_constant * self.filter_width)**2 
                * ncp.sqrt(sigma2) * gamma )
        
        # Compute the turbulent diffusivities
        nu_t = nu_s + self.background_viscosity
        kappa_t = (  nu_s / self.turbulent_prandtl_number
                                  + self.background_diffusivity )

        # Compute the stress tensor
        tau_11 = s_11 * nu_t; tau_12 = s_12 * nu_t; tau_13 = s_13 * nu_t
        tau_21 = tau_12     ; tau_22 = s_22 * nu_t; tau_23 = s_23 * nu_t
        tau_31 = tau_13     ; tau_32 = tau_23     ; tau_33 = s_33 * nu_t

        # Compute the friction terms
        dz.u += diff_mod.div((tau_11, tau_12, tau_13))
        dz.v += diff_mod.div((tau_21, tau_22, tau_23))
        dz.w += diff_mod.div((tau_31, tau_32, tau_33))

        # Compute the mixing terms
        for name, field in z.fields.items():
            if not field.flags["ENABLE_MIXING"]:
                # skip the fields that are not enabled for mixing
                continue

            # Compute the gradient of the field
            df = diff_mod.grad(field)

            # multiply the gradient with the turbulent diffusivity
            df = tuple(d * kappa_t for d in df)

            # Compute the divergence of the gradient
            dz.fields[name] += diff_mod.div(df)

        return dz

    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.dz = self.smagorinsky_lilly_operator(mz.z, mz.dz)
        return mz
