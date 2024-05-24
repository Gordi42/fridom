from fridom.NonHydrostatic.Grid import Grid
from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.Framework.TimingModule import TimingModule
from fridom.NonHydrostatic.State import State

from fridom.NonHydrostatic.Modules.Interpolation \
    .InterpolationModule import InterpolationModule, InterpolationConstructor
from fridom.NonHydrostatic.Modules.Interpolation \
    .LinearInterpolation import LinearInterpolationConstructor as LIC
from fridom.NonHydrostatic.Modules.Advection \
    .AdvectionModule import AdvectionModule, AdvectionConstructor

class CenteredAdvection(AdvectionModule):
    """
    Centered advection scheme.
    The advection term is computed by a centered finite
    difference scheme calculating the advection term as the flux divergence.
    """
    def __init__(self, 
                 grid: Grid,
                 timer: TimingModule,
                 interpolation: InterpolationModule):
        super().__init__(grid, timer)
        self.interpolation = interpolation

        mset = grid.mset
        # grid spacing
        self.dx1 = mset.dtype(1.0) / mset.dx
        self.dy1 = mset.dtype(1.0) / mset.dy
        self.dz1 = mset.dtype(1.0) / mset.dz

        # boundary conditions
        self.bcx = "wrap" if self.mset.periodic_bounds[0] else "constant"
        self.bcy = "wrap" if self.mset.periodic_bounds[1] else "constant"
        self.bcz = "wrap" if self.mset.periodic_bounds[2] else "constant"

    def compute_advection(self, 
                          z: State, 
                          dz: State,
                          z_background: State = None) -> None:
        """
        Compute the advection term of the state vector z.

        Args:
            z (State)  : State object.
            dz (State) : Advection term of the state.
            z_background (State) : Background state object (optional).
        """

        # calculate the full velocity field
        zf = z if z_background is None else z + z_background

        # shorthand notation
        inter = self.interpolation
        inter_xf = inter.sym_xf; inter_xb = inter.sym_xb
        inter_yf = inter.sym_yf; inter_yb = inter.sym_yb
        inter_zf = inter.sym_zf; inter_zb = inter.sym_zb
        uf = zf.u; vf = zf.v; wf = zf.w
        u  = z.u;  v  = z.v;  w  = z.w
        du = dz.u; dv = dz.v; dw = dz.w
        bcx = self.bcx; bcy = self.bcy; bcz = self.bcz

        # -----------------------------------
        #  ADVECTION OF THE U-COMPONENT
        # -----------------------------------
        # position of the fluxes:
        #       ------v-----[fy]---v----[fy]        ------w-----[fz]---w----[fz]
        #       |            |            |         |            |            |
        #       |            |            |         |            |            |
        #       |    [fx]    u    [fx]    u         |    [fx]    u    [fx]    u
        #       |            |            |         |            |            | 
        #       |            |            |         |            |            | 
        #   ^   ------v-----[fy]---v----[fy]    ^   ------w-----[fz]---w----[fz]
        #   |   |            |            |     |   |            |            | 
        #   |   |            |            |     |   |            |            | 
        # y |   |    [fx]    u    [fx]    u   z |   |    [fx]    u    [fx]    u 
        #   |   |            |            |     |   |            |            | 
        #   |   |            |            |     |   |            |            | 
        #       ---------------------------         --------------------------- 
        #
        #         -------> x                           -------> x

        # calculate the fluxes
        fx = inter_xb(uf) * inter_xb(u)  # flux in x-direction
        fy = inter_xf(vf) * inter_yf(u)  # flux in y-direction
        fz = inter_xf(wf) * inter_zf(u)  # flux in z-direction

        # calculate boundary conditions
        fx = self.grid.cp.pad(fx, ((0,1), (0,0), (0,0)), bcx)
        fy = self.grid.cp.pad(fy, ((0,0), (1,0), (0,0)), bcy)
        fz = self.grid.cp.pad(fz, ((0,0), (0,0), (1,0)), bcz)

        # calculate the flux divergence => tendency term
        du[:] -= self.flux_divergence(fx, fy, fz) * self.grid.mset.Ro

        # -----------------------------------
        #  ADVECTION OF THE V-COMPONENT
        # -----------------------------------

        # position of the fluxes:
        #       ------v-----[fx]---v----[fx]        ------w-----[fz]---w----[fz]
        #       |            |            |         |            |            |
        #       |            |            |         |            |            |
        #       |    [fy]    u    [fy]    u         |    [fy]    v    [fy]    v
        #       |            |            |         |            |            | 
        #       |            |            |         |            |            | 
        #   ^   ------v-----[fx]---v----[fx]    ^   ------w-----[fz]---w----[fz]
        #   |   |            |            |     |   |            |            | 
        #   |   |            |            |     |   |            |            | 
        # y |   |    [fy]    u    [fy]    u   z |   |    [fy]    v    [fy]    v 
        #   |   |            |            |     |   |            |            | 
        #   |   |            |            |     |   |            |            | 
        #       ---------------------------         --------------------------- 
        #
        #         -------> x                           -------> y

        # calculate the fluxes
        fx = inter_yf(uf) * inter_xf(v)  # flux in x-direction
        fy = inter_yb(vf) * inter_yb(v)  # flux in y-direction
        fz = inter_yf(wf) * inter_zf(v)  # flux in z-direction

        # calculate boundary conditions
        fx = self.grid.cp.pad(fx, ((1,0), (0,0), (0,0)), bcx)
        fy = self.grid.cp.pad(fy, ((0,0), (0,1), (0,0)), bcy)
        fz = self.grid.cp.pad(fz, ((0,0), (0,0), (1,0)), bcz)

        # calculate the flux divergence => tendency term
        dv[:] -= self.flux_divergence(fx, fy, fz) * self.grid.mset.Ro

        # -----------------------------------
        #  ADVECTION OF THE W-COMPONENT
        # -----------------------------------

        # position of the fluxes:
        #       ------w-----[fx]---w----[fx]        ------w-----[fy]---w----[fy]
        #       |            |            |         |            |            |
        #       |            |            |         |            |            |
        #       |    [fz]    u    [fz]    u         |    [fz]    v    [fz]    v
        #       |            |            |         |            |            | 
        #       |            |            |         |            |            | 
        #   ^   ------w-----[fx]---w----[fx]    ^   ------w-----[fy]---w----[fy]
        #   |   |            |            |     |   |            |            | 
        #   |   |            |            |     |   |            |            | 
        # z |   |    [fz]    u    [fz]    u   z |   |    [fz]    v    [fz]    v 
        #   |   |            |            |     |   |            |            | 
        #   |   |            |            |     |   |            |            | 
        #       ---------------------------         --------------------------- 
        #
        #         -------> x                           -------> y

        # calculate the fluxes
        fx = inter_zf(uf) * inter_xf(w)  # flux in x-direction
        fy = inter_zf(vf) * inter_yf(w)  # flux in y-direction
        fz = inter_zb(wf) * inter_zb(w)  # flux in z-direction

        # calculate boundary conditions
        fx = self.grid.cp.pad(fx, ((1,0), (0,0), (0,0)), bcx)
        fy = self.grid.cp.pad(fy, ((0,0), (1,0), (0,0)), bcy)
        fz = self.grid.cp.pad(fz, ((0,0), (0,0), (0,1)), bcz)

        # calculate the flux divergence => tendency term
        dw[:] -= self.flux_divergence(fx, fy, fz) * self.grid.mset.Ro

        # -----------------------------------
        #  ADVECTION OF THE B-COMPONENT (buoyancy)
        # -----------------------------------

        # position of the fluxes:
        #       ----w,[fz]-------w,[fz]----         ----w,[fz]-------w,[fz]----
        #       |            |            |         |            |            |
        #       |            |            |         |            |            |
        #       |     b   u,[fx]   b   u,[fx]       |     b   v,[fy]   b  v,[fy]
        #       |            |            |         |            |            | 
        #       |            |            |         |            |            | 
        #   ^   ----w,[fz]-------w,[fz]----     ^   ----w,[fz]-------w,[fz]----
        #   |   |            |            |     |   |            |            | 
        #   |   |            |            |     |   |            |            | 
        # z |   |     b   u,[fx]   b   u,[fx] z |   |     b   v,[fy]   b  v,[fy]
        #   |   |            |            |     |   |            |            | 
        #   |   |            |            |     |   |            |            | 
        #       ---------------------------         --------------------------- 
        #
        #         -------> x                           -------> y
        
        # calculate the fluxes
        fx = uf * inter_xf(z.b)  # flux in x-direction
        fy = vf * inter_yf(z.b)  # flux in y-direction
        fz = wf * inter_zf(z.b)  # flux in z-direction

        # calculate boundary conditions
        fx = self.grid.cp.pad(fx, ((1,0), (0,0), (0,0)), bcx)
        fy = self.grid.cp.pad(fy, ((0,0), (1,0), (0,0)), bcy)
        fz = self.grid.cp.pad(fz, ((0,0), (0,0), (1,0)), bcz)

        # calculate the flux divergence => tendency term
        dz.b[:] -= self.flux_divergence(fx, fy, fz) * self.grid.mset.Ro

        return dz


    def flux_divergence(self, flux_x, flux_y, flux_z):
        """
        Calculate the flux divergence. 
        Assumes that fluxes are situated half a grid cell away from the center
        in the positive direction.
        """
        f = slice(1,None); b = slice(None,-1)
        divergence = (flux_x[f,:,:] - flux_x[b,:,:]) * self.dx1 + \
                     (flux_y[:,f,:] - flux_y[:,b,:]) * self.dy1 + \
                     (flux_z[:,:,f] - flux_z[:,:,b]) * self.dz1

        return divergence


class CenteredAdvectionConstructor(AdvectionConstructor):
    """
    Constructor for the centered advection scheme.
    """
    def __init__(
            self, 
            interpolation_constructor: InterpolationConstructor = LIC()):
        """
        Constructor for the centered advection scheme.

        Args:
            interpolation_constructor (InterpolationConstructor) : 
                Constructor for the interpolation scheme. 
                Default is the linear interpolation scheme.
        """
        self.interpolation_constructor = interpolation_constructor

    def __call__(self, 
                 grid: Grid,
                 timer: TimingModule) -> CenteredAdvection:
        """
        Create a new instance of the centered advection scheme.
        """
        interpolation = self.interpolation_constructor(grid.mset, grid)
        return CenteredAdvection(grid, timer, interpolation)

    def __repr__(self):
        res = "  Advection Scheme: \n"
        res += "    scheme = Centered Advection\n"
        res += "    interpolation = {}\n".format(self.interpolation_constructor)
        return res

# remove symbols from the namespace
del Grid, ModelSettings, TimingModule, State, InterpolationModule, \
    InterpolationConstructor, LIC, AdvectionModule, AdvectionConstructor