from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.NonHydrostatic.Grid import Grid
from fridom.NonHydrostatic.State import State
from fridom.Framework.TimingModule import TimingModule
from fridom.Framework.FieldVariable import FieldVariable
from fridom.NonHydrostatic.BoundaryConditions import PBoundary


class PressureSolve:
    """
    This class solves a tendency term for the pressure with a conjugate gradient
    solver.
    """

    def __init__(self, mset: ModelSettings, grid: Grid, timer: TimingModule):
        """
        Constructor of the Spectral pressure solver class.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        self.mset = mset
        self.grid = grid
        self.timer = timer

        self.div = FieldVariable(mset, grid,
                    name="Divergence", bc=PBoundary(mset))

        self.dx1 = mset.dtype(1.0) / mset.dx
        self.dy1 = mset.dtype(1.0) / mset.dy
        self.dz1 = mset.dtype(1.0) / mset.dz

        if mset.pressure_solver == "CG":
            self.solve_for_pressure = CGSolver(mset, grid)
        elif mset.pressure_solver == "Spectral":
            self.solve_for_pressure = SpectralSolver(mset, grid)
        else:
            raise ValueError(
                "Unknown pressure solver: {}".format(mset.pressure_solver))

        # add a timer
        self.timer.add_component('Pressure Solve')

    def __call__(self, dz:State, p:FieldVariable):
        """
        Solve for the pressure.

        Args:
            dz (State)        : Tendency of the state.
            p (FieldVariable) : Pressure field.
        """
        # start the timer
        self.timer.get("Pressure Solve").start()

        cp = self.grid.cp

        u_pad = cp.pad(dz.u, ((1,0), (0,0), (0,0)), 'wrap')
        v_pad = cp.pad(dz.v, ((0,0), (1,0), (0,0)), 'wrap')
        w_pad = cp.pad(dz.w, ((0,0), (0,0), (1,0)), 'wrap')

        # boundary conditions
        if not self.mset.periodic_bounds[0]:
            u_pad[0,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            v_pad[:,0,:] = 0
        if not self.mset.periodic_bounds[2]:
            w_pad[:,:,0] = 0

        # Slices
        c = slice(1,None); b = slice(None,-1); n = slice(None)
        xb = (b,n,n); xc = (c,n,n)
        yb = (n,b,n); yc = (n,c,n)
        zb = (n,n,b); zc = (n,n,c)

        # calculate divergence
        self.div[:] = (u_pad[xc] - u_pad[xb])*self.dx1 + \
                      (v_pad[yc] - v_pad[yb])*self.dy1 + \
                      (w_pad[zc] - w_pad[zb])*self.dz1

        self.solve_for_pressure(self.div, p)

        # stop the timer
        self.timer.get("Pressure Solve").stop()
        return 


class SpectralSolver:
    def __init__(self, mset: ModelSettings, grid:Grid):
        self.mset = mset
        self.grid = grid
        return

    def __call__(self, div:FieldVariable, p:FieldVariable):
        ps = div.fft() / (-self.grid.k2_hat)
        ps[self.grid.k2_hat_zero] = 0
        p[:] = ps.fft()
        return


class CGSolver:
    def __init__(self, mset: ModelSettings, grid:Grid):
        self.mset = mset
        self.grid = grid

        # Create a function to solve for pressure
        if self.mset.gpu:
            from cupyx.scipy.sparse.linalg import cg, LinearOperator
        else:
            from scipy.sparse.linalg import cg, LinearOperator

        self.cg = cg
        
        # shorthand notation
        Nx  = self.mset.N[0]; Ny  = self.mset.N[1]; Nz  = self.mset.N[2]
        dx2 = mset.dtype(1.0) / mset.dx**2
        dy2 = mset.dtype(1.0) / mset.dy**2
        dz2 = mset.dtype(1.0) / mset.dz**2
        NxNyNz = self.mset.total_grid_points
        dsqr   = self.mset.dsqr
        cp     = self.grid.cp

        # define linear operator
        def laplace(p):
            p = p.reshape(Nx,Ny,Nz)
            p_pad = cp.pad(p, ((1,1),(1,1),(1,1)), "wrap")

            f = slice(2,None); b = slice(None,-2); c = slice(1,-1)
            xb = (b,c,c); xf = (f,c,c); xc = (c,c,c)
            yb = (c,b,c); yf = (c,f,c); yc = (c,c,c)
            zb = (c,c,b); zf = (c,c,f); zc = (c,c,c)

            p_laplace = ((p_pad[xf] -2*p_pad[xc] + p_pad[xb]) * dx2 +
                        (p_pad[yf] -2*p_pad[yc] + p_pad[yb]) * dy2 +
                        (p_pad[zf] -2*p_pad[zc] + p_pad[zb]) * dz2 / dsqr )
            return p_laplace.reshape(-1)

        self.A = LinearOperator((NxNyNz, NxNyNz), matvec=laplace)
        return

    def __call__(self, div:FieldVariable, p:FieldVariable):
        """
        Solve for the pressure.

        Args:
            div (FieldVariable) : Divergence.
            p (FieldVariable)   : Pressure field.
        """
        maxiter = self.mset.max_cg_iter
        cg_tol  = self.mset.cg_tol

        p[:], info = self.cg(self.A, div.reshape(-1), x0=p.reshape(-1),
                             tol=cg_tol, maxiter=maxiter)
        return
