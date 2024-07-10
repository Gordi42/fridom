from fridom.nonhydro.state import State
from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class CGPressureSolver(Module):
    """
    This class solves the pressure field with a conjugate gradient solver.
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-10):
        """
        # Conjugate Gradient Pressure Solver
        ## Arguments:
        - `max_iter` (int) : Maximum number of iterations.
        - `tol` (float)    : Tolerance.
        """
        super().__init__(name="Pressure Solver", max_iter=max_iter, tol=tol)

    @start_module
    def start(self):
        # cast the parameters to the correct data type
        self.tol = self.mset.dtype(self.tol)

        # Create a function to solve for pressure
        if self.mset.gpu:
            from cupyx.scipy.sparse.linalg import cg, LinearOperator
        else:
            from scipy.sparse.linalg import cg, LinearOperator

        self.cg = cg
        
        # shorthand notation
        mset = self.mset
        Nx  = mset.N[0]; Ny  = mset.N[1]; Nz  = mset.N[2]
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

    @update_module
    def update(self, mz: ModelState) -> None:
        """
        Solve for the pressure field.

        Args:
            mz (ModelState) : Model state.
        """
        p_flat, info = self.cg(self.A, mz.z_diag.div.reshape(-1), x0=mz.p.reshape(-1),
                             tol=self.tol, maxiter=self.max_iter)
        mz.z_diag.p[:] = p_flat.reshape(self.mset.N)
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    solver = CG\n"
        res += f"    max_iter = {self.max_iter}\n"
        res += f"    tol = {self.tol}\n"
        return res

# remove symbols from the namespace
del State, ModelState, Module, update_module, start_module