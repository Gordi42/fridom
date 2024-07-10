import fridom.nonhydro as nh
import numpy as np
from time import time
from mpi4py import MPI

# nh.config.set_backend("numpy")
ncp = nh.config.ncp

steps = 100

def benchmark(N):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Running benchmark with N = {N}", flush=True)
    f0 = 1e-4
    N2 = (50 * f0) ** 2
    N = tuple([N] * 3)
    L = (10_000, 10_000, 100)

    grid = nh.grid.CartesianGrid(N=N, L=L)
    mset = nh.ModelSettings(grid, f_coriolis=f0, N2=N2)
    mset.time_stepper.dt = np.timedelta64(2, 'm')
    mset.tendencies.advection.disable()
    mset.setup()

    X, Y, Z = grid.X
    Lx, Ly, Lz = grid.L

    z = nh.State(mset)
    z.u[:] = ncp.exp(-(Y - Ly/2)**2 / (0.2*Ly)**2) * ncp.exp(-(Z - Lz/2)**2 / (0.2*Lz)**2)
    z.sync()

    model = nh.Model(mset)
    model.z = z
    model.run(steps=steps)
        
    duration = model.timer.get("Total Integration").time
    #duration = MPI.COMM_WORLD.reduce(duration, op=MPI.MAX, root=0)
    durpstep = duration/steps*1000
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Total: {duration:.2f} s, Avg: {durpstep:.2f} ms/step", flush=True)

benchmark(32)
benchmark(64)
benchmark(128)
benchmark(256)