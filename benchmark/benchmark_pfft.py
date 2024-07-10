import fridom.framework as fr
from time import time
from mpi4py import MPI

fr.config.set_backend("numpy")

n_reps = 100

def benchmark(N):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Running benchmark with N = {N}", flush=True)
    L = tuple([1.0] * len(N))
    grid = fr.grid.CartesianGrid(N=N, L=L, shared_axes=[0])
    mset = fr.ModelSettingsBase(grid)
    mset.setup()

    u = fr.FieldVariable(mset, "u")
    u.arr = fr.config.ncp.random.randn(*N)

    start_time = time()
    for _ in range(n_reps):
        u.fft()
    duration = time() - start_time
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Total duration: {duration:.2f} s  |  Avg duration: {duration/n_reps*1000:.2f} ms", flush=True)

if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"Starting benchmark on {MPI.COMM_WORLD.Get_size()} MPI processes", flush=True)

benchmark((128, 128, 128))
benchmark((256, 256, 256))
benchmark((512, 512, 512))
benchmark((1024, 1024, 1024))
