enable_gpu = True
size2D = 1024
size3D = 1024
n_dims = 3
n_reps = 10
halo = 1


from domain_decomposition import DomainDecomposition
from parallel_fft import ParallelFFT
from mpi4py import MPI
from timing_module import TimingModule
timer = TimingModule()
import numpy as np
if enable_gpu:
    import cupy as cp
    backend = "cupy"
else:
    backend = "numpy"

rank = MPI.COMM_WORLD.Get_rank()

ncp = cp if enable_gpu else np
def print0(msg):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(msg, flush=True)


def test_fft(n_globals, halos, halos_out):
    domain_ph = DomainDecomposition(
        n_global=n_global, shared_axes=[0], halo=halo, backend=backend)
    subdom_ph = domain_ph.my_subdomain

    pfft = ParallelFFT(domain_ph, halo_out=halo_out)

    domain_sp = pfft.domain_out
    subdom_sp = domain_sp.my_subdomain

    ncp.random.seed(0)  # Make sure the random numbers are the same for all ranks
    u_global = ncp.random.rand(*domain_ph.n_global)

    u = ncp.zeros(domain_ph.my_subdomain.shape)
    u[subdom_ph.inner_slice] = u_global[subdom_ph.global_slice]
    domain_ph.sync(u)

    u_hat = pfft.forward(u)

    u_hat_global = ncp.fft.fftn(u_global)

    try:
        inner = subdom_sp.inner_slice
        glob = subdom_sp.global_slice
        assert ncp.allclose(u_hat[inner], u_hat_global[glob])
    except AssertionError:
        print0(f"Forward with {n_global=}, {halo=}, {halo_out=} failed (assertion error)")
    except Exception as e:
        print0(f"Forward with {n_global=}, {halo=}, {halo_out=} failed")
        print0(f"Error: {e}")


    u_back = pfft.backward(u_hat).real

    try:
        assert ncp.allclose(u_back, u)
    except AssertionError:
        print0(f"Backward transform with {n_global=}, {halo=} failed")
    return

def test_fft_performance(n_global, shared_axes, halo, halo_out):
    domain = DomainDecomposition(
        n_global=n_global, shared_axes=shared_axes, halo=halo, backend=backend)
    pfft = ParallelFFT(domain, halo_out=halo_out)
    u = ncp.random.rand(*domain.my_subdomain.shape)
    domain.sync(u)


    timer = TimingModule()
    timer.total.start()
    timer.get("Forward").start()
    for _ in range(n_reps):
        u_hat = pfft.forward(u)
    timer.get("Forward").stop()
    timer.get("Backward").start()
    for _ in range(n_reps):
        u2 = pfft.backward(u_hat)
    timer.get("Backward").stop()
    timer.total.stop()
    return timer

def test_single_fft(n_global, shared_axes, halo, halo_out):
    timer = TimingModule()
    timer.total.start()
    domain = DomainDecomposition(
        n_global=n_global, shared_axes=shared_axes, halo=halo, backend=backend)
    
    u = ncp.random.rand(*domain.my_subdomain.shape)
    timer.get("Forward").start()
    for _ in range(n_reps):
        u_hat = ncp.fft.fftn(u[domain.my_subdomain.inner_slice])
    timer.get("Forward").stop()
    timer.get("Backward").start()
    for _ in range(n_reps):
        u2 = ncp.empty_like(u)
        u2[domain.my_subdomain.inner_slice] = ncp.fft.ifftn(u_hat).real
        domain.sync(u2)
    timer.get("Backward").stop()
    timer.total.stop()
    return timer

n_globals = [[64, 51], [51, 20, 31]]#, [size3D]*3]
halos = [0, 1, 2]
halos_out = [0, 1, 2]

print0(f"Testing fft forward and backward")

for n_global in n_globals:
    for halo in halos:
        for halo_out in halos_out:
            test_fft(n_global, halo, halo_out)

print0(f"Test finished")

# print0(f"Testing 2D fft performance")
# timer = test_fft_performance([size2D]*2, [0], halo, 0)
# print0(timer)

print0(f"Testing 3D fft performance (pencil decomposition)")
timer = test_fft_performance([size3D]*3, [0], halo, 0)
print0(timer)

print0(f"Testing 3D fft performance (slab decomposition)")
timer = test_fft_performance([size3D]*3, [0,1], halo, 0)
print0(timer)

# if MPI.COMM_WORLD.Get_size() == 1:
#     print0(f"Testing 3D fft performance (single fft)")
#     timer = test_single_fft([size3D]*3, [0], halo, 0)
#     print0(timer)