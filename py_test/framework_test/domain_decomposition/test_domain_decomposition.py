
enable_gpu = False

size = 1024
n_reps = 1_000

import fridom.framework as fr

from domain_decomposition import DomainDecomposition
from transformer import Transformer
from mpi4py import MPI
from timing_module import TimingModule

timer = TimingModule()
def print0(msg):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(msg, flush=True)

if enable_gpu:
    backend = "cupy"
else:
    backend = "numpy"


# --------------------------------------------------------------------------
#  Test domain decomposition
# --------------------------------------------------------------------------

in_halos = [0, 1, 4]
out_halos = [0, 0, 3]

for in_halo, out_halo in zip(in_halos, out_halos):
    print0(f"Testing transformation with halo regions: in_halo={in_halo}, out_halo={out_halo}")

    print0("Testing transformation from x-pencils to y-pencils")
    domain_x = DomainDecomposition(
        n_global=[128, 128, 128], shared_axes=[0], halo=in_halo, backend=backend)
    domain_y = DomainDecomposition(
        n_global=[128, 128, 128], shared_axes=[1], halo=out_halo, backend=backend)
    transformer = Transformer(domain_x, domain_y)
    u = domain_x.ncp.random.rand(*domain_x.my_subdomain.shape)
    domain_x.sync(u)
    u_tmp = transformer.forward(u)
    assert u_tmp.shape == domain_y.my_subdomain.shape
    assert domain_x.ncp.allclose(u, transformer.backward(u_tmp))
    MPI.COMM_WORLD.Barrier()
    print0("Test passed")

    print0("Testing transformation from x-pencils to no shared axes")
    domain_x = DomainDecomposition(
        n_global=[128, 128, 128], shared_axes=[0], halo=in_halo, backend=backend)
    domain_no_shared = DomainDecomposition(
        n_global=[128, 128, 128], halo=out_halo, backend=backend)
    transformer = Transformer(domain_x, domain_no_shared)
    u = domain_x.ncp.random.rand(*domain_x.my_subdomain.shape)
    domain_x.sync(u)
    u_tmp = transformer.forward(u)
    assert u_tmp.shape == domain_no_shared.my_subdomain.shape
    assert domain_x.ncp.allclose(u, transformer.backward(u_tmp))
    print0("Test passed")


# --------------------------------------------------------------------------
#  Test performance
# --------------------------------------------------------------------------


# time the transformer
domain1 = DomainDecomposition(
    n_global=[size, size], shared_axes=[0], backend=backend)
domain2 = DomainDecomposition(
    n_global=[size, size], shared_axes=[1], backend=backend)
transformer = Transformer(domain1, domain2).forward
u = domain1.ncp.random.rand(*domain1.my_subdomain.shape)
domain1.sync(u)
v = domain1.ncp.zeros(domain2.my_subdomain.shape)

timer.total.start()
for _ in range(n_reps):
    transformer(u, v)
timer.total.stop()

if MPI.COMM_WORLD.Get_rank() == 0:
    print(timer)