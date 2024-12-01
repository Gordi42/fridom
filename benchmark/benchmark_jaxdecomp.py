import jax
import jax.numpy as jnp
import fridom.framework as fr
import timeit
import jaxdecomp
import numpy as np

jax.distributed.initialize()

rank = jax.process_index()
size = jax.process_count()
if rank == 0:
    print(f"Rank: {rank}, Size: {size}")
    print(f"Devices: {jax.devices()}")

def bench(func: "str", number: int = 3, repeat: int = 20):
    # execute the function once to ensure it is compiled
    func()
    times = timeit.repeat(func, globals=globals(), number=number, repeat=repeat)
    mean = np.mean(times)
    std = np.std(times)
    if rank == 0:
        print(f"Function: {func}, Mean: {mean}, Std: {std}")

# # # N = [64, 128, 256, 512, 1024, 2048]
N = 64
def benchmark_fft(N: int):
    domain = fr.domain_decomposition.JaxDecomposition((N,N,N), halo=2, p_dims=(4,4))
    p_for = domain.parallel_forward_transform(jnp.fft.fftn)
    fftn_jit = jax.jit(jnp.fft.fftn)
    pfft3d_jit = jax.jit(jaxdecomp.pfft3d)
    x = domain.create_array()
    x_unpad = domain.unpad(x)


    def fftn():
        fftn_jit(x_unpad).block_until_ready()

    def pfft3d():
        pfft3d_jit(x_unpad).block_until_ready()

    def my_fft():
        p_for(x).block_until_ready()


    # bench(fftn)
    bench(pfft3d)
    bench(my_fft)

def benchmark_gather(N: int):
    domain = fr.domain_decomposition.JaxDecomposition((N,N,N), halo=2, p_dims=(size,1))
    x = domain.create_array()

    def gather_all():
        domain.gather(x)

    def gather_part():
        domain.gather(x, slc=(slice(None), slice(None), 0))

    bench(gather_all)
    bench(gather_part)
    
def run_benchmark(func: callable) -> None:
    try: 
        func()
    except Exception as e:
        print(f"Failed to run benchmark")
        print(e)
        jax.distributed.shutdown()
        exit(1)


for N in [64, 128, 256, 512, 1024, 2048]:
# for N in [1024]:
    if rank == 0:
        print("=========================================")
        print(f"Running benchmark for N={N}")
        print("=========================================")

    run_benchmark(lambda : benchmark_fft(N))
    # run_benchmark(lambda _: benchmark_gather(N))

jax.distributed.shutdown()
exit(0)