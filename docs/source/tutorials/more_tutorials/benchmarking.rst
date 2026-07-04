Benchmarking
============

FRIDOM ships a jax-aware benchmarking toolkit in
``fridom.benchmarking``. It measures wall times (correctly blocking on
jax's asynchronous dispatch), trace and compile times, a static memory
analysis of the compiled executable, an estimate of the floating point
operations, and — on backends that report memory statistics, such as
GPUs — the peak device memory usage.

Benchmarking a function
-----------------------

Use :func:`fridom.benchmarking.benchmark` to measure any function:

.. code-block:: python

   import jax.numpy as jnp
   import fridom.benchmarking as bm

   def laplacian(f):
       return sum(jnp.gradient(g, axis=i)
                  for i, g in enumerate(jnp.gradient(f)))

   f = jnp.ones((512, 512))
   result = bm.benchmark(laplacian, f, reps=10, warmup=2)

   print(result.wall_median)    # median wall time in seconds
   print(result.compile_time)   # compile time in seconds
   print(result.temp_bytes)     # temporary buffer size in bytes
   print(result.flops)          # estimated flops per call

By default, the function is jit-compiled ahead of time and the timed
repetitions run the compiled executable. For functions that are not
jit-compatible (or that manage their own jit, like ``model.step``),
pass ``measure_compile=False``; only wall times are recorded then.

.. note::

   jax dispatches asynchronously: the measurement blocks on the return
   value of the function. Always return the arrays (or fields) that
   the computation produces — a function returning nothing is only
   timed for its dispatch overhead.

The benchmark suite
-------------------

The repository contains a benchmark suite in the ``benchmarks/``
directory. Suite files (``bench_*.py``) declare cases with the
:func:`fridom.benchmarking.benchmark_case` decorator:

.. code-block:: python

   import jax.numpy as jnp
   from fridom.benchmarking import benchmark_case

   @benchmark_case(params={"n": [64, 128, 256]})
   def bench_square_sum(n):
       x = jnp.ones((n, n))     # setup code is not timed

       def run(x):
           return (x * x).sum()

       return run, (x,), {"points": float(n * n)}

The decorated function is the *setup* (untimed); it returns the
callable to be measured, optionally with a tuple of arguments and a
dictionary of extra metrics. Pass arrays as arguments instead of
closing over them: arguments are traced by the jit compiler, while
closed-over arrays are embedded as constants and the computation may
be folded away.

Run the suite from the repository root:

.. code-block:: bash

   uv run python -m fridom.benchmarking run benchmarks
   uv run python -m fridom.benchmarking run benchmarks --filter bench_nonhydro
   uv run python -m fridom.benchmarking list benchmarks

Each case instance runs in a fresh subprocess so that peak-memory
counters are independent, jit caches do not interact, and a crashing
case does not abort the suite (use ``--no-isolate`` for debugging).
Results are written as JSON files to ``benchmarks/results/``.

Comparing two runs
------------------

The typical workflow to evaluate a refactoring:

.. code-block:: bash

   git switch main
   uv run python -m fridom.benchmarking run benchmarks -o results/main.json

   git switch my-feature
   uv run python -m fridom.benchmarking run benchmarks -o results/feature.json

   uv run python -m fridom.benchmarking compare results/main.json results/feature.json

The comparison report flags cases whose median wall time changed by
more than a threshold (5% by default, ``--threshold``). Use
``--markdown`` to render a markdown table that can be pasted into a
pull request, and ``--fail-on-regression`` to get a non-zero exit code
if any case got slower or errored (useful in scripts).

On the levante HPC system, ``benchmarks/start_job.sh`` submits the
suite as a GPU job.
