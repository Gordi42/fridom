Parallelizing simulations
=========================

Although the basic structure for parallelization is already prepared, FRIDOM does not yet support parallelization. We plan to parallelize the framework using jaxDecomp. Nevertheless, thanks to its compatibility with GPUs, simulations with grid sizes on the order of (10^6) grid points—such as (512^3) or (8192^2) grid points—can already be run in a reasonable amount of time.
