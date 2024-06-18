For performance reasons, one may want to use the preinstalled openmpi library on levante. This is a guide on how to setup the conda environment, such that it use the levante openmpi library.

# Create Conda Environment

```bash
module load python3
conda create --name fridom python=3.11.9 mpi4py 'openmpi=4.1.*=external_*' cupy 'netcdf4=*=mpi_openmpi*'
```

# Set external variables
Create the file `env_vars.sh` in the `activate.d` directory of the conda environment. Usually under `~/.conda/envs/fridom/etc/conda/activate.d`. And add the content:
```bash
export OMPI_MCA_osc="ucx"
export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl="self"
export UCX_HANDLE_ERRORS="bt"
export OMPI_MCA_pml_ucx_opal_mem_hooks=1
export LD_PRELOAD=/sw/spack-levante/pmix-3.2.1-chn3vj/lib/libpmi2.so.1.0.0
export LD_LIBRARY_PATH=/sw/spack-levante/openmpi-4.1.6-mjsagq/lib 
```
# Set external variables for VSCode
For some reason the variables set in the `env_vars.sh` are not loaded when selecting the interpreter in VSCode. A workaround for this problem is to write a wrapper around the python interpreter that also sets the environmental variables at the beginning. For that we first have to find out the path of the python binary:
```bash
conda activate fridom
which python3
```
In my case, the binary is located under `~/.conda/envs/fridom2/bin/python3`. Now we create the wrapper around the binary. For example in `~/my_envs/fridom/bin/` create the file `python` with the following content:
```bash
export OMPI_MCA_osc="ucx"
export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl="self"
export UCX_HANDLE_ERRORS="bt"
export OMPI_MCA_pml_ucx_opal_mem_hooks=1
export LD_PRELOAD=/sw/spack-levante/pmix-3.2.1-chn3vj/lib/libpmi2.so.1.0.0
export LD_LIBRARY_PATH=/sw/spack-levante/openmpi-4.1.6-mjsagq/lib 

/path/to/python/binary "$@"
```
and make the file executable with 
```bash
chmod +x ~/my_envs/fridom/bin/python
```
Finally we go into VSCode open the command palette and search for `select interpreter`. There we enter the path of the new python program.