[![](Experiments/ShallowWater/FridomAnimation/fridom-title.png)](https://www.youtube.com/watch?v=Fotni4P2ZQs)

# Framework for Idealized Ocean Models (FRIDOM)

FRIDOM is a modeling framework designed with a singular goal in mind: to provide a high-level interface for the development of idealized ocean models. FRIDOM leverages the power of CUDA arrays on GPU through CuPy, enabling the execution of models at medium resolutions, constrained only by your hardware capabilities, right within Jupyter Notebook.

FRIDOM is restricted to models with structured grids in cartesian coordinates.

At the current time, two idealized ocean models are implemented:
- A pseudo-spectral non-hydrostatic Boussinesq model adapted from [ps3d](https://github.com/ceden/ps3d)
- A finite differences rotating shallow water model

## Getting Started

### Prerequisites
A working conda environment. To test the conda installation run the command
```sh
conda list
```
If conda is installed and initialized, it should print a list of the installed packages.

### Installation

1. Clone the repo
```sh
git clone https://github.com/Gordi42/FRIDOM
```
2. Create a new conda environment
```sh
conda create -n fridom python=3.9
```
3. Activate the new environment
```sh
conda activate fridom
```
4. Install cupy
```sh
conda install -c conda-forge cupy
```
5. Install other packages
```sh
pip install netCDF4 plotly=5.18.0 matplotlib tqdm scipy
```
6. Upgrade nbformat
```sh
pip install --upgrade nbformat
```

## Usage

### Quickstart
The following code is an example on how to intialize and run the rotating shallow water model with a barotropic instable jet.

```python
# load modules
from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid
from fridom.ShallowWater.InitialConditions import Jet
from fridom.ShallowWater.Model import Model
from fridom.ShallowWater.Plot import Plot
# initialize and run model
mset = ModelSettings(
    Ro=0.5, N=[256,256], L=[6,6])       # create model settings
grid = Grid(mset)                       # create grid
model = Model(mset, grid)               # create model
model.z = Jet(mset, grid)               # set initial conditions
model.run(runlen=2)                     # Run the model
Plot(model.z.ekin())(model.z)           # plot top view of final kinetic energy
```
<p float="left">
  <img src="media/ShallowWater/SW_Jet_ini.png" width="250" />
  <img src="media/ShallowWater/SW_Jet_evo.png" width="250" /> 
</p>


## Gallery
https://github.com/Gordi42/FRIDOM/assets/118457787/66cca07d-5893-4c1b-af13-901dc78bdd6b


## Author
    * Silvano Rosenau

## License


[MIT](LICENSE.txt)



