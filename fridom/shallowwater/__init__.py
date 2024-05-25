# import all the classes and functions from the nonhydro module
from fridom.shallowwater.model_settings import ModelSettings
from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State
from fridom.shallowwater.model import Model
from fridom.shallowwater.plot import Plot, PlotContainer
from fridom.shallowwater.model_plotter import ModelPlotter
from fridom.shallowwater.diagnosed_imbalance import DiagnoseImbalance

# move the field variable into the nonhydro namespace
from fridom.framework.field_variable import FieldVariable

# import modules
import fridom.shallowwater.initial_conditions
import fridom.shallowwater.projection
import fridom.shallowwater.source
import fridom.shallowwater.eigenvectors