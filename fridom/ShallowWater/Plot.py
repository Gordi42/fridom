from typing import Any
import numpy as np

from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid
from fridom.Framework.FieldVariable import FieldVariable


class PlotContainer:
    """
    Class to store plotting function.
    """

    def update_colorlimits(field, is_positive:bool, cmin=None, cmax=None):
        """
        Get the color settings for the plot.

        Arguments:
            fields     : field to plot
            cmin       : minimum value for the colormap
            cmax       : maximum value for the colormap

        Returns:
            cmin : minimum value for the colormap
            cmax : maximum value for the colormap
        """
        # if cmax is not given, use the maximum value of the field
        if cmax is None:
            cmax = np.max(np.abs(field))
            # if cmax is zero, set it to one
            cmax = 1 if cmax == 0 else cmax

        # if cmin is not given, use the negative value of cmax
        if cmin is None:
            # use symmetric colormap
            cmin = 0 if is_positive else -cmax
        return float(cmin), float(cmax)
    
    def update_colormap(is_positive:bool, cmap=None):
        """
        Set default colormap if not given.

        Arguments:
            is_positive: whether the field is positive definite
            cmap       : colormap to use

        Returns:
            cmap : colormap to use
        """
        # if field is positive definite, use a diverging colormap
        # otherwise use a symmetric colormap
        if is_positive:
            cmap = "OrRd" if cmap is None else cmap
        else:
            cmap = "RdBu_r" if cmap is None else cmap
        return cmap

    def get_coarse_skips(nx, ny, shape):
        """
        Get the number of points to skip for a coarse resolution plot.

        Arguments:
            nx    : desired number of points in x direction
            ny    : desired number of points in y direction
            shape : full shape

        Returns:
            x_skip : number of points to skip in x direction
            y_skip : number of points to skip in y direction
        """
        # get shape of the full grid
        x_count = shape[0]
        y_count = shape[1]

        # get number of points to skip
        x_skip = max(int(x_count / nx), 1)
        y_skip = max(int(y_count / ny), 1)
        return x_skip, y_skip

    def get_quiver_scale(vmax, u, v):
        """
        Get the scaling factor for the quiver plot.

        Arguments:
            vmax  : maximum velocity for the quiver plot
            u     : U velocity
            v     : V velocity
            w     : W velocity
        """
        x_skip, y_skip = PlotContainer.get_coarse_skips(
                                        nx=40, ny=40, shape=u.shape)

        # max number of arrows in one direction
        x_tmp = u[::x_skip, ::y_skip]
        arrow_number = max([x_tmp.shape[0], x_tmp.shape[1]])

        # prepare scaling factors
        if vmax is None:
            vmax = float(np.max(np.sqrt(u**2 + v**2)))

        if vmax == 0:
            vmax = 1

        return vmax * arrow_number, vmax

    def top_on_axis(ax, field, X, Y, cmin, cmax, cmap, vmax,
                      u=None, v=None):
        """
        Plot a top view of the field on a given axis.

        Arguments:
            ax   : matplotlib axis to plot on
            field: field to plot
            u    : U velocity
            v    : V velocity
            cmin : minimum value for the colormap
            cmax : maximum value for the colormap
            vmax : maximum velocity for the quiver plot
            cmap : colormap to use
        """
        ax.set_title("Top View")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

        im = ax.pcolor(X, Y, field,
                       cmap=cmap, vmin=cmin, vmax=cmax)
        
        if u is not None:
            # get skips
            x_skip, y_skip = PlotContainer.get_coarse_skips(
                                        nx=40, ny=40, shape=u.shape)
            # create selection slice tuple
            s = (slice(None, None, x_skip), slice(None, None, y_skip))
            # get scaling factor for quiver plot
            scale, umax = PlotContainer.get_quiver_scale(vmax, u, v)

            # plot quiver plot
            PlotContainer.quiver_on_axis(
                ax, X[s], Y[s], u[s], v[s], scale, umax)

        return im

    def quiver_on_axis(ax, x, y, u, v, scale, vmax):
        """
        Plot a quiver plot on a given axis.

        Arguments:
            ax (axis)      : matplotlib axis to plot on
            x (2D array)   : x coordinates of the arrows
            y (2D array)   : y coordinates of the arrows
            u (2D array)   : x component of the arrows
            v (2D array)   : y component of the arrows
            scale (float)  : scaling factor for the arrows
            vmax (float)   : maximum velocity for the quiver plot
        """
        qv = ax.quiver(x, y, u, v, color='k', scale=scale, width=0.002)
        q_key = ax.quiverkey(qv, 0.95, 1.06, vmax, 
                             "{:.2f}".format(vmax), coordinates="axes")
        return


class Plot:
    """
    Plotting class for quick visualisation of the model state.

    Methods:
        __call__  : plot a top view of the model state
    """

    def __init__(self, field: FieldVariable):
        """
        Constructor of the Plot class.
        """
        self.mset = mset = field.mset
        self.grid = grid = field.grid
        self.name = name = field.name

        # convert to numpy array if necessary
        get = lambda x: x.get() if mset.gpu else x

        # save field and grid information
        self.field = get(field.arr)
        self.x     = get(grid.x[0])
        self.y     = get(grid.x[1])
        self.X     = get(grid.X[0])
        self.Y     = get(grid.X[1])

    # ========================================================================
    #  Main Plotting functions
    # ========================================================================

    def __call__(self, state=None, fig=None, ax=None,
            cmin=None, cmax=None, vmax=None, cmap=None):
        """
        Plot a top view of the field.

        Arguments:
            state : model state to plot
            fig   : matplotlib figure to plot on
            cmin  : minimum value for the colormap
            cmax  : maximum value for the colormap
            vmax  : maximum velocity for the quiver plot
            cmap  : colormap to use
        """
        import matplotlib.pyplot as plt

        # update color settings if necessary
        is_positive = np.all(self.field >= 0) if cmin is None else cmin >= 0
        cmap = PlotContainer.update_colormap(is_positive, cmap)
        cmin, cmax = PlotContainer.update_colorlimits(
            self.field, is_positive, cmin, cmax)

        if fig is None:
            fig = plt.figure(figsize=(5,5), dpi=200, tight_layout=True)

        if state is None:
            u = None; v = None
        else:
            get=lambda x: np.array(x.get()) if self.mset.gpu else np.array(x)
            u = get(state.u); v = get(state.v)
        
        if ax is None:
            ax = fig.add_subplot(111)
        im = PlotContainer.top_on_axis(
            ax, self.field, self.X, self.Y, cmin, cmax, cmap, vmax, u, v)

        shrink = min(self.mset.L[1] / self.mset.L[0],1)
        
        cbar = plt.colorbar(im, shrink=0.9*shrink)
        cbar.set_label(self.name)
        return

# remove symbols from namespace
del Any, ModelSettings, Grid, FieldVariable