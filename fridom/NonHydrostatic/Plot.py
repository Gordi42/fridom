import numpy as np

from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.NonHydrostatic.Grid import Grid
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

    def get_coord_index(coord_ind, coord_val, coord_arr):
        """
        Get the index of the selected coordinate.

        Arguments:
            coord_ind : index of the selected coordinate
            coord_val : value of the selected coordinate
            coord_arr : array of coordinate

        Returns:
            ind : index of the selected coordinate
        """
        ind = 0
        if coord_ind is None:
            ind = np.argmin(np.abs(coord_arr - coord_val))
        else:
            ind = coord_ind
        return ind

    def get_coarse_skips(nx, ny, nz, shape):
        """
        Get the number of points to skip for a coarse resolution plot.

        Arguments:
            nx    : desired number of points in x direction
            ny    : desired number of points in y direction
            nz    : desired number of points in z direction
            shape : full shape

        Returns:
            x_skip : number of points to skip in x direction
            y_skip : number of points to skip in y direction
            z_skip : number of points to skip in z direction
        """
        # get shape of the full grid
        x_count = shape[0]
        y_count = shape[1]
        z_count = shape[2]

        # get number of points to skip
        x_skip = max(int(x_count / nx), 1)
        y_skip = max(int(y_count / ny), 1)
        z_skip = max(int(z_count / nz), 1)
        return x_skip, y_skip, z_skip

    def get_quiver_scale(vmax, u, v, w):
        """
        Get the scaling factor for the quiver plot.

        Arguments:
            vmax  : maximum velocity for the quiver plot
            u     : U velocity
            v     : V velocity
            w     : W velocity
        """
        x_skip, y_skip, z_skip = PlotContainer.get_coarse_skips(
                                        nx=40, ny=40, nz=40, shape=u.shape)

        # max number of arrows in one direction
        x_tmp = u[::x_skip, ::y_skip, ::z_skip]
        arrow_number = max([x_tmp.shape[0], x_tmp.shape[1], x_tmp.shape[2]])

        # prepare scaling factors
        if vmax is None:
            vmax = float(np.max(np.sqrt(u**2 + v**2 + w**2)))

        if vmax == 0:
            vmax = 1

        return vmax * arrow_number, vmax

    def front_on_axis(ax, field, X, Y, Z, ysel, cmin, cmax, cmap, vmax,
                      u=None, v=None, w=None):
        """
        Plot a front view of the field on a given axis.

        Arguments:
            ax   : matplotlib axis to plot on
            field: field to plot
            u    : U velocity
            v    : V velocity
            w    : W velocity
            cmin : minimum value for the colormap
            cmax : maximum value for the colormap
            vmax : maximum velocity for the quiver plot
            cmap : colormap to use
            ysel : y index of the section
        """
        ax.set_title("Front View")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal", adjustable="box")

        im = ax.pcolor(X[:,ysel,:], Z[:,ysel,:], field[:,ysel,:],
                       cmap=cmap, vmin=cmin, vmax=cmax)

        if u is not None:
            # make quiver plot
            x_skip, y_skip, z_skip = PlotContainer.get_coarse_skips(
                                        nx=40, ny=40, nz=40, shape=u.shape)
            # create selection slice tuple
            s = (slice(None, None, x_skip), ysel, slice(None, None, z_skip))
            # get scaling factor for quiver plot
            scale, umax = PlotContainer.get_quiver_scale(vmax,u,v,w)

            # plot quiver plot
            PlotContainer.quiver_on_axis(
                ax, X[s], Z[s], u[s], w[s], scale, umax)

        return im
    
    def side_on_axis(ax, field, X, Y, Z, xsel, cmin, cmax, cmap, vmax,
                      u=None, v=None, w=None):
        """
        Plot a side view of the field on a given axis.

        Arguments:
            ax   : matplotlib axis to plot on
            field: field to plot
            u    : U velocity
            v    : V velocity
            w    : W velocity
            cmin : minimum value for the colormap
            cmax : maximum value for the colormap
            vmax : maximum velocity for the quiver plot
            cmap : colormap to use
            xsel : x index of the section
        """
        ax.set_title("Side View")
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_aspect("equal", adjustable="box")

        im = ax.pcolor(Y[xsel,:,:],
                       Z[xsel,:,:],
                       field[xsel,:,:],
                        cmap=cmap, vmin=cmin, vmax=cmax)
        
        if u is not None:
            # make quiver plot
            x_skip, y_skip, z_skip = PlotContainer.get_coarse_skips(
                                        nx=40, ny=40, nz=40, shape=u.shape)
            # create selection slice tuple
            s = (xsel, slice(None, None, y_skip), slice(None, None, z_skip))
            # get scaling factor for quiver plot
            scale, umax = PlotContainer.get_quiver_scale(vmax,u,v,w)

            # plot quiver plot
            PlotContainer.quiver_on_axis( 
                ax, Y[s], Z[s], v[s], w[s], scale, umax)

        return im

    def top_on_axis(ax, field, X, Y, Z, zsel, cmin, cmax, cmap, vmax,
                      u=None, v=None, w=None):
        """
        Plot a top view of the field on a given axis.

        Arguments:
            ax   : matplotlib axis to plot on
            field: field to plot
            u    : U velocity
            v    : V velocity
            w    : W velocity
            cmin : minimum value for the colormap
            cmax : maximum value for the colormap
            vmax : maximum velocity for the quiver plot
            cmap : colormap to use
            zsel : z index of the section
        """
        ax.set_title("Top View")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

        im = ax.pcolor(X[:,:,zsel], Y[:,:,zsel], field[:,:,zsel],
                       cmap=cmap, vmin=cmin, vmax=cmax)
        
        if u is not None:
            # get skips
            x_skip, y_skip, z_skip = PlotContainer.get_coarse_skips(
                                        nx=40, ny=40, nz=40, shape=u.shape)
            # create selection slice tuple
            s = (slice(None, None, x_skip), slice(None, None, y_skip), zsel)
            # get scaling factor for quiver plot
            scale, umax = PlotContainer.get_quiver_scale(vmax, u, v, w)

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
        front     : plot a front view of the model state
        side      : plot a side view of the model state
        top       : plot a top view of the model state
        sec       : front + side + top view of the model state
        vol       : 3D volume plot of the model state
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
        self.z     = get(grid.x[2])
        self.X     = get(grid.X[0])
        self.Y     = get(grid.X[1])
        self.Z     = get(grid.X[2])

    # ========================================================================
    #  Main Plotting functions
    # ========================================================================

    def vol(self, cmax=None, cmin=None, cmap=None, opacity=0.8, 
            fig=None, show=True):
        """
        Create a 3D volume plot of the field.

        Arguments:
            cmax    : maximum value for the colormap
            cmin    : minimum value for the colormap
            cmap    : colormap to use
            opacity : opacity of the volume plot
            fig     : plotly figure to add the trace to
            show    : whether to show the figure
        """
        import plotly.graph_objects as go
        # get number of points to skip for a coarse resolution plot
        x_skip, y_skip, z_skip = PlotContainer.get_coarse_skips(
                nx=40, ny=40, nz=40, shape=self.field.shape)

        # create selection tuple
        sel = (slice(None, None, x_skip), 
               slice(None, None, y_skip), 
               slice(None, None, z_skip))

        # check if field is positive
        is_positive = np.all(self.field[sel] >= 0)

        # update color settings if necessary
        cmap = PlotContainer.update_colormap(is_positive, cmap)
        cmin, cmax = PlotContainer.update_colorlimits(
            self.field, is_positive, cmin, cmax)

        # get opacity scale (different for symmetric and diverging colormaps)
        if is_positive:             # diverging scale
            opacityscale = [[0, 0], [0.5,0.5], [1, 1]]
        else:                       # symmetric scale
            opacityscale = [[0, 1], [0.5, 0], [1, 1]]

        # create figure if necessary
        if fig is None:
            fig = go.Figure()

        # add trace to figure
        fig.add_trace(go.Volume(
            x=self.X[sel].flatten(),
            y=self.Y[sel].flatten(),
            z=self.Z[sel].flatten(),
            value=np.array(self.field[sel]).flatten(),
            isomin=cmin,
            isomax=cmax,
            opacity=opacity,
            opacityscale=opacityscale,
            surface_count=21,
            colorscale=cmap,
        ))

        fig.update_layout(
            title=self.name,
            scene=dict(
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.25), 
                    center=dict(x=0, y=0, z=-0.5),),
            xaxis=dict(),
            yaxis=dict(),
            zaxis=dict(),
            aspectmode="data",
            ))

        if show:
            fig.show()
        return

    def front(self, state=None, y=0, yi=None, fig=None,
              cmin=None, cmax=None, vmax=None, cmap=None):
        """
        Plot a front view of the field.

        Arguments:
            state : model state to plot
            y     : y coordinate of the section
            yi    : y index of the section
            fig   : matplotlib figure to plot on
            cmin  : minimum value for the colormap
            cmax  : maximum value for the colormap
            vmax  : maximum velocity for the quiver plot
            cmap  : colormap to use
        """
        import matplotlib.pyplot as plt

        ysel = PlotContainer.get_coord_index(yi, y, self.y)

        # update color settings if necessary
        is_positive = np.all(self.field >= 0)
        cmap = PlotContainer.update_colormap(is_positive, cmap)
        cmin, cmax = PlotContainer.update_colorlimits(
            self.field, is_positive, cmin, cmax)
        
        if fig is None:
            fig = plt.figure(figsize=(5,5), dpi=200, tight_layout=True)
        
        if state is None:
            u = None; v = None; w = None
        else:
            get = lambda x: np.array(x.get()) if self.mset.gpu else np.array(x)
            u = get(state.u); v = get(state.v); w = get(state.w)

        ax = fig.add_subplot(111)
        im = PlotContainer.front_on_axis(
            ax, self.field, self.X, self.Y, self.Z,
            ysel, cmin, cmax, cmap, vmax, u, v, w)

        shrink = min(self.mset.L[2] / self.mset.L[1],1)
        
        cbar = plt.colorbar(im, shrink=0.9*shrink)
        cbar.set_label(self.name)
        
        return

    def side(self, state=None, x=0, xi=None, fig=None,
                cmin=None, cmax=None, vmax=None, cmap=None):
        """
        Plot a side view of the field.
    
        Arguments:
            state : model state to plot
            x     : x coordinate of the section
            xi    : x index of the section
            fig   : matplotlib figure to plot on
            cmin  : minimum value for the colormap
            cmax  : maximum value for the colormap
            vmax  : maximum velocity for the quiver plot
            cmap  : colormap to use
        """
        import matplotlib.pyplot as plt
        xsel = PlotContainer.get_coord_index(xi, x, self.x)
    
        # update color settings if necessary
        is_positive = np.all(self.field >= 0)
        cmap = PlotContainer.update_colormap(is_positive, cmap)
        cmin, cmax = PlotContainer.update_colorlimits(
            self.field, is_positive, cmin, cmax)
    
        if fig is None:
            fig = plt.figure(figsize=(5,5), dpi=200, tight_layout=True)

        if state is None:
            u = None; v = None; w = None
        else:
            get=lambda x: np.array(x.get()) if self.mset.gpu else np.array(x)
            u = get(state.u); v = get(state.v); w = get(state.w)

            
        ax = fig.add_subplot(111)
        im = PlotContainer.side_on_axis(
            ax, self.field, self.X, self.Y, self.Z,
            xsel, cmin, cmax, cmap, vmax, u, v, w)
    
        shrink = min(self.mset.L[2] / self.mset.L[1],1)
            
        cbar = plt.colorbar(im, shrink=0.9*shrink)
        cbar.set_label(self.name)
        return

    def top(self, state=None, z=0, zi=None, fig=None,
            cmin=None, cmax=None, vmax=None, cmap=None):
        """
        Plot a top view of the field.

        Arguments:
            state : model state to plot
            z     : z coordinate of the section
            zi    : z index of the section
            fig   : matplotlib figure to plot on
            cmin  : minimum value for the colormap
            cmax  : maximum value for the colormap
            vmax  : maximum velocity for the quiver plot
            cmap  : colormap to use
        """
        import matplotlib.pyplot as plt
        zsel = PlotContainer.get_coord_index(zi, z, self.z)

        # update color settings if necessary
        is_positive = np.all(self.field >= 0)
        cmap = PlotContainer.update_colormap(is_positive, cmap)
        cmin, cmax = PlotContainer.update_colorlimits(
            self.field, is_positive, cmin, cmax)

        if fig is None:
            fig = plt.figure(figsize=(5,5), dpi=200, tight_layout=True)

        if state is None:
            u = None; v = None; w = None
        else:
            get=lambda x: np.array(x.get()) if self.mset.gpu else np.array(x)
            u = get(state.u); v = get(state.v); w = get(state.w)
        
        ax = fig.add_subplot(111)
        im = PlotContainer.top_on_axis(
            ax, self.field, self.X, self.Y, self.Z,
            zsel, cmin, cmax, cmap, vmax, u, v, w)

        shrink = min(self.mset.L[1] / self.mset.L[0],1)
        
        cbar = plt.colorbar(im, shrink=0.9*shrink)
        cbar.set_label(self.name)
        return

    def sec(self, state=None, x=0, y=0, z=0, xi=None, yi=None, zi=None, fig=None,
            cmin=None, cmax=None, vmax=None, cmap=None, add_lines=True):
        import matplotlib.pyplot as plt

        xsel = PlotContainer.get_coord_index(xi, x, self.x)
        ysel = PlotContainer.get_coord_index(yi, y, self.y)
        zsel = PlotContainer.get_coord_index(zi, z, self.z)
        
        is_positive = np.all(self.field >= 0)
        cmap = PlotContainer.update_colormap(is_positive, cmap)
        cmin, cmax = PlotContainer.update_colorlimits(
            self.field, is_positive, cmin, cmax)

        if fig is None:
            fig, axs = plt.subplots(1, 3, figsize=(15,5), dpi=200)
        else:
            axs = [fig.add_subplot(131 + i) for i in range(3)]

        if state is None:
            u = None; v = None; w = None
        else:
            get=lambda x: np.array(x.get()) if self.mset.gpu else np.array(x)
            u = get(state.u); v = get(state.v); w = get(state.w)

        im = PlotContainer.side_on_axis(
            axs[0], self.field, self.X, self.Y, self.Z,
            xsel, cmin, cmax, cmap, vmax, u, v, w)
        PlotContainer.top_on_axis(
            axs[1], self.field, self.X, self.Y, self.Z,
            zsel, cmin, cmax, cmap, vmax, u, v, w)
        PlotContainer.front_on_axis(
            axs[2], self.field, self.X, self.Y, self.Z,
            ysel, cmin, cmax, cmap, vmax, u, v, w)


        cbar = plt.colorbar(im, ax=axs)
        cbar.set_label(self.name)

        # add section lines
        if add_lines:
            # side view
            y = self.Y[xsel,:,:]
            z = self.Z[xsel,:,:]
            axs[0].plot(y[ysel, :], z[ysel, :], c="red", linewidth=0.5)
            axs[0].plot(y[:, zsel], z[:, zsel], c="red", linewidth=0.5)

            # top view
            x = self.X[:,:,zsel]
            y = self.Y[:,:,zsel]
            axs[1].plot(x[xsel, :], y[xsel, :], c="red", linewidth=0.5)
            axs[1].plot(x[:, ysel], y[:, ysel], c="red", linewidth=0.5)

            # front view
            x = self.X[:,ysel,:]
            z = self.Z[:,ysel,:]
            axs[2].plot(x[xsel, :], z[xsel, :], c="red", linewidth=0.5)
            axs[2].plot(x[:, zsel], z[:, zsel], c="red", linewidth=0.5)
        return


# remove symbols from namespace
del ModelSettings, Grid, FieldVariable