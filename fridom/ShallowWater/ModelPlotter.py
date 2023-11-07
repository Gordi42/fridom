import matplotlib.pyplot as plt
import numpy as np

from fridom.Framework.Animation import ModelPlotterBase
from fridom.ShallowWater.Plot import Plot


class ModelPlotter(ModelPlotterBase):
    def create_figure():
        fig = plt.figure(figsize=(5,5), dpi=200, tight_layout=True)
        return fig

    def update_figure(fig, z, time, **kwargs):
        Plot(z.h).top(z, fig=fig)
        fig.suptitle("Time: {:.2f} s".format(time))
        return

    def convert_to_img(fig):
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer._renderer)
        return img
