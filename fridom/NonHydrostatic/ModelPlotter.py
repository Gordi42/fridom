import numpy as np

from fridom.Framework.Animation import ModelPlotterBase


class ModelPlotter(ModelPlotterBase):
    def create_figure():
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5,5), dpi=200, tight_layout=True)
        return fig

    def update_figure(fig, z, p, time, **kwargs):
        from fridom.NonHydrostatic.Plot import Plot
        Plot(z.b).top(z, fig=fig)
        fig.suptitle("Time: {:.2f} s".format(time))
        return

    def convert_to_img(fig):
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer._renderer)
        return img

# remove symbols from namespace
del ModelPlotterBase