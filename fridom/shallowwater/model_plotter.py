from fridom.framework.animation import ModelPlotterBase


class ModelPlotter(ModelPlotterBase):
    def create_figure():
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5,5), dpi=200, tight_layout=True)
        return fig

    def update_figure(fig, z, time, **kwargs):
        from fridom.shallowwater.plot import Plot
        Plot(z.h).top(z, fig=fig)
        fig.suptitle("Time: {:.2f} s".format(time))
        return

    def convert_to_img(fig):
        fig.canvas.draw()
        import numpy as np
        img = np.array(fig.canvas.renderer._renderer)
        return img

# remove symbols from namespace
del ModelPlotterBase