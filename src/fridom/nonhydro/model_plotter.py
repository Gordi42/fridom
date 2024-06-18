import numpy as np

from fridom.framework.modules.animation.model_plotter import ModelPlotterBase
from fridom.framework.model_state import ModelState


class ModelPlotter(ModelPlotterBase):
    def create_figure():
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5,5), dpi=200, tight_layout=True)
        return fig

    def update_figure(fig, mz: ModelState):
        from fridom.nonhydro.plot import Plot
        Plot(mz.z.b).top(mz.z, fig=fig)
        fig.suptitle("Time: {:.2f} s".format(mz.time))
        return

    def convert_to_img(fig):
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer._renderer)
        return img

# remove symbols from namespace
del ModelPlotterBase, ModelState