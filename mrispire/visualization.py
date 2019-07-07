import numpy as np
from bokeh.plotting import *
from bokeh.layouts import row, layout, gridplot
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, Slider, Legend, \
                         ColorBar, LinearColorMapper

from .reconstruction import *

class BokehStorerVis(object):
    """Storer interactive visualization in Bokeh.

    Parameters
    ----------
    storer: Storer
        The Storer containing the data to be visualized.
    ground_truth: ImageSeries
        The ImageSeries representing the ground truth of the reconstruction

    Attributes
    ----------
    storer: Storer
        The Storer containing the data to be visualized.
    ground_truth: ImageSeries
        The ImageSeries representing the ground truth of the reconstruction

    """

    def __init__(self, storer, ground_truth):
        self.storer = storer
        self.ground_truth = ground_truth

    def modify_doc(self, doc):

        sz = self.storer.stored_reconstructions.shape
        p_values = self.ground_truth.p_values
        scale_gt = np.max(np.abs(self.ground_truth.image_space))

        i_curr = len(self.storer.step_rec)-1
        p_curr = 0
        x_curr = 0
        y_curr = 0
        sel = np.full(len(p_values), np.nan)
        sel[p_curr] = \
            np.abs(self.storer.stored_reconstructions[-1, p_curr, y_curr, x_curr])

        src_plt = ColumnDataSource({
            'x': p_values, 
            'y': np.abs(self.storer.stored_reconstructions[-1, :, 0, 0]),
            'y2': np.full(len(self.ground_truth.p_values), np.nan),
            'selected': sel,
            'gt': np.abs(self.ground_truth.image_space[:, y_curr, x_curr])
        })
        src_im = ColumnDataSource({
            'image': [np.abs(self.storer.stored_reconstructions[-1, p_curr, :, :])],
            'gt': [np.abs(self.ground_truth.image_space[p_curr, :, :])]
        })

        # colorbar
        mapper = LinearColorMapper(palette="Viridis256", 
                                   low=0, high=1.2*scale_gt)
        color_bar = ColorBar(color_mapper=mapper, location=(0,0))

        im = figure(match_aspect=True,
                    tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        im.x_range.range_padding = im.y_range.range_padding = 0
        im.x_range.bounds = (0, sz[-1])
        im.y_range.bounds = (0, sz[-2])
        im.title.text = "Reconstruction"

        im_gt = figure(match_aspect=True,
                    tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        im_gt.x_range.range_padding = im_gt.y_range.range_padding = 0
        im_gt.x_range.bounds = (0, sz[-1])
        im_gt.y_range.bounds = (0, sz[-2])
        im_gt.title.text = "Ground truth"

        p = figure(x_range = (p_values[0], p_values[-1]),
                   y_range = (-0.1*scale_gt, 1.2*scale_gt),
                   tooltips=[("p", "$x"), ("value", "$y")])
        p.title.text = "Signal"
        p.xaxis.axis_label = "P value"
        r0 = p.line(x='x', y='y', source=src_plt, 
                    line_color="black", line_width=2)
        r1 = p.line(x='x', y='y2', source=src_plt,
                    line_dash=[4, 4], line_color="black", line_width=2)
        r2 = p.line(x='x', y='gt', source=src_plt,
                    line_color="green", line_width=2)
        r3 = p.circle(x='x', y='selected', source=src_plt,
                      color="red", size=5)

        legend = Legend(items=[
            ("Reconstruction", [r0]),
            ("Ground truth", [r2]),
            ("Sparse projection", [r1]),
            ("Selected value", [r3]),
        ], location="center")

        p.add_layout(legend, 'right')

        # must give a vector of image data for image parameter
        im.image(image='image', source=src_im,
                 x=0, y=0, dw=sz[-1], dh=sz[-2], color_mapper=mapper)
        im_gt.image(image='gt', source=src_im,
                    x=0, y=0, dw=sz[-1], dh=sz[-2], color_mapper=mapper)
        # im_gt.add_layout(color_bar, 'right')

        def update_plt():
            src_plt.data['y'] = \
                np.abs(self.storer.stored_reconstructions[i_curr, :, y_curr, x_curr])
            if i_curr == sz[0]-1 or i_curr == -1:
                src_plt.data['y2'] = np.full(len(p_values), np.nan)
            else:
                src_plt.data['y2'] = np.abs(self.storer.stored_sparse[
                    i_curr, :, y_curr, x_curr])
            sel = np.full(len(p_values), np.nan)
            sel[p_curr] = \
                np.abs(self.storer.stored_reconstructions[i_curr, p_curr, 
                                                   y_curr, x_curr])
            src_plt.data['selected'] = sel

        def update_im():
            src_im.data['image'] = [
                np.abs(self.storer.stored_reconstructions[i_curr, p_curr, :, :])
            ]
            src_im.data['gt'] = [
                np.abs(self.ground_truth.image_space[p_curr, :, :])
            ]

        def on_im_tap(event):
            nonlocal x_curr, y_curr
            x_curr = int(event.x)
            y_curr = int(event.y)
            update_plt()
            src_plt.data['gt'] = \
                np.abs(self.ground_truth.image_space[:, y_curr, x_curr])

        im.on_event(Tap, on_im_tap)
        im_gt.on_event(Tap, on_im_tap)

        def on_p_slider_change(attr, old, new):
            nonlocal p_curr, i_curr
            p_curr = new
            update_im()
            sel = np.full(len(p_values), np.nan)
            sel[p_curr] = \
                np.abs(self.storer.stored_reconstructions[i_curr, p_curr, 
                                                   y_curr, x_curr])
            src_plt.data['selected'] = sel

        def on_i_slider_change(sttr, old, new):
            nonlocal i_curr
            i_curr = new
            src_im.data['image'] = [
                np.abs(self.storer.stored_reconstructions[i_curr, p_curr, :, :])
            ]
            update_plt()

        p_slider = Slider(start=0, end=len(p_values)-1, 
                          value=p_curr, step=1, 
                          title="Selected p")
        p_slider.on_change('value', on_p_slider_change)

        i_slider = Slider(start=0, 
                          end=len(self.storer.step_rec)-1, 
                          value=i_curr, 
                          step=1, 
                          title="Reconstruction iteration")
        i_slider.on_change('value', on_i_slider_change)

        
        l = gridplot([
            [im_gt, im, p],
            [None, i_slider, p_slider]
        ], sizing_mode='scale_width') #
        doc.add_root(l)
