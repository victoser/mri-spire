from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

from .representations import *
from .mrimages import *

class SparseReconstructor(object):
    """Class performing reconstruction based on a sparse representation.
    
    Parameters
    ----------
    sparse_representation: SparseRepresentation
        The sparse representation used as a prior in reconstruction.
    n_iter: int
        Number of iterations to carry out in the reconstruction.
    progress_analysis: bool
        Whether to trigger events for analysis in each iteration.
    # TODO: continue here
    
    Attributes
    ----------
    sparse_representation: SparseRepresentation
        The sparse representation used as a prior in reconstruction.
    n_iter: int
        Number of iterations to carry out in the reconstruction.
    progress_analysis: bool
        Whether to trigger events for analysis in each iteration.
    end_after_proj: bool
        Whether to end the reconstruction after the projection in the
        last loop.
    """

    def __init__(self, sparse_representation=None, n_iter=20, mask_background=False,
                 progress_analysis=True, rec_analyzers=[]):
        self.sparse_representation = sparse_representation
        self.end_after_proj = False
        self.n_iter = n_iter
        self.mask_background = mask_background
        self.progress_analysis = progress_analysis
        self.rec_analyzers = rec_analyzers

    def reconstruct(self, undersampled_stack, stack_to_rec=None, in_place=False):
        """Recontstruct the undersampled stack.

        Parameters
        ----------
        undersampled_stack: UndersampledStack
        stack_to_rec: ImageStack
            Image stack to reconstruct. It is also 
        in_place: bool, optional
            Whether to keep the original stack unmodified or do the recontruction
            within the given initial object. Default is True.

        Returns
        -------
        ImageStack
            Reconstructed image stack. If the original is not kept it is either
            the stack_to_rec if specified in input, or the same undersampled_stack 
            in the input. Otherwise it is a new ImageStack object created by the method.
        """
        if stack_to_rec is None:
            stack_to_rec = undersampled_stack
        if in_place:
            reconstr_stack = stack_to_rec
        else:
            reconstr_stack = deepcopy(stack_to_rec)

        if self.progress_analysis:
            for analyzer in self.rec_analyzers:
                # send step number 0 to signal initial data
                analyzer.on_data_inserted(0, reconstr_stack)

        for i in range(self.n_iter):
            
            # project onto the sparse representation space
            reconstr_stack.voxels[:] = self.sparse_representation.project(
                np.array(reconstr_stack.voxels))[0]
            # mask background
            if self.mask_background:
                reconstr_stack.image_space[:, reconstr_stack.background_mask] = 0

            if self.progress_analysis:
                for analyzer in self.rec_analyzers:
                    # send step number which is i+1
                    analyzer.on_signal_projected(i+1, reconstr_stack)

            if self.end_after_proj:
                break

            # fill in the data
            undersampled_stack.checkout_data(reconstr_stack)

            if self.progress_analysis:
                for analyzer in self.rec_analyzers:
                    # send step number which is i+1
                    analyzer.on_data_inserted(i+1, reconstr_stack)
                
        return reconstr_stack

class Storer(object):
    """Reconstruction analyzer which stores all the intermediary results.

    Parameters
    ----------

    Attributes
    ----------
    stored_projections: float 4D array
        The series of reconstructions immediately after projection onto 
        the sparse representation space.
    stored_reconstructions: float 4D array
        The series of reconstructions after consistency with data 
        is imposed.
    # TODO: continue here
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.stored_projections = np.empty((0))
        self.stored_reconstructions = np.empty((0))
        # the step of reconstruction at which the projection was captured
        self.step_proj = []
        # the step of reconstruction at which the reconstruction was captured
        self.step_rec = []

    def on_signal_projected(self, i, reconstr_stack):
        proj_list = list(self.stored_projections)
        proj_list.append(reconstr_stack.image_space)
        self.stored_projections = np.array(proj_list)
        self.step_proj.append(i)

    def on_data_inserted(self, i, reconstr_stack):
        if i == 0:
            self.reset()
        rec_list = list(self.stored_reconstructions)
        rec_list.append(reconstr_stack.image_space)
        self.stored_reconstructions = np.array(rec_list)
        self.step_rec.append(i)

    def view_evo_all(self, x, y, plot_ax=None, slider_ax=None, fig=None):
        evo_view = self.view_evo(x, y, plot_ax=plot_ax, slider_ax=slider_ax, fig=fig, plot_proj=False)
        return self.view_evo(x, y, evo_view=evo_view, plot_proj=True)
        
    # TODO: add TE values to plot and object
    def view_evo(self, x, y, plot_ax=None, slider_ax=None, fig=None, evo_view=None, plot_proj=False):
        """"View the evolution of the reconstruction at the specified location of the voxel.
        
        """

        if evo_view is None: # if no evolution view provided, make one
            evo_view = EvoView(self, plot_ax, slider_ax, fig)

        # draw the curve for the first time
        rec_step = int(evo_view.slider.val)
        line = self.plot_signal(rec_step, x, y, evo_view.plot_ax, plot_proj)
        if plot_proj:
            evo_view.line_proj = line
        else:
            evo_view.line_rec = line

        def update(evo_view, storer, val):
            rec_step = int(val)
            if plot_proj:
                line = evo_view.line_proj
            else:
                line = evo_view.line_rec
            evo_view.plot_ax.lines.remove(line)
            line = storer.plot_signal(rec_step, x, y, evo_view.plot_ax, plot_proj)
            if plot_proj:
                evo_view.line_proj = line
            else:
                evo_view.line_rec = line
        evo_view.updates.append(update)
        
        return evo_view # this makes sure the slider is not garbage collected wherever the function is called

    def plot_signal(self, step, x, y, plot_ax, plot_proj=False, marker=None, label=None):
        """Plot the reconstructed/projected signal at a given reconstruction step and voxel"""
        if plot_proj and step != 0:
            if label is None:
                label = "projection"
            storage = self.stored_projections
            if marker is None:
                marker = 'k--'
            step -= 1
        else:
            if label is None:
                label = "reconstruction"
            storage = self.stored_reconstructions
            if marker is None:
                marker = 'k-'
        line_rec = plot_ax.plot(np.abs(storage[step,:,y,x]), marker, label=label)[0]
        plot_ax.figure.canvas.draw()
        plt.show()
        return line_rec

    def voxel_plot(self, x, y, plot_ax=None):
        return self.view_evo_all(x, y, plot_ax)

class ProgressRecorder(object):
    """Reconstruction analyzer which stores all the intermediary errors.

    Parameters
    ----------

    Attributes
    ----------
    RMSE: float vector
        The series of RMSE of the image stack immediately after 
        the data consistency steps.
    nRMSE: dictionary
        The series of nRMSE of the image stack immediately after 
        the data consistency steps. Normalized per total energy
        (key = 'total'), per signal(key = 'signal') and per
        voxel (key = 'voxel').
    proj_RMSE: float vector
        The series of RMSE of the image stack immediately after
        the projection steps.
    proj_nRMSE: dictionary
        The series of nRMSE of the image stack immediately after 
        the projection steps. Normalized per total energy
        (key = 'total'), per signal(key = 'signal') and per
        voxel (key = 'voxel').
    # TODO: correct
    """

    def __init__(self, ground_truth):
        self.reset()
        self.ground_truth = ground_truth

    def reset(self):
        keys = ['RMSE','nRMSE_total', 'nRMSE_signal', 'nRMSE_voxel',
                'dist_from_start']
        self.metrics = {key: np.array([]) for key in keys}
        self.proj_metrics = {key: np.array([]) for key in keys}
        self.dist_from_projection = np.array([])
        self.step_proj = []
        self.step_rec = []

    def on_signal_projected(self, i, reconstr_stack):
        metrics = self.proj_metrics
        self.step_proj.append(i)
        self.analyze_progress(reconstr_stack, metrics)
        diff_proj = self.last_step_reconstr.voxels - reconstr_stack.voxels
        self.dist_from_projection = np.append(self.dist_from_projection,
                                              np.sqrt(np.mean(np.abs(diff_proj)**2)))

    def on_data_inserted(self, i, reconstr_stack):
        if i == 0:
            self.reset()
            self.initial_stack = deepcopy(reconstr_stack)
        metrics = self.metrics
        self.step_rec.append(i)
        self.analyze_progress(reconstr_stack, metrics)
        self.last_step_reconstr = deepcopy(reconstr_stack)

    def analyze_progress(self, reconstr_stack, metrics):
        data = reconstr_stack.voxels_masked(self.ground_truth.background_mask)
        data0 = self.ground_truth.voxels_masked()
        norms = np.linalg.norm(self.ground_truth.voxels_masked(), axis=0)

        diff0 = data - data0

        metrics['RMSE'] = np.append(metrics['RMSE'],
                         np.sqrt(np.mean(np.abs(diff0)**2)))
        metrics['nRMSE_total'] = np.append(metrics['nRMSE_total'],
                         np.sqrt(np.mean(np.abs(diff0)**2) / np.mean(norms**2)))
        metrics['nRMSE_signal'] = np.append(metrics['nRMSE_signal'],
                         np.sqrt(np.mean(np.linalg.norm(diff0, axis=0)**2 / norms**2)))
        metrics['nRMSE_voxel'] = np.append(metrics['nRMSE_voxel'],
                         np.sqrt(np.mean(np.abs(diff0)**2 / np.abs(data0)**2)))

        diffi = reconstr_stack.voxels - self.initial_stack.voxels

        metrics['dist_from_start'] = np.append(metrics['dist_from_start'], 
                                    np.sqrt(np.mean(np.abs(diffi)**2)))

class EvoView(object):
    """Class which stores the components of an evolution view to avoid garbage collection."""

    def __init__(self, storer, plot_ax=None, slider_ax=None, fig=None):
        if plot_ax is None: # need somewhere to plot
            if fig is None:
                fig = plt.figure()
            plot_ax = fig.add_subplot(111)
            plt.sca(plot_ax)
            plt.subplots_adjust(bottom=0.25)
        self.storer = storer
        self.plot_ax = plot_ax
        if slider_ax is None: # somewhere to put the slider
            l, b, w, _ = plot_ax.get_position().bounds
            slider_ax = plt.axes([l, b-0.1, w, 0.03])
        self.slider = Slider(slider_ax, 'Reconstruction step', self.storer.step_rec[0], 
                            self.storer.step_rec[-1], valinit=self.storer.step_rec[-1], valstep=1)
        self.slider.on_changed(self.update)
        self.updates = []

        l, b, w, _ = self.slider.ax.get_position().bounds
        button_ax = plt.axes([l+2/5*w, b-0.05, w/5, 0.04])
        self.play_button = Button(button_ax, 'Play')
        self.play_button.on_clicked(self.play)
        self.playing = False

    def update(self, val):
        for func in self.updates:
            func(self, self.storer, val)

    def play(self, event):
        if not self.playing:
            self.anim = animation.FuncAnimation(self.plot_ax.figure, self._advance_slider)
            self.play_button.label.set_text('Stop')
            self.playing = True
        else:
            self.anim.event_source.stop()
            self.play_button.label.set_text('Play')
            self.playing = False

    def _advance_slider(self, frame):
        if self.slider.val == self.storer.step_rec[-1]:
            self.slider.set_val(self.storer.step_rec[0])
        else:
            self.slider.set_val(self.slider.val + 1)

    def clear(self):
        self.slider.ax.remove()
        self.play_button.ax.remove()
        self.plot_ax.clear()

class InteractiveMap(object):
    
    def __init__(self, controller, map_mat, fig=None, map_ax=None, plot_ax=None):
        self.fig = fig
        self.create_view(map_ax, plot_ax)
        self.controller = controller
        im_ax = self.controller.map(map_mat, self.map_ax)
        self.cid = im_ax.figure.canvas.mpl_connect('button_press_event', self.voxel_plot)

    def create_view(self, map_ax=None, plot_ax=None):
        if map_ax is None or plot_ax is None:
            if self.fig is None:
                self.fig = plt.figure()
            self.fig.clear()
            self.map_ax = self.fig.add_subplot(121)
            self.plot_ax = self.fig.add_subplot(122)
        else:
            self.map_ax = map_ax
            self.plot_ax = plot_ax
        plt.sca(self.plot_ax)
        plt.subplots_adjust(bottom=0.25)

    def voxel_plot(self, event):
        if event.inaxes is not self.map_ax.axes: return
        x = int(event.xdata + 0.5)
        y = int(event.ydata + 0.5)
        self.controller.voxel_plot(x, y, self.plot_ax)

class DistanceRecorder(object):
    """Reconstruction analyzer which stores all the intermediary errors.

    Parameters
    ----------

    Attributes
    ----------
    RMSE: float vector
    """

    def __init__(self, ref_points=[], masks=[], distance=rms_diff_stack):
        self.ref_points = ref_points
        self.masks = masks
        # function that calculates the distance between two ImageStack instances
        self.distance = distance
        self.reset()

    def reset(self):
        self.dist_ref_to_rec = np.empty((len(self.ref_points), 0))
        self.dist_ref_to_proj = np.empty((len(self.ref_points), 0))
        self.dist_proj_to_rec = []        
        self.step_rec = []
        self.step_proj = []

    def on_signal_projected(self, i, reconstr_stack):
        self.step_proj.append(i)
        ref_proj = [[self.distance(reconstr_stack, ref)] 
                    for ref in self.ref_points]
        self.dist_ref_to_proj = np.append(self.dist_ref_to_proj, 
                                          ref_proj, axis=1)
        self.last_proj = deepcopy(reconstr_stack)

    def on_data_inserted(self, i, reconstr_stack):
        if i == 0:
            self.reset()
        else:
            self.dist_proj_to_rec = np.append(
                self.dist_proj_to_rec, 
                self.distance(reconstr_stack, self.last_proj)
            )
        self.step_rec.append(i)
        ref_rec = [[self.distance(reconstr_stack, ref)] 
                    for ref in self.ref_points]
        self.dist_ref_to_rec = np.append(self.dist_ref_to_rec, 
                                          ref_rec, axis=1)
