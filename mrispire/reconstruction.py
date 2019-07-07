from copy import deepcopy
import numpy as np

from .metrics import *

class SparseReconstructor(object):
    """Class performing reconstruction based on a sparse representation.
    
    Parameters
    ----------
    sparse_representation: SparseRepresentation
        The sparse representation used as a prior in reconstruction.
    n_iter: int, optional
        Number of iterations to carry out in the reconstruction.
        The default is 20.
    progress_analysis: bool, optional
        Whether to trigger events to analyze the reconstruction in each 
        iteration. The default is True.
    rec_analyzers: list, optional
        List of objects that implement the reconstruction analyzer methods:
        - on_signal_sparsified(i, reconstr_series)
        - on_data_inserted(i, reconstr_series)
        The default is an empty list.
    mask_background: bool, optional
        Whether to remove the background after each sparsifying step.
        The default is False.
    
    Attributes
    ----------
    sparse_representation: SparseRepresentation
        The sparse representation used as a prior in reconstruction.
    n_iter: int
        Number of iterations to carry out in the reconstruction.
    progress_analysis: bool
        Whether to trigger events for analysis in each iteration.
    rec_analyzers: list
        List of objects that implement the reconstruction analysis methods:
        - on_signal_sparsified(i, reconstr_series)
        - on_data_inserted(i, reconstr_series)
    mask_background: bool
        Whether to remove the background after each sparsifying step.
    end_after_sp: bool
        Whether to end the reconstruction after the sparsifying step in the
        last loop. It is set to False by default.

    """

    def __init__(self, sparse_representation=None, n_iter=20,
                 progress_analysis=True, rec_analyzers=[], 
                 mask_background=False):
        self.sparse_representation = sparse_representation
        self.end_after_sp = False
        self.n_iter = n_iter
        self.progress_analysis = progress_analysis
        self.rec_analyzers = rec_analyzers
        self.mask_background = mask_background
        self.end_after_sp = False

    def reconstruct(self, undersampled_series, series_to_rec=None, 
                    in_place=False):
        """Reconstruct the undersampled series.

        Parameters
        ----------
        undersampled_series: UndersampledSeries
            Undersampled k-space data to be used in the data consistency step.
        series_to_rec: ImageSeries, optional
            Initial image series to reconstruct. By default, it is the 
            undersampled series i.e. the missing k-space data is filled 
            with zeros.
        in_place: bool, optional
            Whether to keep the original series unmodified or do the 
            recontruction within the given initial object. Default is False.

        Returns
        -------
        ImageSeries
            Reconstructed image series. If in_place is set to True it is 
            either the series_to_rec if specified in input, or the
            undersampled_series. Otherwise it is a copy of the series_to_rec 
            object created by the method.

        """

        # if no initial series provided, use the undersampled series
        if series_to_rec is None:
            series_to_rec = undersampled_series
        # if not in_place, make a copy of the series_to_rec on which to 
        # carry out the reconstruction
        if in_place:
            reconstr_series = series_to_rec
        else:
            reconstr_series = deepcopy(series_to_rec)

        # analyze if required
        if self.progress_analysis:
            for analyzer in self.rec_analyzers:
                # send step number 0 to signal initial data
                analyzer.on_data_inserted(0, reconstr_series)

        for i in range(self.n_iter):
            # sparsify using the sparse representation
            reconstr_series.voxels[:] = self.sparse_representation.sparsify(
                np.array(reconstr_series.voxels))
            # mask background
            if self.mask_background:
                reconstr_series.image_space[
                    :, reconstr_series.background_mask
                ] = 0

            if self.progress_analysis:
                for analyzer in self.rec_analyzers:
                    # send step number which is i+1
                    analyzer.on_signal_sparsified(i+1, reconstr_series)

            # break at the last step here if required
            if i==self.n_iter-1 and self.end_after_sp:
                break

            # fill in the data
            undersampled_series.checkout_data(reconstr_series)

            if self.progress_analysis:
                for analyzer in self.rec_analyzers:
                    # send step number which is i+1
                    analyzer.on_data_inserted(i+1, reconstr_series)
                
        return reconstr_series

class Storer(object):
    """Reconstruction analyzer which stores all the intermediary results.

    Attributes
    ----------
    stored_sparse: ndarray
        4D array that stores the series of reconstructions immediately after 
        the sparsifying step.
    stored_reconstructions: ndarray
        4D array that stores the series of reconstructions after the 
        data consistency is imposed.
    step_sp: list of int
        The steps of reconstruction at which the sparsified reconstructions 
        were captured.
    step_rec: list of int
        The steps of reconstruction at which the data consistent 
        reconstructions were captured.

    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.stored_sparse = np.empty((0))
        self.stored_reconstructions = np.empty((0))
        # the steps of reconstruction at which the sparsified reconstructions 
        # were captured
        self.step_sp = []
        # the steps of reconstruction at which the data consistent 
        # reconstructions were captured
        self.step_rec = []

    def on_signal_sparsified(self, i, reconstr_series):
        proj_list = list(self.stored_sparse)
        proj_list.append(reconstr_series.image_space)
        self.stored_sparse = np.array(proj_list)
        self.step_sp.append(i)

    def on_data_inserted(self, i, reconstr_series):
        if i == 0:
            self.reset()
        rec_list = list(self.stored_reconstructions)
        rec_list.append(reconstr_series.image_space)
        self.stored_reconstructions = np.array(rec_list)
        self.step_rec.append(i)

class DistanceRecorder(object):
    """Reconstruction analyzer which calculates distances of the 
    reconstruction from specified reference ImageSeries points.

    Parameters
    ----------
    ref_points: list of ImageSeries
        ImageSeries instances which are used as reference points to calculate
        distances to the reconstructed series during the reconstruction.
    distance: function, optional
        Function with signature:
            distance(ImageSeries, ImageSeries)
        which implements a way to calculate distances between two ImageSeries
        instances. The default is rms_diff_series.

    Attributes
    ----------
    ref_points: list of ImageSeries
        ImageSeries instances which are used as reference points to calculate
        distances to the reconstructed series during the reconstruction.
    distance: function
        Function with signature:
            distance(ImageSeries, ImageSeries)
        which implements a way to calculate distances between two ImageSeries
        instances.  
    dist_ref_to_rec: ndarray
        Matrix in which each row represents the recorded distances from the 
        corresponding reference point in ref_points to the reconstruction
        after the data consistency steps.
    dist_ref_to_proj: ndarray
        Matrix in which each row represents the recorded distances from the 
        corresponding reference point in ref_points to the reconstruction
        after the sparsifying steps.
    dist_proj_to_rec: ndarray
        Vector containing the recorded distances from the latest projection
        to the reconstruction after the data consistency steps.
    step_sp: list of int
        The steps of reconstruction at which the sparsified reconstructions 
        were captured.
    step_rec: list of int
        The steps of reconstruction at which the data consistent 
        reconstructions were captured.

    
    """

    def __init__(self, ref_points=[], distance=rms_diff_series):
        self.ref_points = ref_points
        # function that calculates the distance between two ImageSeries 
        # instances
        self.distance = distance
        self.reset()

    def reset(self):
        self.dist_ref_to_rec = np.empty((len(self.ref_points), 0))
        self.dist_ref_to_proj = np.empty((len(self.ref_points), 0))
        self.dist_proj_to_rec = []        
        self.step_rec = []
        self.step_sp = []

    def on_signal_sparsified(self, i, reconstr_series):
        self.step_sp.append(i)
        # calculate distances to reference points
        ref_proj = [[self.distance(reconstr_series, ref)] 
                    for ref in self.ref_points]
        # append to stored distances
        self.dist_ref_to_proj = np.append(self.dist_ref_to_proj, 
                                          ref_proj, axis=1)
        # keep a copy of the latest projection onto the sparse representation
        self.last_proj = deepcopy(reconstr_series)

    def on_data_inserted(self, i, reconstr_series):
        if i == 0:
            self.reset()
        else:
            # calculate the distance between the data consistent series and 
            # the latest projected/sparsified series
            self.dist_proj_to_rec = np.append(
                self.dist_proj_to_rec, 
                self.distance(reconstr_series, self.last_proj)
            )
        self.step_rec.append(i)
        # calculate distances to reference points
        ref_rec = [[self.distance(reconstr_series, ref)] 
                    for ref in self.ref_points]
        # append to stored distances
        self.dist_ref_to_rec = np.append(self.dist_ref_to_rec, 
                                          ref_rec, axis=1)

