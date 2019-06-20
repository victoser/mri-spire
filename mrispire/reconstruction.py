from copy import deepcopy
import numpy as np

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
        List of objects that implement the reconstruction analysis methods:
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
        """Recontstruct the undersampled series.

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

    def on_signal_sparsified(self, i, reconstr_series):
        proj_list = list(self.stored_projections)
        proj_list.append(reconstr_series.image_space)
        self.stored_projections = np.array(proj_list)
        self.step_proj.append(i)

    def on_data_inserted(self, i, reconstr_series):
        if i == 0:
            self.reset()
        rec_list = list(self.stored_reconstructions)
        rec_list.append(reconstr_series.image_space)
        self.stored_reconstructions = np.array(rec_list)
        self.step_rec.append(i)

class DistanceRecorder(object):
    """Reconstruction analyzer which stores all the intermediary errors.

    Parameters
    ----------

    Attributes
    ----------
    RMSE: float vector
    """

    def __init__(self, ref_points=[], masks=[], distance=rms_diff_series):
        self.ref_points = ref_points
        self.masks = masks
        # function that calculates the distance between two ImageSeries instances
        self.distance = distance
        self.reset()

    def reset(self):
        self.dist_ref_to_rec = np.empty((len(self.ref_points), 0))
        self.dist_ref_to_proj = np.empty((len(self.ref_points), 0))
        self.dist_proj_to_rec = []        
        self.step_rec = []
        self.step_proj = []

    def on_signal_sparsified(self, i, reconstr_series):
        self.step_proj.append(i)
        ref_proj = [[self.distance(reconstr_series, ref)] 
                    for ref in self.ref_points]
        self.dist_ref_to_proj = np.append(self.dist_ref_to_proj, 
                                          ref_proj, axis=1)
        self.last_proj = deepcopy(reconstr_series)

    def on_data_inserted(self, i, reconstr_series):
        if i == 0:
            self.reset()
        else:
            self.dist_proj_to_rec = np.append(
                self.dist_proj_to_rec, 
                self.distance(reconstr_series, self.last_proj)
            )
        self.step_rec.append(i)
        ref_rec = [[self.distance(reconstr_series, ref)] 
                    for ref in self.ref_points]
        self.dist_ref_to_rec = np.append(self.dist_ref_to_rec, 
                                          ref_rec, axis=1)
