import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy import sparse

class SparseRepresentation(object):
    """Abstract class for all sparse representations."""

    def error_profile(self, data):
        pass

class GridDatabase(SparseRepresentation):
    """Class containing the grid database sparse representation.

    Parameters
    ----------
    database: ndarray, optional
        (PxA) matrix of float values, the columns of which represent the 
        the intensity profiles of a voxel along the parameter encoding
        dimension (e.g. time) at specific values of a parameter (e.g. T2) that 
        maps to that voxel. The default is an empty array.
    param_values: ndarray, optional
        (A) vector of float values specifying the physical parameter values 
        corresponding to the columns of the database. The default is a range
        from 0 to the number of columns.

    Attributes
    ----------
    database
    norms
    n_atoms

    """

    def __init__(self, database=np.array([[]]), param_values=None):
        self.database = database
        if param_values is not None:
            self.param_values = param_values
        else:
            self.param_values = np.arange(database.shape[1])

    @property
    def database(self):
        """ndarray: The database of signal profiles for different parameter 
        values.

        (PxA) matrix of float values, the columns of which represent the 
        the intensity profiles of a voxel along the parameter encoding
        dimension (e.g. time) at specific values of a parameter (e.g. T2) that 
        maps to that voxel.
        
        """

        # return the privately stored database
        return self._database

    @database.setter
    def database(self, database):
        # store the database privately
        self._database = database
        # whenever the database is stored/changed the setter also stores a
        # normalized version of it together with its norms
        self._norm_database = database / np.linalg.norm(database, axis=0)
        self._norms = np.linalg.norm(database, axis=0)

    @property
    def norms(self):
        """ndarray: (A) vector of float contaning the norms of the signals in 
        the database.
        """

        return self._norms

    @property
    def n_atoms(self):
        """int: The number A of atoms i.e. signals stored in the database."""

        return self._database.shape[1]

    def project(self, data):
        """Dot product projection on the database space.

        Parameters
        ----------
        data: ndarray
            (PxN) matrix of complex values in which the signals to be 
            projected are placed along the columns.

        Returns
        -------
        projected_data: ndarray
            (PxN) matrix of float values in which the signal projections of 
            the data are placed along the columns.
        coeff: ndarray
            (N) vector of complex representing the coefficients by which the 
            atoms are multiplied to obtain the projections.
        selected_atoms: ndarray
            (N) vector of the indices of the atoms onto which the columns of 
            the data were projected.
        rho_values: ndarray
            (N) vector of float representing the coefficients by which the 
            atoms are multiplied to obtain the projections.
        ph_values: ndarray
            (N) vector of float - the phases associated with the projections.
        param_values: ndarray
            (N) vector of float - the parameter values associated with the 
            selected atoms.

        """

        # calculate the dot products of the data with the normalized atoms and
        # select the atom with the largest product for each column
        dot_prod_all = self._norm_database.T.conj() @ data
        selected_atoms = np.argmax(np.abs(dot_prod_all), axis=0)
        # from each column select the maximum dot product
        selected_prod = dot_prod_all[selected_atoms, np.arange(data.shape[1])]
        # to get the rho values need to divide by norms
        coeff = selected_prod / self.norms[selected_atoms]
        # the rho and ph values are the magnitudes and the phases of the 
        # complex coefficients respectively 
        rho_values = np.abs(coeff)
        ph_values = np.angle(selected_prod)
        param_values = self.param_values[selected_atoms]
        projected_data = self.database[:, selected_atoms] * coeff

        return projected_data, coeff, selected_atoms, rho_values, ph_values, \
               param_values

    def sparsify(self, data):
        """Sparsifying method of the sparse representation.

        Parameters
        ----------
        data: ndarray
            (PxN) matrix of complex values in which the signals to be 
            sparsified are placed along the columns.

        Returns
        -------
        ndarray
            (PxN) matrix of float values in which the sparsified signals of 
            the data are placed along the columns.

        """

        # wrap around the project method
        return self.project(data)[0]

    def fit_param(self, data_to_fit, threshold_signal=1e-5):
        """Fit parameter values to data based on projection onto the database.
        
        Parameters
        ----------
        data_to_fit: ndarray
            (Px...) array of complex values in which the signals to be 
            sparsified are placed along the columns.
        threshold_signal: float
            Threshold value for a signal to be fit. If all of the intensity
            values of a signal fall under this threshold a zero will be 
            returned.

        Returns
        -------
        param_map: ndarray
            (...) array of float - the parameter values associated with the 
            selected atoms.
        rho_map: ndarray
            (...) array of float representing the coefficients by which the 
            atoms are multiplied to obtain the projections.
        ph_map: ndarray
            (...) array of float - the phases associated with the projections.

        """

        shape_map = data_to_fit.shape[1:]
        # intialize with maps of zeros
        rho_map = np.zeros(shape_map)
        ph_map = np.zeros(shape_map)
        param_map = np.zeros(shape_map)
        # select only the signals strong enough to be considered
        mask_nonzero = np.any(np.abs(data_to_fit) > threshold_signal, axis=0)
        # use the project method
        rho_map[mask_nonzero], ph_map[mask_nonzero], param_map[mask_nonzero] \
            = self.project(data_to_fit[:, mask_nonzero])[-3:]
        return param_map, rho_map, ph_map

class DictionaryKSVD(SparseRepresentation):

    def __init__(self, n_rep=3, *, dictionary=None):
        self.n_rep = n_rep
        if dictionary is not None:
            self.dictionary = dictionary
            
    @property
    def n_atoms(self):
        return self.dictionary.shape[1]

    def project(self, data):
        coeff = OMP_vic(self.dictionary, data, self.n_rep)
        projected_data = self.dictionary @ coeff
        return projected_data, coeff

class PrincipalComponents(SparseRepresentation):

    def __init__(self, n_pc=None, train_data=None, n_pc_stored=None, norm_data=True):
        self.n_pc = n_pc
        if train_data is not None:
            self.train(train_data, n_pc_stored=n_pc_stored, norm_data=norm_data)

    @property
    def pcs(self):
        return self.pcs_stored[:, 0:self.n_pc]

    def train(self, train_data, n_pc_stored=None, norm_data=True):

        if norm_data:
            train_data = train_data / np.linalg.norm(train_data, axis=0)
        u, s, vh = svd(train_data, full_matrices=False)
        if n_pc_stored is None:
            n_pc_stored = u.shape[1]
        else:
            n_pc_stored = n_pc_stored
        if self.n_pc is None:
            self.n_pc = n_pc_stored        
        self.pcs_stored = u[:, 0:n_pc_stored]
        coeff = (np.diag(s) @ vh)[0:self.n_pc]
        err = np.sqrt(((self.pcs@coeff - train_data)**2).mean())
        return self.pcs, coeff, err

    def project(self, data):
        """Dot product projection on the PC space.

        Parameters
        ----------
        data: float
            Matrix in which the signals to be projected 
            are placed along the columns.

        Returns
        -------
        projected_data: float
            Matrix in which the signal projections of 
            the data are placed along the columns.
        coeff: float
            Matrix with the PC coefficients corresponding
            to each signal in data placed along columns.
        """
        coeff = self.pcs.T.conj() @ data
        projected_data = self.pcs @ coeff
        return projected_data, coeff

def OMP_vic(dictionary, data, L, threshold=1e-6):
    """Orthogonal mathcing pursuit projection onto dictionary.

    Parameters
    ----------
    dictionary: ndarray   
        Matrix representing a dictionary with entries along
        the columns. The columns need to be normalized.
    data: ndarray
        Matrix in which the signals to be projected 
        are placed along the columns. It can be complex.
    L: int
        The number of dictionary entries used in 
        the sparse representation.
    threshold: float
        Threshold to stop OMP for a column if the norm
        of its residual is less than the threshold.
    
    Returns
    -------
    A: sparse.csc_matrix
        Matrix with the OMP coefficients corresponding
        to each signal in data placed along columns.
    """
    # n is the number of dimensions of the signal
    # P is the number of data points
    n, P = data.shape
    K = dictionary.shape[-1]
    A = sparse.lil_matrix((K, P), dtype=data.dtype)
    # matrix containing up to L atom indices for each
    # data point corresponding to the selected atoms
    # for that point
    indx = np.full((L, P), -1, dtype=int)

    remain_n = P
    remain_indx = np.array(range(P)) # indices of remaining data
    residual = data.copy()

    for i in range(L):
        # do the dot product; dictionary assumed normalized
        proj = dictionary.T.conj() @ residual[:, remain_indx]
        # choose the atoms giving the maximum dot product for
        # each data point
        pos = np.argmax(np.abs(proj), axis=0)
        
        ok = np.full(remain_n, False)
        # check if the atom is repeating for each data point
        # if it does it means the residual for that signal 
        # is zero so mark it as ok and remove it from residual
        for j in range(remain_n):
            if pos[j] in indx[:, remain_indx[j]]:
                ok[j] = True
                
        # keep the data points that are not ok
        remain_indx = remain_indx[np.logical_not(ok)]
        remain_n = remain_n - np.sum(ok)
        # break if there is no remaining data
        if remain_n == 0:
            break
        
        # update the indices
        indx[i, remain_indx] = pos[np.logical_not(ok)]

        for j in remain_indx:
            # calculate the coefficients of the projection of the
            # data point onto the linear span of the atoms found so far
            a = np.linalg.pinv(dictionary[:, indx[:i+1, j]]) @ data[:, j]
            # update the sparse matrix of coefficients
            A[indx[:i+1, j], j] = a[:, np.newaxis]
            # update the residual
            residual[:, j] = data[:, j] - dictionary[:, indx[:i+1, j]] @ a

    return sparse.csc_matrix(A)

def OMP_non_sparse(dictionary, data, L):
    """Orthogonal mathcing pursuit projection onto dictionary.

    Parameters
    ----------
    dictionary: float matrix   
        Matrix representing a dictionary with entries along
        the columns. The columns need to be normalized.
    data: float matrix
        Matrix in which the signals to be projected 
        are placed along the columns.
    L: int
        The number of dictionary entries used in 
        the sparse representation.
    
    Returns
    -------
    A: float matrix
        Matrix with the OMP coefficients corresponding
        to each signal in data placed along columns.
    """
    n, P = data.shape
    K = dictionary.shape[-1]
    A = np.zeros((K, P), dtype=data.dtype)
    indx = np.full((L, P), -1, dtype=int)

    remain_n = P
    remain_indx = np.array(range(P)) # indices of remaining data
    residual = data.copy()

    for i in range(L):
        proj = dictionary.T.conj() @ residual
        pos = np.argmax(np.abs(proj), axis=0)
        
        ok = np.full(remain_n, False)
        # check if the atom is repeating
        # TODO: need to also check a threshold of error
        for j in range(remain_n):
            if pos[j] in indx[:,remain_indx[j]]:
                ok[j] = True
        remain_indx = remain_indx[np.logical_not(ok)]
        remain_n = remain_n - np.sum(ok)
        # break if there is no new atom to project onto
        if remain_n == 0:
            break

        indx[i,remain_indx] = pos[np.logical_not(ok)]

        indx[:i+1,remain_indx] = np.sort(indx[:i+1,remain_indx], axis=0)
        # C does not include -1 indices because only the indices of the remaining data are sorted
        C, ic = np.unique(indx[:i+1,remain_indx], axis=1, return_inverse=True)
        residual = np.zeros((n,remain_n), dtype=data.dtype)
        for j in range(C.shape[1]):
            selected_data = remain_indx[ic==j]
            a = np.linalg.pinv(dictionary[:,C[:,j]]) @ data[:,selected_data]

            temp = np.zeros((K,len(selected_data)), dtype=data.dtype)
            temp[C[:,j],:] = a
            A[:,selected_data] = temp

            residual[:,ic==j] = data[:,selected_data] - dictionary[:,C[:,j]] @ a

    return A


