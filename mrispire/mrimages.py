from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

class ArrayProperty(np.ndarray):
    """NumPy ndarray subclass to be used as the property of another class.
    
    This class was created to solve the following issue: when the property 
    of a class is a NumPy array, assigning new values to it through array 
    indexing does not call the setter of that property. For example:

    obj.arr_property[:] = 0

    When this line is executed, the getter of arr_property returns a NumPy 
    array the values of which are all set to 0. This is not the desired 
    behaviour. We would like the setter of arr_property to be executed instead.
    
    This class allows that. The getter must return the array property cast 
    as an ArrayProperty. The returned instance keeps the property setter as
    a callback function executed when it is assigned values through indexing. 

    The problem is described in detail here:
    https://stackoverflow.com/questions/24890393/

    """

    def __new__(cls, input_array, arr_setter):
        # The use of __new__ is detailed at:
        # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj =  np.asarray(input_array).view(cls)
        # add the array property setter as an attribute to the created 
        # instance; the setter is a callback to the class where the property 
        # is defined
        obj.arr_setter = arr_setter
        # Finally, we must return the newly created object:
        return obj
    
    def __setitem__(self, key, value):
        # call the __setitem__ method of the superclass;
        # this modifies the array according to the indexed assignment
        super().__setitem__(key, value)
        # if the instance has an array setter attribute call the 
        # setter with the new array as argument
        if getattr(self, 'arr_setter', None) is not None:
            self.arr_setter(np.array(self))

class ImageSeries(object):
    """Class representing the images collected in an MRI sequence.
    
    Parameters
    ----------
    image_space: ndarray
        (PxMxN) array of complex values representing P (MxN) MRI images
        collected at P different values along the P (parameter encoding) 
        dimension e.g. time.
    background_mask: ndarray, optional
        (MxN) array of bool values where False marks the voxels belonging to 
        the background of the images in the series. The default is an array 
        of ones i.e. no background.
    p_values: ndarray, optional
        (P) vector of float values representing the physical values of the 
        encoding dimension at which the images were taken e.g. time [msec]. 
        The default is a uniform range of values from 0 to P-1.
    x_limits: tuple of float. optional
        Two values specifying the physical coordinates of the left corner and
        right corner of the images along the x dimension (last in the image
        space with N corresponding voxels) e.g. values in [m]. 
        Default is (0, 1).
    y_limits: tuple of float, optional
        Two values specifying the physical coordinates of the top corner and
        bottom corner of the images along the y dimension (last in the image
        space with M corresponding voxels) e.g. values in [m].
        Default is (0, 1).

    Attributes
    ----------
    image_space: ndarray
        (PxMxN) array of complex values representing the image space.
    background_mask: ndarray
        (MxN) array of bool values masking out the background.
    p_values: ndarray
        (P) vector of coordinates along the encoding dimension.
    x_limits: tuple of float
        The coordinates of the limits of each image along the x dimension.
    y_limits: tuple of float
        The coordinates of the limits of each image along the y dimension.
    shape
    k_space
    voxels
    
    """
    
    def __init__(self, image_space, background_mask=None, p_values=None, 
                 x_limits=(0,1), y_limits=(0,1)):
        self.image_space = image_space
        if p_values is None:
            self.p_values = np.arange(self.image_space.shape[0])
        else:
            self.p_values = p_values
        if background_mask is None:
            self.background_mask = np.ones(image_space.shape[1:], dtype=bool)
        else:
            self.background_mask = background_mask
        self.x_limits = x_limits
        self.y_limits = y_limits
         
    @property
    def shape(self):
        """tuple of int: The shape of the image space array i.e. (P, M, N)."""
        
        return self.image_space.shape
        
    @property
    def k_space(self):
        """ndarray: (PxMxN) array of complex values representing the k-space.
        
        Only the image space of the MR image series is stored by the class.
        The k-space is calculated from it using FFT2 whenever required.

        """

        # calculate the k-space from the image space
        k_space_array = fftshift(fft2(self.image_space), (1, 2))
        # cast as an ArrayProperty and pass the setter of k-space to it
        k_space_array = ArrayProperty(k_space_array, self._k_space_setter)
        return k_space_array
    
    def _k_space_setter(self, k_space):
        # the class stores the data in image space only, therefore the image 
        # space is updated whenever changes to the k-space are made
        self.image_space = ifft2(ifftshift(k_space, (1, 2)))

    @k_space.setter
    def k_space(self, k_space):
        self._k_space_setter(k_space)
        
    @property
    def voxels(self):
        """ndarray: (PxMN) array of complex values representing the voxels.
        
        Returns a matrix in which each column represents the signal in a voxel
        along the parameter encoding dimension.

        """

        # flatten all the P images and return as an ArrayProperty
        voxels_array = self.image_space.reshape([self.shape[0], -1])
        voxels_array = ArrayProperty(voxels_array, self._voxels_setter)
        return voxels_array
    
    def _voxels_setter(self, voxels):
        # reshape as image space and modify the stored data
        self.image_space = voxels.reshape(self.shape)

    @voxels.setter
    def voxels(self, voxels):
        self._voxels_setter(voxels)

    def voxels_masked(self, mask=None):
        """Returns the voxels selected by the input mask in matrix format.
        
        Parameters
        ----------
        mask: ndarray, optional
            (MxN) array of bool values selecting the masked voxels. 
            The default is the background mask.

        Returns
        -------
            (PxZ) matrix in which each column represents the signal in a voxel
            along the parameter encoding dimension. Only the voxels that are
            not masked out are returned.
        
        """
        
        if mask is None:
            mask = self.background_mask
        return self.image_space[:, mask]

    def image_space_masked(self, mask=None):
        """Returns an image space where some voxels are masked out.
        
        Parameters
        ----------
        mask: ndarray, optional
            (MxN) array of bool values selecting the masked voxels. 
            The default is the background mask.

        Returns
        -------
            (PxMxN) array of complex values representing the image space in 
            which the voxels that are masked are replaced by 0.
        
        """

        if mask is None:
            mask = self.background_mask
        return self.image_space * mask

    def __sub__(self, other_im_stack):
        # implement subtraction of two image series
        diff_im_stack = deepcopy(self) # meta-data from this instance
        # subtract the two image spaces
        diff_im_stack.image_space = self.image_space - \
            other_im_stack.image_space
        return diff_im_stack
        
class UndersampledStack(ImageSeries):
    """Subclass of Image Stack with undersampled k-space data attached.

    Parameters
    ----------
    k_space_data: complex vector
        Vector containing the undersampled data corresponding to the
        k-space locations in mask.
    mask: bool 3D array
        The undersampling mask.

    Attributes
    ----------
    # TODO
    """

    def __init__(self, k_space_data, mask, p_values, background_mask=None, 
                 x_limits=(0,1), y_limits=(0,1)):
        self.mask = mask
        self.k_space_data = k_space_data
        image_space = np.zeros_like(mask, dtype=complex)
        super().__init__(image_space, p_values, background_mask=background_mask, 
                         x_limits=x_limits, y_limits=y_limits)
        self.checkout_data()

    def checkout_data(self, image_stack=None):
        if image_stack is None:
            image_stack = self
        image_stack.k_space[self.mask] = self.k_space_data

    @property
    def undersampling_factor(self):
        return self.mask.size / np.count_nonzero(self.mask)

    @staticmethod
    def from_complete_stack(complete_im_stack, yp_mask):
        """Create an UndersampledStack from an ImageSeries using the given yp mask.
        """

        mask = np.moveaxis(yp_mask, -1, 0)
        mask = np.tile(mask[:,:,np.newaxis], complete_im_stack.shape[2])
        undersamp_stack = UndersampledStack(complete_im_stack.k_space[mask], mask,
                                            p_values=complete_im_stack.p_values[:],
                                            background_mask=complete_im_stack.background_mask[:],
                                            x_limits=complete_im_stack.x_limits[:],
                                            y_limits=complete_im_stack.y_limits[:])
        return undersamp_stack

def rms_diff_stack(im_stack1, im_stack2):
    return rms_diff(im_stack1.voxels, im_stack2.voxels)

def rms_diff(data1, data2):
    return np.sqrt(np.mean(np.abs(data1-data2)**2))

def phantom_circles(sz):
    '''Creates a square phantom of 5 circles.

    Parameters
    ----------
    sz: int
        Size of the side of the square phantom.

    Returns
    -------
    ndarray
        (sz, sz) array of integers representing the phantom,
        where the background has value 0 and the circles have values
        1 to 5 from the largest to the smallest.
    '''

    def unit_circle(sz, x, y, r):
        xx, yy = np.indices((sz, sz))
        return (np.hypot(x - xx, y - yy) < r + 0.5).astype(int)

    phantom_ind = np.full((sz, sz), 0)
    r = np.round(sz / 256) # integer ratio of sz to 256
    phantom_ind += unit_circle(sz, 80*r, 80*r, 50*r) * 1
    phantom_ind += unit_circle(sz, 100*r, 180*r, 40*r) * 2
    phantom_ind += unit_circle(sz, 180*r, 160*r, 30*r) * 3
    phantom_ind += unit_circle(sz, 175*r, 100*r, 20*r) * 4
    phantom_ind += unit_circle(sz, 150*r, 70*r, 10*r) * 5
    return phantom_ind