from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

class KSpace(np.ndarray):

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.image_stack.k_space = np.array(self)

class Voxels(np.ndarray):

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.image_stack.voxels = np.array(self)

class ImageStack(object):
    """Class representing the images collected in an MRI sequence.
    
    Parameters
    ----------
    p_values: NumPy vector
    
    Attributes
    ----------
    p_values: NumPy vector
    """
    
    def __init__(self, image_space, p_values=None, background_mask=None, x_limits=(0,1), y_limits=(0,1)):
        self.image_space = image_space
        if p_values is None:
            self.p_values = np.arange(self.image_space.shape[0])
        else:
            self.p_values = p_values
        if background_mask is None:
            self.background_mask = np.zeros(image_space.shape[1:], dtype=bool)
        else:
            self.background_mask = background_mask
        self.x_limits = x_limits
        self.y_limits = y_limits
         
    @property
    def shape(self):
        return self.image_space.shape
        
    @property
    def k_space(self):
        k_space_array = fftshift(fft2(self.image_space), (1, 2))
        k_space_array = k_space_array.view(KSpace)
        k_space_array.image_stack = self
        return k_space_array
        
    @k_space.setter
    def k_space(self, k_space):
        self.image_space = ifft2(ifftshift(k_space, (1, 2)))
        
    @property
    def voxels(self):
        voxels_array = self.image_space.reshape([self.shape[0], -1])
        voxels_array = voxels_array.view(Voxels)
        voxels_array.image_stack = self
        return voxels_array
    
    @voxels.setter
    def voxels(self, voxels):
        self.image_space = voxels.reshape(self.shape)

    def voxels_masked(self, mask=None):
        if mask is None:
            mask = self.background_mask
        return self.image_space[:, np.logical_not(mask)]

    def image_space_masked(self, mask=None):
        if mask is None:
            mask = self.background_mask
        return self.image_space * np.logical_not(mask)

    def view_stack(self, im_ax=None, slider_ax=None, fig=None):
        if im_ax is None:
            if fig is None:
                fig = plt.figure()
            im_ax = fig.add_subplot(111)
            plt.subplots_adjust(bottom=0.25)
        im_ax.imshow(np.abs(self.image_space[0]))
        if slider_ax is None:
            l, b, w, _ = im_ax.get_position().bounds
            slider_ax = plt.axes([l, b-0.1, w, 0.03])
        sindex = Slider(slider_ax, 'Image index', 1., float(self.image_space.shape[0]), valstep=1.)
        def update(val):
            im_indx = int(val) - 1
            im_ax.imshow(np.abs(self.image_space[im_indx]))
            im_ax.figure.canvas.draw()
            plt.show()
        sindex.on_changed(update)
        return sindex # this makes sure the slider is not garbage collected wherever the function is called

    def plot_images(self, im_to_plot=None, fig_seq=None):
        if im_to_plot is None:
            im_to_plot = range(self.shape[0])
        if fig_seq is None:
            fig_seq = [plt.figure() for _ in range(len(im_to_plot))]
        for im, fig in zip(im_to_plot, fig_seq):
            plt.figure(fig.number)
            plt.imshow(np.abs(self.image_space[im]))
            plt.colorbar()
            plt.title('Parameter value: ' + str(self.p_values[im]))

    def __sub__(self, other_im_stack):
        diff_im_stack = deepcopy(self)
        diff_im_stack.image_space = self.image_space - \
            other_im_stack.image_space
        return diff_im_stack
        
class UndersampledStack(ImageStack):
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
        """Create an UndersampledStack from an ImageStack using the given yp mask.
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