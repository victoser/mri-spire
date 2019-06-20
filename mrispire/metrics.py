import numpy as np

def rms_diff(data1, data2):
    return np.sqrt(np.mean(np.abs(data1-data2)**2))

def rms_diff_series(im_stack1, im_stack2):
    return rms_diff(im_stack1.voxels, im_stack2.voxels)
