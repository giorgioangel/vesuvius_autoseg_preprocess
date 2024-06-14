# 2024 Giorgio Angelotti

# I enabled in the env variables CUPY_ACCELERATORS="cub,cutensor" and CUPY_TF32=1, perform in float32
# 3D implementation of Noise Robust Gradient Operators by Pavel Holoborodko http://www.holoborodko.com/pavel/image-processing/edge-detection/

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import convolve, map_coordinates

def divide_nonzero(array1, array2):
    precision = array1.dtype
    denominator = np.copy(array2)
    denominator[denominator == 0] = np.finfo(precision).tiny
    return np.divide(array1, denominator)

@cp.fuse()
def calculate_det(a,b,c,d,e,f):
    det = cp.abs(a*(b*c-e**2)-d*(d*c-e*f)+f*(d*e-b*f))
    return det

@cp.fuse()
def normalize_gradient(a,b):
    return a/b
    
def grad_and_det(volume, precision):
    a = cp.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=precision)
    b = cp.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=precision)
    c = cp.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=precision)
    d = cp.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=precision)
    e = cp.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=precision)
    f = cp.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=precision)
    
    # Using Pavel Hodborodko's derivative
    pavel_1d = cp.array([2,1,-16,-27,0,27,16,-1,-2], dtype=precision)  # Derivative approximation
    pavel_1d_smooth = cp.array([1, 4, 6, 4, 1], dtype=precision)  # Smoothing
    pavel_1d_2nd = cp.array([-7,12,52,-12,-90,-12,52,12,-7], dtype=precision)
    
    # Create 3D kernels by outer products and normalization
    kz = cp.outer(cp.outer(pavel_1d, pavel_1d_smooth), pavel_1d_smooth).reshape(9, 5, 5)/ (96*16*16)
    ky = cp.outer(cp.outer(pavel_1d_smooth, pavel_1d), pavel_1d_smooth).reshape(5, 9, 5)/ (96*16*16)
    kx = cp.outer(pavel_1d_smooth, cp.outer(pavel_1d_smooth, pavel_1d)).reshape(5, 5, 9)/ (96*16*16)
    kzz = cp.outer(cp.outer(pavel_1d_2nd, pavel_1d_smooth), pavel_1d_smooth).reshape(9, 5, 5)/ (192*16*16)
    kyy = cp.outer(cp.outer(pavel_1d_smooth, pavel_1d_2nd), pavel_1d_smooth).reshape(5, 9, 5)/ (192*16*16)
    kxx = cp.outer(pavel_1d_smooth, cp.outer(pavel_1d_smooth, pavel_1d_2nd)).reshape(5, 5, 9)/ (192*16*16)

    gradient = cp.zeros((3, volume.shape[0], volume.shape[1], volume.shape[2]), dtype=cp.float32)

    a = convolve(volume, kzz)
    b = convolve(volume, kyy)
    c = convolve(volume, kxx)

    gradient[0] = convolve(volume, kz)
    d = convolve(gradient[0], ky)
    f = convolve(gradient[0], kx)

    gradient[1] = convolve(volume, kx)
    e = convolve(gradient[1], kx)

    gradient[2] = convolve(volume, kx)
    
    det = calculate_det(a,b,c,d,e,f)

    del a, b, c, d, e, f
    
    magnitude = cp.sqrt(gradient[0]**2+gradient[1]**2+gradient[2]**2)

    gradient = normalize_gradient(gradient, magnitude)
    
    #return joint_hessian.get(), zero_mask.get()
    return det, gradient, magnitude

def nms_3d(magnitude, grad):
    """
    Applies Non-Maximum Suppression on a 3D volume using interpolation along gradient directions with CUDA optimization.

    Parameters:
    - magnitude: 3D cupy array representing the magnitude of gradients.
    - grad: 3D cupy array of shape (3, *magnitude.shape) representing gradient vectors.

    Returns:
    - nms_volume: 3D cupy array after applying NMS.
    """

    # Get the shape of the volume
    z_dim, y_dim, x_dim = magnitude.shape
    
    # Create meshgrid of indices
    Z, Y, X = cp.meshgrid(cp.arange(z_dim), cp.arange(y_dim), cp.arange(x_dim), indexing='ij')

    # Calculate continuous indices for forward and backward positions based on gradients
    forward_indices = cp.array([Z, Y, X]) + grad
    backward_indices = cp.array([Z, Y, X]) - grad

    # Interpolate the magnitude values at these continuous indices
    forward_values = map_coordinates(magnitude, forward_indices, order=1, mode='nearest')
    backward_values = map_coordinates(magnitude, backward_indices, order=1, mode='nearest')

    # Apply conditions for NMS using CuPy logical functions
    condition1 = cp.logical_and(magnitude >= forward_values, magnitude > backward_values)
    condition2 = cp.logical_and(magnitude > forward_values, magnitude >= backward_values)
    mask = cp.logical_or(condition1, condition2)

    # Apply mask to set NMS volume
    magnitude[~mask] = 0

    return magnitude
    
def edge_detection(volume, precision, threshold_grad=0.2, threshold_det=0.001):
    det, gradient, magnitude = grad_and_det(volume, precision)
    magnitude = nms_3d(magnitude, gradient)
    det /= det.max()
    magnitude /= magnitude.max()
    mask = (det < threshold_det) & (magnitude > threshold_grad)
    del det, gradient, magnitude
    edges = cp.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=cp.uint8)
    edges[mask] = 1
    return edges