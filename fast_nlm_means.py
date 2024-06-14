# Slightly modified from Brett Olsen code https://github.com/caethan/vesuvius_image/blob/master/vesuvius/fast_nl_means.py

import numpy as np
try:
    import cupy as cp
    xp = cp
    DEVICE = 'gpu'
except ImportError:
    xp = np
    DEVICE = 'cpu'
import sys

# Define some constants for a fast exponential function
EXP_A = 1512775.3951951856938 # 2^20 / ln(2)
EXP_BC = 1072632447 # 1023 * 2 ^ 20 - 60801
if sys.byteorder == "little":
    EXPSLICE = slice(1, None, 2)
else:
    EXPSLICE = slice(0, None, 2)


squared_minus = cp.ElementwiseKernel(
        'float64 x, float64 y',
        'float64 z',
        'z = x * x - y',
        'squared_minus'
    )

fast_exp_op = cp.ElementwiseKernel(
        'float64 y, float64 a, float64 b',
        'int32 z',
        'z = a * y + b',
        'fast_exp_op'
    )

def fast_approx_exp(y):
    y = xp.ascontiguousarray(y, dtype=xp.float64)
    y.view(xp.int32)[..., EXPSLICE] = fast_exp_op(y, EXP_A, EXP_BC)
    
    
def iter_directions(d):
    cache = set([(0, 0)])
    for t_pln in range(-d, d + 1):
        for t_row in range(-d, d + 1):
            for t_col in range(0, d + 1):
                if t_col == 0:
                    if (t_pln, t_row) not in cache:
                        cache.add((t_pln, t_row))
                        cache.add((-t_pln, -t_row))
                        yield (t_pln, t_row, t_col)
                else:
                    yield (t_pln, t_row, t_col)

def fast_nl_means_denoising_3d_gpu(image, s=5, d=7, h=0.1, var=0.0):
    if s % 2 == 0:
        s += 1  # We want an odd value so we have a symmetric patch

    offset = s // 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    pad_size = offset + d + 1
    pad_to_offset = pad_size - offset

    padded = cp.ascontiguousarray(
            cp.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), mode='reflect'),
            dtype=xp.float64,
        )
        
    summed_area_table = cp.zeros(
            ((image.shape[0] + 2 * offset), (image.shape[1] + 2 * offset), (image.shape[2] + 2 * offset)),
            dtype=xp.float64,
        )
        
    weights = cp.zeros(padded.shape, dtype=cp.float64)
    result = cp.zeros(padded.shape, dtype=cp.float64)
    
    s_cube_h_square_inv = -1 / (h * h * s * s * s)
    
    slice_short = slice(pad_to_offset, -pad_to_offset, None)
    slice_simple = slice(pad_size, -pad_size, None)
    slice_hi = slice(2 * offset, None, None)
    slice_lo = slice(None, -2 * offset, None)
    
    for t_pln, t_row, t_col in iter_directions(d):       
        pslice_offset = slice(pad_to_offset + t_pln, -pad_to_offset + t_pln, None)
        rslice_offset = slice(pad_to_offset + t_row, -pad_to_offset + t_row, None)
        cslice_offset = slice(pad_to_offset + t_col, -pad_to_offset + t_col, None)
            
        cp.subtract(
                padded[slice_short, slice_short, slice_short], 
                padded[pslice_offset, rslice_offset, cslice_offset],
                out=summed_area_table
            )
            
        squared_minus(summed_area_table, var, summed_area_table)
        cp.cumsum(summed_area_table, axis=0, out=summed_area_table)
        cp.cumsum(summed_area_table, axis=1, out=summed_area_table)
        cp.cumsum(summed_area_table, axis=2, out=summed_area_table) 
            
        local_weight = (
                + summed_area_table[slice_hi, slice_hi, slice_hi]
                - summed_area_table[slice_lo, slice_lo, slice_lo]
                + summed_area_table[slice_lo, slice_lo, slice_hi]
                + summed_area_table[slice_lo, slice_hi, slice_lo]
                + summed_area_table[slice_hi, slice_lo, slice_lo]
                - summed_area_table[slice_lo, slice_hi, slice_hi]
                - summed_area_table[slice_hi, slice_lo, slice_hi]
                - summed_area_table[slice_hi, slice_hi, slice_lo]
            ) * s_cube_h_square_inv
        cp.clip(local_weight, None, 0, out=local_weight)
            
        fast_approx_exp(local_weight)
            
        pslice_offset = slice(pad_size + t_pln, -pad_size + t_pln, None)
        rslice_offset = slice(pad_size + t_row, -pad_size + t_row, None)
        cslice_offset = slice(pad_size + t_col, -pad_size + t_col, None)
            
        weights[slice_simple, slice_simple, slice_simple] += local_weight
        result[pslice_offset, rslice_offset, cslice_offset] += cp.multiply(local_weight, padded[slice_simple, slice_simple, slice_simple])
    
        weights[pslice_offset, rslice_offset, cslice_offset] += local_weight
        result[slice_simple, slice_simple, slice_simple] += cp.multiply(local_weight, padded[pslice_offset, rslice_offset, cslice_offset])

    cp.cuda.Stream.null.synchronize()
    result[slice_simple, slice_simple, slice_simple] /= weights[slice_simple, slice_simple, slice_simple]
    
    # Transfer result to CPU and release GPU memory
    #result_cpu = cp.asnumpy(result[slice_simple, slice_simple, slice_simple])
    
    del local_weight, padded, summed_area_table, weights
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return result[slice_simple, slice_simple, slice_simple]

def nlm_3d(image, sigma, patch_size=7, patch_distance=1, h=0.03):        
    if not image.flags.c_contiguous:
        image = xp.ascontiguousarray(image)
    else:
        image = xp.asarray(image)
    dn = fast_nl_means_denoising_3d_gpu(image, s=patch_size, d=patch_distance, h=h, var=2 * sigma * sigma)
    del image
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return dn

