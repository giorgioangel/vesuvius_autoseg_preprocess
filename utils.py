# https://github.com/KhartesViewer/khartes/blob/main/volume_zarr.py

import os
import zarr
import tifffile
import cupy as cp

def load_tifstack(path):
    # Get a list of .tif files
    tiffs = [filename for filename in os.listdir(path) if filename.endswith(".tif")]
    if all([filename[:-4].isnumeric() for filename in tiffs]):
        # This looks like a set of z-level images
        tiffs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        paths = [os.path.join(path, filename) for filename in tiffs]
        store = tifffile.imread(paths, aszarr=True)
    elif all([filename.startswith("cell_yxz_") for filename in tiffs]):
        # This looks like a set of cell cuboid images
        images = tifffile.TiffSequence(os.path.join(path, "*.tif"), pattern=r"cell_yxz_(\d+)_(\d+)_(\d+)")
        store = images.aszarr(axestiled={0: 1, 1: 2, 2: 0})
    stack_array = zarr.open(store, mode="r")
    return stack_array

def free_memory():
    # Free all blocks in the default memory pool
    cp.get_default_memory_pool().free_all_blocks()
    # Free all blocks in the pinned memory pool (if used)
    cp.get_default_pinned_memory_pool().free_all_blocks()

# Function to yield chunk coordinates
def chunk_generator(full_shape, chunk_size):
    for z in range(0, full_shape[0], chunk_size[0]):
        for y in range(0, full_shape[1], chunk_size[1]):
            for x in range(0, full_shape[2], chunk_size[2]):
                yield z, y, x