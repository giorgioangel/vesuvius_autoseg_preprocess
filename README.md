# Vesuvius Challenge: Preprocess for Automated CT Scan Papyrus Scroll Segmentation

This repository contains the preprocessing steps for creating a training dataset for automated segmentation of CT scans of papyrus scrolls. This work is part of the [Vesuvius Challenge](https://scrollprize.org/).

The pipeline is divided into separate steps, with each step handled in a dedicated notebook.

Every step handles the volume in separated chunks.

UPDATE: We are currently focusing on Semantic (and Instance) segmentation with Deep Neural Networks: [See 3D UX-Net section](#semantic-segmentation-with-deep-neural-networks)


## Current Contents

### 1. Denoising and Contrast Enhancement
- **Notebook:** `1_denoise_clahe.ipynb`
- **Description:** This notebook performs denoising of volumes using Non-Local-Means and enhances contrast with CLAHE in a parallel multi-GPU setup.

### 2. Volume to Stack Conversion
- **Notebook:** `2_b2nd_to_tiff.ipynb`
- **Description:** Converts the denoised and contrast-enhanced volume, saved as a B2NDarray, into a stack of TIFF images.

### 3. Morphological Chan Vese Segmentation
- **Notebook:** `3_morphological_chan_vese.ipynb`
- **Description:** Applies the Morphological Chan Vese algorithm to produce thick volumetric labels.

### 4. Surface Detection in 3D
- **Notebook:** `4_surface_detection.ipynb`
- **Description:** Performs surface detection using thresholding on the gradient magnitude and the absolute value of the determinant of the Hessian for each voxel. The gradient and Hessian are estimated using a 3D implementation of Noise Robust Gradient operators by Pavel Holoborodko, combining isotropic noise suppression and precise gradient estimation. Non-Maximum Suppression is applied to the result.


### Synthetic Dataset
- **script** `.\synthetic_data\dataset.py`
- **Description:**  Contains a script to generate a completely synthetic dataset.
    Example to create the dataset:
    ```python
    block_size = [128, 128, 128]
    synthetic_dataset = SyntheticDataset(num_samples=1000, array_shape=tuple(block_size), transform=get_transforms(tuple(block_size)))
    ```

### Semantic Segmentation with Deep Neural Networks
#### 3D UX-Net
- **(UPDATE) 3D UX-Net training script** `.\unet\train_uxnet.py --config config-uxnet.yaml`
- **Description**: Training script to train a 3D UX-Net [(Lee et al., 2023)](https://arxiv.org/abs/2209.15076) (model code taken from https://github.com/MASILab/3DUX-Net/tree/main) for Semantic Segmentation. Used loss function is a mixture of Binary Focal Loss and Boundary Loss [(Kervadec et al., 2021)](https://arxiv.org/abs/1812.07032). I noticed that to obtain an improvement in instance separation, after a first training on the full cubes, masking everything but the borders with medium intensity voxels (clearly not air) can help getting higher metrics.
- **checkpoints**: (https://dl.ash2txt.org/other/semantic-segmentation-checkpoints/160724_uxnet_mask_epoch_28.pth)

#### 3D UNet
- **training script** `.\unet\train.py --config config.yaml`
- **Description**: Training script to train a 3D UNet for Semantic Segmentation on several datasets. Intuitive code. Used loss function is a mixture of Unified Symmetric Focal Loss [(Yeung et al., 2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8785124/) and Boundary Loss [(Kervadec et al., 2021)](https://arxiv.org/abs/1812.07032), which are particularly good on imbalanced 3D segmentation datasets. Exponential Moving Average of weights (EMA) is applied to stabilize the training, also because in the literature there are promising results where it has been used to improve training performance in contexts with noisy labels [(Morales-Brotons et al., 2024)](https://openreview.net/forum?id=2M9CUnYnBA), which is exactly our scenario because we don't have (yet) ground truths.
- **checkpoints**: (https://dl.ash2txt.org/other/semantic-segmentation-checkpoints/)

#### Where we are going?
We aim to improve semantic segmentation to allow better annotation in compressed regions of the papyrus. This will eventually lead to:
1) semantic segmentators that can already separate sheet instances
2) a dataset that will allow us to train an instance segmentator. An interesting architecture is Flood-Filling Networks [(Januszewski et al., 2016)](https://arxiv.org/abs/1611.00421)

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/vesuvius-ct-scan-segmentation.git
    cd vesuvius-ct-scan-segmentation
    ```

2. Install the dependencies (requirements file coming soon).

3. Open and run the notebooks in the provided sequence to preprocess your CT scan data. Remember to change the paths in the cells.

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to [Brett Olsen](https://github.com/caethan) for the original implementation of the GPU version of Non-Local-Means, which is based on the skimage version. Tim Skinner for the Resisudal Unet edit of the MONAI 3D Unet. 'unet/lib' and 'unet/networks' folder are taken from the official 3D UX-Net repository (https://github.com/MASILab/3DUX-Net/tree/main)
