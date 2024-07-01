# Vesuvius Challenge: Preprocess for Automated CT Scan Papyrus Scroll Segmentation

This repository contains the preprocessing steps for creating a training dataset for automated segmentation of CT scans of papyrus scrolls. This work is part of the [Vesuvius Challenge](https://scrollprize.org/).

The pipeline is divided into separate steps, with each step handled in a dedicated notebook.

Every step handles the volume in separated chunks.

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

### Semantic Segmentation
- **training script** `.\unet\train.py --config config.yaml`
- **Description**: Training script to train a 3D UNet for Semantic Segmentation on several datasets. Intuitive code. Used loss function is a mixture of Unified Symmetric Focal Loss [(Yeung et al., 2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8785124/) and Boundary Loss [(Kervadec et al., 2021)](https://arxiv.org/abs/1812.07032), which are particularly good on imbalanced 3D segmentation datasets. Exponential Moving Average of weights (EMA) is applied to stabilize the training, also because in the literature there are promising results where it has been used to improve training performance in contexts with noisy labels [(Morales-Brotons et al., 2024)](https://openreview.net/forum?id=2M9CUnYnBA), which is exactly our scenario because we don't have (yet) ground truths.
- **checkpoints**: (https://dl.ash2txt.org/other/semantic-segmentation-checkpoints/)

## Upcoming Additions

- **Requirements File:** A file to download the correct dependencies for running the algorithms.
- **Refinement Script:** A script to refine the output of the Morphological Chan Vese segmentation with the voxels identified by surface detection.
- **Final Refinement:** A final refinement step by thresholding on median air intensity values and median papyrus intensity values to cancel out outliers and recover improperly canceled voxels.

## FAQ
Why not using just the results of Morphological Chan Vese or Surface Detection? Well, the results are not accurate enough to separate papyrus sheets in compressed regions! And the refined results won't be as well! Our vision is to produce manually refined labels as well to train an efficient segmentator neural network.

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

Special thanks to [Brett Olsen](https://github.com/caethan) for the original implementation of the GPU version of Non-Local-Means, which is based on the skimage version. Tim Skinner for the Resisudal Unet edit of the MONAI 3D Unet.