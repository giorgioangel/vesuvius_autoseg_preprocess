# 2024 - Giorgio Angelotti

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
import torch
from torch.utils.data import Dataset
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform

class RandomPermuteAxesTransform:
    def __call__(self, **data_dict):
        # Permute only the spatial dimensions (assumed to be the last three dimensions)
        spatial_axes = np.random.permutation(3)
        new_axes = [0, 1] + [i + 2 for i in spatial_axes]
        
        data_dict["data"] = np.transpose(data_dict["data"], new_axes)
        data_dict["seg"] = np.transpose(data_dict["seg"], new_axes)
        
        return data_dict
    
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, array_shape=(64, 64, 64), transform=None):
        self.num_samples = num_samples
        self.array_shape = array_shape
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label_volume = self.generate_synthetic_data()
        input_volume = self.create_input_from_label(label_volume)
        
        if self.transform:
            # Apply the same transform to both input and label
            data_dict = {"data": input_volume.unsqueeze(0).unsqueeze(0).numpy(), "seg": label_volume.unsqueeze(0).unsqueeze(0).numpy()}
            transformed = self.transform(**data_dict)
            input_volume = torch.tensor(transformed['data'][0, 0]).float()
            label_volume = torch.tensor(transformed['seg'][0, 0]).float()
        
        # Normalize the input volume
        #input_volume = (input_volume - input_volume.mean()) / input_volume.std()
        input_volume /= 255.
        
        # Ensure the input_volume and label_volume have 1 channel
        input_volume = input_volume.unsqueeze(0)
        label_volume = label_volume.unsqueeze(0)
        
        return input_volume, label_volume

    def generate_synthetic_data(self):
        array = np.zeros(self.array_shape, dtype=np.uint8)
        counter = 0
        counter2 = 0
        thickness = np.random.randint(5, 13)
        gap = np.random.randint(0, 2)
        amplitude = 5
        frequency = 0.1
        x_indices = np.arange(self.array_shape[1])

        for i in range(self.array_shape[0]):
            if counter <= thickness and counter2 == 0:
                value = 1
                counter += 1
            elif counter2 <= gap:
                thickness = np.random.randint(5, 13)
                value = 0
                counter2 += 1
            else:
                gap = np.random.randint(0, 2)
                value = 0
                counter = 0
                counter2 = 0

            y_perturbation = (amplitude * np.sin(frequency * x_indices)).astype(int)
            y_perturbation = np.convolve(y_perturbation, np.ones(3)/3, mode='same').astype(int)
            y_indices = (i + y_perturbation) % self.array_shape[2]
            array[x_indices[:, None], y_indices[:, None], np.arange(self.array_shape[2])] = value
        
        return torch.tensor(array, dtype=torch.float32)

    def add_rician_noise(self, volume, noise_level=20):
        noise = noise_level * torch.randn_like(volume)
        noisy_volume = torch.sqrt((volume + noise)**2 + noise**2)
        return noisy_volume

    def smooth_volume(self, volume, sigma=1):
        volume_np = volume.numpy()
        smoothed_volume_np = gaussian_filter(volume_np, sigma=sigma)
        return torch.tensor(smoothed_volume_np, dtype=torch.float32)

    def uniform_smooth_volume(self, volume, size=3):
        volume_np = volume.numpy()
        smoothed_volume_np = uniform_filter(volume_np, size=size)
        return torch.tensor(smoothed_volume_np, dtype=torch.float32)

    def create_input_from_label(self, label_volume):
        input_volume = torch.zeros_like(label_volume, dtype=torch.float32)
        input_volume[label_volume == 1] = torch.from_numpy(np.random.uniform(145, 255, size=(label_volume == 1).sum().item())).float()
        input_volume[label_volume == 0] = torch.from_numpy(np.random.uniform(90, 145, size=(label_volume == 0).sum().item())).float()

        noise_level = np.random.uniform(10,30)
        input_volume = self.add_rician_noise(input_volume, noise_level=noise_level)

        gauss_sigma = np.random.uniform(1.5,5)
        input_volume = self.smooth_volume(input_volume, sigma=gauss_sigma)

        smooth_size = np.random.randint(3,11)
        input_volume = self.uniform_smooth_volume(input_volume, size=smooth_size)
        return input_volume

# Define the batchgenerators transformations
def get_transforms(patch_size):
    transforms = Compose([
        MirrorTransform((0, 1, 2)),
        SpatialTransform(patch_size=patch_size, do_elastic_deform=True, alpha=(0, 50), sigma=(1,4),
                         do_rotation=True, angle_x=(-np.pi/20, np.pi/20), angle_y=(-np.pi/20, np.pi/20), angle_z=(-np.pi/20, np.pi/20),
                         do_scale=False, border_mode_data='nearest', border_mode_seg='nearest'),
        RandomPermuteAxesTransform(),
        ContrastAugmentationTransform(),
    ])
    return transforms
