import blosc2
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate, gaussian_filter, uniform_filter
from torch.utils.data import Dataset
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from boundary_loss_dataloader import dist_map_transform
import numpy as np
import torch
import tifffile
import os
import nrrd
from scipy.ndimage import label, binary_erosion
from skimage.morphology import ball
import zarr

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
    
class VolumetricDataset(Dataset):
    def __init__(self, input_file, label_file, block_size=(512, 512, 512)):
        self.input_file = input_file
        self.label_file = label_file
        self.block_size = block_size

        
        self.input_data = blosc2.open(input_file, mode="r")
        self.input_shape = self.input_data.shape
        
        self.label_data = blosc2.open(label_file, mode="r")

        self.blocks = self._get_blocks()

        self.disttransform = dist_map_transform([1, 1, 1])
    def _get_blocks(self):
        blocks = []
        for z in range(2000, self.input_shape[0] - 2000 - self.block_size[0] + 1, self.block_size[0]):
            for y in range(2000, self.input_shape[1] - 3000 - self.block_size[1] + 1, self.block_size[1]):
                for x in range(2000, self.input_shape[2] - 3000 - self.block_size[2] + 1, self.block_size[2]):
                    blocks.append((z, y, x))
        return blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.blocks))
        z, y, x = self.blocks[idx]
        input_block = torch.tensor(self.input_data[z:z+self.block_size[0], y:y+self.block_size[1], x:x+self.block_size[2]].astype(np.float32)/255,dtype=torch.float32).unsqueeze(0)
        label_block = torch.tensor(self.label_data[z:z+self.block_size[0], y:y+self.block_size[1], x:x+self.block_size[2]],dtype=torch.float32).unsqueeze(0)
        dist_map_tensor = self.disttransform(label_block)
        return input_block, label_block, dist_map_tensor

class VolumetricInferenceDataset(Dataset):
    def __init__(self, input_file, block_size=(512, 512, 512)):
        self.input_file = input_file
        self.block_size = block_size
        
        input_data = blosc2.open(self.input_file, mode="r")
        self.input_shape = input_data.shape
        
        self.blocks = self._get_blocks()

    def _get_blocks(self):
        blocks = []
        for z in range(2000, self.input_shape[0] - 2000 - self.block_size[0] + 1, self.block_size[0]):
            for y in range(2000, self.input_shape[1] - 3000 - self.block_size[1] + 1, self.block_size[1]):
                for x in range(2000, self.input_shape[2] - 3000 - self.block_size[2] + 1, self.block_size[2]):
                    blocks.append((z, y, x))
        return blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        z, y, x = self.blocks[idx]
        input_data = blosc2.open(self.input_file, mode="r")
        input_block = torch.tensor(
            input_data[z:z+self.block_size[0], y:y+self.block_size[1], x:x+self.block_size[2]].astype(np.float32) / 255,
            dtype=torch.float32
        ).unsqueeze(0)
        return input_block, (z, y, x)
    
class SynthData2(Dataset):
    def __init__(self, input_folder, label_folder):
        self.input_folder = input_folder
        self.label_folder = label_folder

        # Load label and input data
        self.label_data = self._load_label_data()
        self.input_data = self._load_input_data()

        # Create a list of all possible (label_id, style_id) pairs
        self.label_style_pairs = [
            (label_id, style_id)
            for label_id, styles in self.input_data.items()
            for style_id in styles.keys()
        ]

        self.disttransform = dist_map_transform([1, 1, 1])

    def _load_label_data(self):
        label_files = sorted([f for f in os.listdir(self.label_folder) if f.endswith('.zarr')])
        label_data = {int(f.split('.')[0]): zarr.open(os.path.join(self.label_folder, f), mode='r') for f in label_files}
        return label_data

    def _load_input_data(self):
        input_files = sorted([f for f in os.listdir(self.input_folder) if f.endswith('.zarr') and '_input_' in f])
        input_data = {}
        for f in input_files:
            parts = f.split('_')
            label_id = int(parts[0])
            style_id = int(parts[-1].split('.')[0])
            input_zarr = zarr.open(os.path.join(self.input_folder, f), mode='r')
            if input_zarr.shape == (256, 256, 256, 3):
                if label_id not in input_data:
                    input_data[label_id] = {}
                input_data[label_id][style_id] = input_zarr
        return input_data

    def __len__(self):
        return len(self.label_style_pairs)

    def __getitem__(self, idx):
        label_id, style_id = self.label_style_pairs[idx]

        label_block = self._get_full_block(self.label_data[label_id])//255
        input_block = torch.mean(self._get_full_block(self.input_data[label_id][style_id]),dim=-1)

        dist_map_tensor = self.disttransform(label_block)

        return input_block, label_block, dist_map_tensor

    def _get_full_block(self, data):
        block = torch.tensor(data[:], dtype=torch.float32).unsqueeze(0)
        return block

class AnnotatedCubes(Dataset):
    def __init__(self, input_folder, size=256):
        self.input_folder = input_folder
        self.data = self._load_data()
        self.disttransform = dist_map_transform([1, 1, 1])
        self.transforms = get_transforms_ann()
        self.struct = ball(1)
        self.size = size

    def _load_data(self):
        data = []
        for subdir in os.listdir(self.input_folder):
            subdir_path = os.path.join(self.input_folder, subdir)
            if os.path.isdir(subdir_path):
                # Extract z, y, x from the subdir name
                z, y, x = subdir.split('_')
                volume_file = os.path.join(subdir_path, f'{z}_{y}_{x}_volume.nrrd')
                mask_file = os.path.join(subdir_path, f'{z}_{y}_{x}_mask.nrrd')
                if os.path.exists(volume_file) and os.path.exists(mask_file):
                    data.append((volume_file, mask_file))
        print(f"Dataset length: {len(data)}")
        return data

    def __len__(self):
        return len(self.data)

    def preprocess_mask(self, mask):
        # Label connected components
        labels = np.unique(mask[mask!=0])
        final_mask = np.zeros_like(mask).astype(np.uint8)
        for n in labels:
            temp_mask = (mask == n).astype(np.uint8)
            # Erode the labeled mask
            eroded_mask = binary_erosion(temp_mask, structure=self.struct)
            
            # Assign original values to the boundaries of the chunk
            eroded_mask[0, :, :] = temp_mask[0, :, :]
            eroded_mask[-1, :, :] = temp_mask[-1, :, :]
            eroded_mask[:, 0, :] = temp_mask[:, 0, :]
            eroded_mask[:, -1, :] = temp_mask[:, -1, :]
            eroded_mask[:, :, 0] = temp_mask[:, :, 0]
            eroded_mask[:, :, -1] = temp_mask[:, :, -1]
            
            # Relabel all non-zero components as 1
            processed_mask = np.where(eroded_mask > 0, 1, 0)
            final_mask += processed_mask.astype(np.uint8)

        return final_mask
    

    def __getitem__(self, idx):
        volume_path, mask_path = self.data[idx]
        #print(volume_path, mask_path)
        volume, _ = nrrd.read(volume_path)
        mask, _ = nrrd.read(mask_path)

        if self.size < volume.shape[0]:
            corner = np.random.randint(0,volume.shape[0]-self.size, size=3)
            volume = volume[corner[0]:corner[0]+self.size, corner[1]:corner[1]+self.size, corner[2]:corner[2]+self.size]
            mask = mask[corner[0]:corner[0]+self.size, corner[1]:corner[1]+self.size, corner[2]:corner[2]+self.size]

        mask = self.preprocess_mask(mask)

        volume = np.expand_dims(volume, axis=0).astype(np.float32)/255

        mask = np.expand_dims(mask, axis=0)
        # Apply transformations
        data_dict = {'data': volume, 'seg': mask}
        transformed = self.transforms(**data_dict)
        # Convert to torch tensors
        volume = torch.tensor(transformed['data'], dtype=torch.float32)
        mask = torch.tensor(transformed['seg'], dtype=torch.float32)

        dist_map_tensor = self.disttransform(mask)

        assert volume.dim() == mask.dim(), f"{volume_path}"

        return volume, mask, dist_map_tensor


class InstanceCubes(Dataset):
    def __init__(self, input_folder, max_instance_number):
        self.input_folder = input_folder
        self.data = self._load_data()
        self.transforms = get_transforms_instances()
        self.max_instance_number = max_instance_number
        
    def _load_data(self):
        data = []
        for subdir in os.listdir(self.input_folder):
            subdir_path = os.path.join(self.input_folder, subdir)
            if os.path.isdir(subdir_path):
                # Extract z, y, x from the subdir name
                _, z, y, x = subdir.split('_')
                volume_file = os.path.join(subdir_path, f'volume_{z}_{y}_{x}.nrrd')
                mask_file = os.path.join(subdir_path, f'mask_{z}_{y}_{x}.nrrd')
                if os.path.exists(volume_file) and os.path.exists(mask_file):
                    data.append((volume_file, mask_file))
        return data

    def __len__(self):
        return len(self.data)

    def change_colors(self, mask):
        # Ensure that all values in mask are within the allowed range
        assert np.max(mask) < self.max_instance_number, "Instance mask contains values greater than the number of instances"
        
        # Identify unique values and inverse indices
        unique_values, inverse_indices = np.unique(mask, return_inverse=True)
        
        # Generate an array with values from 1 to max_instance_number-1 and shuffle it
        shuffled_values = np.arange(1, self.max_instance_number, dtype=np.uint8)
        np.random.shuffle(shuffled_values)
        
        # Create a mapping array with same shape as unique_values
        mapping_array = np.zeros_like(unique_values, dtype=np.uint8)

        # Map non-zero unique values to shuffled values
        non_zero_unique = unique_values[unique_values != 0]
        mapping_array[unique_values != 0] = shuffled_values[:len(non_zero_unique)]

        # Apply the mapping to the original array using inverse_indices and reshape to original shape
        new_mask = mapping_array[inverse_indices].reshape(mask.shape)
        
        return new_mask

    def __getitem__(self, idx):
        volume_path, mask_path = self.data[idx]
        volume, _ = nrrd.read(volume_path)
        mask, _ = nrrd.read(mask_path)
        mask = self.change_colors(mask)

        volume = np.expand_dims(volume, axis=0) # keep uint8
        mask = np.expand_dims(mask, axis=0)

        # Apply transformations
        data_dict = {'data': volume, 'seg': mask}
        transformed = self.transforms(**data_dict)
        # Convert to torch tensors
        volume = torch.tensor(transformed['data'], dtype=torch.uint8)
        mask = torch.tensor(transformed['seg'], dtype=torch.uint8)

        return mask, volume



class RandomPermuteAxesTransform:
    def __call__(self, **data_dict):
        # Permute only the spatial dimensions (assumed to be the last three dimensions)
        spatial_axes = np.random.permutation(3)
        new_axes = [0, 1] + [i + 2 for i in spatial_axes]
        
        data_dict["data"] = np.transpose(data_dict["data"], new_axes)
        data_dict["seg"] = np.transpose(data_dict["seg"], new_axes)
        
        return data_dict

class RandomPermuteAxesTransformAnnCubes:
    def __call__(self, **data_dict):
        # Permute only the spatial dimensions (assumed to be the last three dimensions)
        spatial_axes = np.random.permutation(3)
        new_axes = [0] + [i + 1 for i in spatial_axes]
        
        data_dict["data"] = np.transpose(data_dict["data"], new_axes)
        data_dict["seg"] = np.transpose(data_dict["seg"], new_axes)
        
        return data_dict
       
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, array_shape=(64, 64, 64), transform=None):
        self.num_samples = num_samples
        self.array_shape = array_shape
        self.transform = transform
        self.disttransform = dist_map_transform([1, 1, 1]) #initialize transform 3D with same resolution for each dim and 2 classes

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
        
        dist_map_tensor = self.disttransform(label_volume)

        return input_volume, label_volume, dist_map_tensor

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
        MirrorTransform((0, 1, 2)),  # Random flips
        SpatialTransform(patch_size=patch_size, do_elastic_deform=True, alpha=(0, 50), sigma=(1,4),
                         do_rotation=True, angle_x=(-np.pi/20, np.pi/20), angle_y=(-np.pi/20, np.pi/20), angle_z=(-np.pi/20, np.pi/20),
                         do_scale=False, border_mode_data='nearest', border_mode_seg='nearest'),
        RandomPermuteAxesTransform(),
        ContrastAugmentationTransform(),
    ])
    return transforms

# Define the batchgenerators transformations
def get_transforms_ann():
    transforms = Compose([
        MirrorTransform((0, 1, 2)),  # Random flips
        RandomPermuteAxesTransformAnnCubes(),
        #ContrastAugmentationTransform(),
    ])
    return transforms

# Define the batchgenerators transformation for instance segmentation
def get_transforms_instances():
    transforms = Compose([
        MirrorTransform((0, 1, 2)),  # Random flips
        RandomPermuteAxesTransformAnnCubes(),
        #ContrastAugmentationTransform(), disable this for now
    ])
    return transforms