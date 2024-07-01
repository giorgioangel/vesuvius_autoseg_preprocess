# Semantic Segmentation
import yaml
import argparse
import torch
from torch.cuda.amp import GradScaler
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import AdamW
from losses import SymmetricUnifiedFocalLoss, BoundaryLoss
from dataset import VolumetricDataset, SyntheticDataset, SynthData2, AnnotatedCubes, get_transforms
from model import UNet
from ema import EMA
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics import Dice
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Define the dataloader function
def get_dataloader_mixed(input_file, label_file, block_size, batch_size, workers, synthetic_ratio, prefetch, pers_work):
    # Create the datasets
    volumetric_dataset = VolumetricDataset(input_file=input_file, label_file=label_file, block_size=block_size)
    synthetic_dataset = SyntheticDataset(num_samples=1000, array_shape=tuple(block_size), transform=get_transforms(tuple(block_size)))
    
    # Concatenate the datasets
    combined_dataset = ConcatDataset([volumetric_dataset, synthetic_dataset])
    
    # Define the sample weights
    num_volumetric = len(volumetric_dataset)
    num_synthetic = len(synthetic_dataset)
    weights = [(1-synthetic_ratio) / num_volumetric] * num_volumetric + [synthetic_ratio / num_synthetic] * num_synthetic
    
    # Create the sampler
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Create the dataloader
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True, prefetch_factor=prefetch, persistent_workers=pers_work)
    
    return dataloader

# Define the dataloader function
def get_dataloader_synth(input_file, label_file, block_size, batch_size, workers, synthetic_ratio, prefetch, pers_work):
    # Create the datasets
    synthetic_dataset = SyntheticDataset(num_samples=1000, array_shape=tuple(block_size), transform=get_transforms(tuple(block_size)))
    
    # Create the dataloader
    dataloader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=prefetch, persistent_workers=pers_work)
    
    return dataloader

# Define the dataloader function
def get_dataloader_synth2(input_file, label_file, block_size, batch_size, workers, synthetic_ratio, prefetch, pers_work):
    # Create the datasets
    synthetic_dataset = SynthData2(input_file, label_file)
    # Create the dataloader
    dataloader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=prefetch, persistent_workers=pers_work)
    
    return dataloader

# Define the dataloader function
def get_dataloader_volume(input_file, label_file, block_size, batch_size, workers, prefetch, pers_work):
    # Create the datasets
    volumetric_dataset = VolumetricDataset(input_file=input_file, label_file=label_file, block_size=block_size)
    
    # Create the dataloader
    dataloader = DataLoader(volumetric_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=prefetch, persistent_workers=pers_work)
    
    return dataloader

# Define the dataloader function
def get_dataloader_cubes(input_file,batch_size, workers, prefetch, pers_work):
    # Create the datasets
    volumetric_dataset = AnnotatedCubes(input_file)
    
    # Create the dataloader
    dataloader = DataLoader(volumetric_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=prefetch, persistent_workers=pers_work)
    
    return dataloader

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def validate(model, dataloader, device, metrics):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=True):
                outputs = model(inputs)
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            all_labels.append(labels)
            all_preds.append(preds)
            preds = torch.tensor(preds).round().int()  # Ensure preds are binarized and integer
            labels = torch.tensor(labels).int()  # Ensure labels are integer
            for metric in metrics:
                metric.update(preds, labels)
    results = {metric.__class__.__name__: metric.compute().item() for metric in metrics}
    for metric in metrics:
        metric.reset()
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return results, all_labels, all_preds

def find_best_threshold(labels, preds):
    fpr, tpr, thresholds_roc = roc_curve(labels.flatten(), preds.flatten())
    precision, recall, thresholds_pr = precision_recall_curve(labels.flatten(), preds.flatten())
    f1_scores = 2 * recall * precision / (recall + precision)
    
    best_threshold_roc = thresholds_roc[np.argmax(tpr - fpr)]
    best_threshold_pr = thresholds_pr[np.argmax(f1_scores)]
    
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    
    return best_threshold_roc, best_threshold_pr, roc_auc, pr_auc

def train(config):
    set_random_seeds(config['training']['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = UNet(
            spatial_dims=3,
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            channels=config['model']['channels'],
            strides=config['model']['strides'],
            kernel_size=config['model']['kernel_size']
        ).to(device)
        try:
            model.load_state_dict(torch.load(config['training']['current_checkpoint']))
            print("Loaded weights.")
        except:
            model.apply(initialize_weights)
            print("Weights not found, initializing new weights.")
        model.train()
        scaler = GradScaler(enabled=config['training']['amp'])

        loss_1 = SymmetricUnifiedFocalLoss().to(device)
        #loss_2 = BoundaryFocalLoss(idc=[1], gamma=3) # gamma 3 because otherwise becomes positive if negative!
        loss_2 = BoundaryLoss(idc=[1])
        optimizer = AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['training']['scheduler_epochs'], eta_min=1e-6)
        ema = EMA(model, beta=config['training']['ema_beta'])

        dataloader = get_dataloader_volume(config['data']['input_file'],config['data']['label_file'],config['data']['block_size'],batch_size=config['training']['batch_size'],workers=config['training']['workers'],
                                   prefetch=config['training']['prefetch_factor'], pers_work=config['training']['persistent_workers'])
        val_dataloader = get_dataloader_volume(config['data']['input_file'],config['data']['label_file'], config['data']['block_size'], batch_size=config['training']['batch_size'], workers=config['training']['workers'],
                                   prefetch=config['training']['prefetch_factor'], pers_work=config['training']['persistent_workers'])

        # Initialize metrics
        metrics = [
            BinaryAccuracy().to(device),
            BinaryPrecision().to(device),
            BinaryRecall().to(device),
            BinaryF1Score().to(device),
            Dice().to(device)
        ]

        for epoch in range(config['training']['num_epochs']):
            running_loss = 0.0
            running_loss_1 = 0.0
            running_loss_2 = 0.0
            for i, (inputs, labels, dist_map_label) in enumerate(dataloader):
                if (i+1) * config['training']['batch_size'] > config['training']['epoch_size']:
                    break
                inputs, labels, dist_map_label = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True), dist_map_label.to(device, non_blocking=True)
                
                with torch.autocast(device_type="cuda", enabled=config['training']['amp']):
                    outputs = model(inputs)
                    loss1 = loss_1(outputs, labels)
                    loss2 = loss_2(outputs, dist_map_label)
                    loss = loss1 + config['training']['boundary_alpha']*loss2
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if torch.isnan(loss):
                    print(f"NaN loss encountered at batch {i}")
                    break
                running_loss_1 += loss1.item()
                running_loss_2 += loss2.item()
                running_loss += loss.item()
            
            ema.update()
            avg_unif = running_loss_1 / (config['training']['epoch_size'])
            avg_boundary = running_loss_2 / (config['training']['epoch_size'])
            avg_loss = running_loss / (config['training']['epoch_size'])
            print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, Unified Loss: {avg_unif}, Boundary Loss: {avg_boundary}, Average Loss: {avg_loss}")

            scheduler.step()
            
            # Validation step
            if (epoch + 1) % config['validation']['epochs'] == 0:
                val_results, all_labels, all_preds = validate(model, val_dataloader, device, metrics)
                print(f"Validation Results - Epoch {epoch+1}: {val_results}")
                model.train()

            
            if (epoch + 1) % config['training']['ema_step'] == 0:
                best_threshold_roc, best_threshold_pr, roc_auc, pr_auc = find_best_threshold(all_labels, all_preds)
                print(f"Best Threshold ROC: {best_threshold_roc}, Best Threshold PR: {best_threshold_pr}, ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
                ema.apply_shadow()
                checkpoint_path = f'./checkpoints/cubes_checkpoint_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                ema.restore()
        
        ema.apply_shadow()
        final_model_path = './checkpoints/final_model_cubes.pth'
        torch.save(model.state_dict(), final_model_path)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Training complete")

def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    train(config)

if __name__ == '__main__':
    main()
