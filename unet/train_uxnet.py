# Semantic Segmentation with 3D UX-Net
import yaml
import argparse
import torch
from torch.cuda.amp import GradScaler
import numpy as np
import random
from torch.utils.data import DataLoader, 
from torch.optim import AdamW
from losses import MaskedBoundaryLoss, MaskedBinaryFocalLoss
from dataset import AnnotatedCubes
from networks.UXNet_3D.network_backbone import UXNET
from ema import EMA
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryFBetaScore
from torchmetrics import Dice
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, total_epochs - warmup_epochs, eta_min=1e-6)
    return warmup_scheduler, cosine_scheduler


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Define the dataloader function
def get_dataloader_cubes(input_file,batch_size, workers, prefetch, pers_work):
    # Create the datasets
    volumetric_dataset = AnnotatedCubes(input_file, size=128)
    
    # Create the dataloader
    dataloader = DataLoader(volumetric_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=prefetch, persistent_workers=pers_work)
    
    return dataloader

# Define the dataloader function
def get_validation_cubes(input_file,batch_size, workers, prefetch, pers_work):
    # Create the datasets
    volumetric_dataset = AnnotatedCubes(input_file, size=128)
    
    # Create the dataloader
    dataloader = DataLoader(volumetric_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, prefetch_factor=prefetch, persistent_workers=pers_work)
    
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
        for inputs, labels, dist_map_label in dataloader:
            inputs, labels, dist_map_label = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True), dist_map_label.to(device, non_blocking=True)
            mask = (dist_map_label < 1.5) & (dist_map_label > -1.5) & (inputs > 0.45)
            #mask = (dist_map_label <= 1) & (dist_map_label >= -1) & (inputs > 0.5)
            with torch.autocast(device_type="cuda", enabled=True):
                outputs = model(inputs)
            preds = torch.sigmoid(outputs[mask]).detach().cpu().numpy()
            labels = labels[mask].detach().cpu().numpy()
            del outputs #mask
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
    f_half_scores = (1 + 0.5**2) * recall * precision / (0.5**2 * precision + recall)
    best_threshold_roc = thresholds_roc[np.argmax(tpr - fpr)]
    best_threshold_pr = thresholds_pr[np.argmax(f1_scores)]
    best_threshold_halfpr = thresholds_pr[np.argmax(f_half_scores)]
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    
    return best_threshold_roc, best_threshold_pr, best_threshold_halfpr, roc_auc, pr_auc

def train(config):
    set_random_seeds(config['training']['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = UXNET(
            in_chans=1,
            out_chans=1,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        ).to(device)

        try:
            model.load_state_dict(torch.load(config['training']['current_checkpoint']))
            print("Loaded weights.")
        except:
            model.apply(initialize_weights)
            print("Weights not found, initializing new weights.")
        model.train()
        scaler = GradScaler(enabled=config['training']['amp'])

        #loss_1 = MaskedSymmetricUnifiedFocalLoss().to(device)
        loss_1 = MaskedBinaryFocalLoss().to(device)
        loss_2 = MaskedBoundaryLoss(idc=[1]) # gamma 3 because otherwise becomes positive if negative!
        optimizer = AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
        warmup_scheduler, cosine_scheduler = get_scheduler(optimizer, config['training']['warmup_epochs'], config['training']['num_epochs'])
        if config['training']['ema']:
            ema = EMA(model, beta=config['training']['ema_beta'])
        else:
            print("EMA not active")
        dataloader = get_dataloader_cubes(config['data']['cubes_folder'],batch_size=config['training']['batch_size'],workers=config['training']['workers'],
                                   prefetch=config['training']['prefetch_factor'], pers_work=config['training']['persistent_workers'])
        val_dataloader = get_validation_cubes(config['data']['val_cubes_folder'], batch_size=config['training']['batch_size'], workers=config['training']['workers'],
                                   prefetch=config['training']['prefetch_factor'], pers_work=config['training']['persistent_workers'])

        # Initialize metrics
        metrics = [
            BinaryAccuracy().to(device),
            BinaryPrecision().to(device),
            BinaryRecall().to(device),
            BinaryF1Score().to(device),
            BinaryFBetaScore(beta=0.5).to(device),
            Dice().to(device)
        ]

        for epoch in range(config['training']['num_epochs']):
            if epoch == 0:
                val_results, all_labels, all_preds = validate(model, val_dataloader, device, metrics)
                print(f"Validation Results - Epoch {epoch}: {val_results}")
                best_threshold_roc, best_threshold_pr, best_threshold_halfpr, roc_auc, pr_auc = find_best_threshold(all_labels, all_preds)
                print(f"Best Threshold ROC: {best_threshold_roc}, Best Threshold F-1: {best_threshold_pr}, Best Threshold F-0.5 PR: {best_threshold_halfpr}, ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
                model.train()

            running_loss = 0.0
            running_loss_1 = 0.0
            running_loss_2 = 0.0
            for i, (inputs, labels, dist_map_label) in enumerate(dataloader):
                if (i+1) * config['training']['batch_size'] > config['training']['epoch_size']:
                    break

                # SOFT MASK
                labels[labels == 1] = 0.95
                labels[labels == 0] = 0.05

                inputs, labels, dist_map_label = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True), dist_map_label.to(device, non_blocking=True)
                #inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                #for the third condition I remove brightness and contrast augmentation from dataset otherwise pointless
                #mask = (dist_map_label < 2) & (dist_map_label > -2) & (inputs > 0.39) # first two conditions (not so far from borders, third condition focus on "papyrus like points"
                # strict ->
                mask = (dist_map_label < 1.5) & (dist_map_label > -1.5) & (inputs > 0.45)
                #mask = (dist_map_label <= 1) & (dist_map_label >= -1) & (inputs > 0.5)
                with torch.autocast(device_type="cuda", enabled=config['training']['amp']):
                    outputs = model(inputs)
                        # Apply the mask to outputs, labels, and dist_map_label
                    loss1 = loss_1(outputs, labels, mask)
                    loss2 = loss_2(outputs, dist_map_label, mask)
                    loss = loss1 + config['training']['boundary_alpha']*loss2
                    del outputs, mask
                    #del outputs
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
                
            if epoch < config['training']['warmup_epochs']:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            if config['training']['ema']:
                ema.update()

            avg_unif = running_loss_1 / (config['training']['epoch_size'])
            avg_boundary = running_loss_2 / (config['training']['epoch_size'])
            avg_loss = running_loss / (config['training']['epoch_size'])
            print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, Unified Loss: {avg_unif}, Boundary Loss: {avg_boundary}, Average Loss: {avg_loss}")


            if (epoch + 1) % config['training']['ema_step'] == 0 and config['training']['ema']:
                ema.apply_shadow()
                val_results, all_labels, all_preds = validate(model, val_dataloader, device, metrics)
                print(f"Validation Results - Epoch {epoch+1}: {val_results}")
                best_threshold_roc, best_threshold_pr, best_threshold_halfpr, roc_auc, pr_auc = find_best_threshold(all_labels, all_preds)
                print(f"Best Threshold ROC: {best_threshold_roc}, Best Threshold F-1: {best_threshold_pr}, Best Threshold F-0.5 PR: {best_threshold_halfpr}, ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
                checkpoint_path = f'./checkpoints/160724_uxnet_mask_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                ema.restore()
                model.train()

            # Validation step
            elif (epoch + 1) % config['validation']['epochs'] == 0:
                val_results, all_labels, all_preds = validate(model, val_dataloader, device, metrics)
                print(f"Validation Results - Epoch {epoch+1}: {val_results}")
                checkpoint_path = f'./checkpoints/160724_uxnet_mask_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                model.train()
        if config['training']['ema']:
            ema.apply_shadow()
        final_model_path = './checkpoints/160724_uxnet_mask_final.pth'
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
