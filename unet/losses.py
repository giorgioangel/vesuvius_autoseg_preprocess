# Giorgio Angelotti - 2023 - Different Losses for imbalanced datasets

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from torch import Tensor, einsum

__all__ = ['BinaryFocalLoss', 'TraditionalBinaryFocalLoss', 'DiceLoss', 'BatchDiceLoss',
           'FocalTverskyLoss', 'SymmetricUnifiedFocalLoss', 'CrossFocalLoss', 'SoftArgmaxMSELoss', 'SurfaceLoss']

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha:float=0.25, gamma:float=2.0, reduction:str='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the binary cross entropy with logits loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute the focal loss adjustment
        p_t = torch.exp(-BCE_loss)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        #F_loss = self.alpha * (1 - p_t)**self.gamma * BCE_loss
        F_loss = at * (1 - p_t)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
              
class MaskedBinaryFocalLoss(nn.Module):
    def __init__(self, alpha:float=0.25, gamma:float=2.0, reduction:str='mean'):
        super(MaskedBinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask):
        # Compute the binary cross entropy with logits loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute the focal loss adjustment
        p_t = torch.exp(-BCE_loss)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        #F_loss = self.alpha * (1 - p_t)**self.gamma * BCE_loss
        F_loss = at * (1 - p_t)**self.gamma * BCE_loss

        F_loss = F_loss[mask]

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
         
class TraditionalBinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(TraditionalBinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to the inputs
        inputs = torch.sigmoid(inputs)
        
        # Compute the binary cross entropy loss
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Apply the focal loss adjustment
        targets = targets.type(inputs.type())
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred_logits, y_true):
        # Apply sigmoid to the logits
        y_pred = torch.sigmoid(y_pred_logits)
        
        # Flatten the prediction and true tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Compute the intersection and cardinality
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
        
        # Compute the dice score
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score

class BatchDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Flatten the batch and channel dimensions
        y_pred_probs = F.softmax(predictions, dim=1)
        y_pred_probs = y_pred_probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (y_pred_probs * targets_flat).sum()
        union = y_pred_probs.sum() + targets_flat.sum()

        # Compute the Dice coefficient for the batch
        dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)

        # Compute Dice loss
        dice_loss = 1 - dice_coefficient

        return dice_loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, gamma: float = 0.75, smooth: float = 1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Apply sigmoid to the logits to get the predicted probabilities
        y_pred = torch.sigmoid(y_pred)

        # Flatten the tensors to make sure that we can perform element-wise multiplications
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True positives, false negatives, and false positives
        true_pos = torch.sum(y_true * y_pred)
        false_neg = torch.sum(y_true * (1 - y_pred))
        false_pos = torch.sum((1 - y_true) * y_pred)

        # Tversky index
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)

        # Focal Tversky loss
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)
        return focal_tversky_loss.mean()

    
class MaskedFocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, gamma: float = 0.75, smooth: float = 1e-6):
        super(MaskedFocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true, mask):
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        # Apply sigmoid to the logits to get the predicted probabilities
        y_pred = torch.sigmoid(y_pred)

        # Flatten the tensors to make sure that we can perform element-wise multiplications
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True positives, false negatives, and false positives
        true_pos = torch.sum(y_true * y_pred)
        false_neg = torch.sum(y_true * (1 - y_pred))
        false_pos = torch.sum((1 - y_true) * y_pred)

        # Tversky index
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)

        # Focal Tversky loss
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)
        return focal_tversky_loss.mean()
       
class SymmetricUnifiedFocalLoss(nn.Module):
  def __init__(self, delta:float=0.6, gamma:float=0.5, smooth:float=1e-6):
    super(SymmetricUnifiedFocalLoss, self).__init__()
    self.focal_cross = BinaryFocalLoss(alpha=delta, gamma=1-gamma)
    self.focal_tve = FocalTverskyLoss(alpha=delta, gamma=gamma, smooth=smooth)
  def forward(self, y_pred, y_true):
    return self.focal_cross(y_pred, y_true) + self.focal_tve(y_pred, y_true)

class MaskedSymmetricUnifiedFocalLoss(nn.Module):
  def __init__(self, delta:float=0.6, gamma:float=0.5, smooth:float=1e-6):
    super(MaskedSymmetricUnifiedFocalLoss, self).__init__()
    self.focal_cross = MaskedBinaryFocalLoss(alpha=delta, gamma=1-gamma)
    self.focal_tve = MaskedFocalTverskyLoss(alpha=delta, gamma=gamma, smooth=smooth)
  def forward(self, y_pred, y_true, mask):
    return self.focal_cross(y_pred, y_true, mask) + self.focal_tve(y_pred, y_true, mask)
   
class CrossFocalLoss(nn.Module):
    def __init__(self, gamma:float=2.0, reduction:str='mean'):
        super(CrossFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross entropy loss
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Apply the focal loss adjustment
        #targets = targets.type(inputs.type())
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class SoftArgmaxMSELoss(nn.Module):
    def __init__(self, values:torch.Tensor, scale:float=1.0, max:float=1.0, min:float=0):
        super(SoftArgmaxMSELoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.values = values.clone().detach().to(torch.float32).to(device)
        self.scale = scale
        self.max = max
        self.min = min

    def forward(self, outputs, targets):
        # Apply a scaled softmax to approximate argmax
        soft_argmax = F.softmax(outputs * self.scale, dim=-1)
        # Determine the correct einsum equation based on the dimensions of outputs
        if outputs.dim() == 3:
            # If outputs is 3D: batch x sequence x dimension
            equation = 'bsd,d->bs'
        elif outputs.dim() == 2:
            # If outputs is 2D: batch x dimension
            equation = 'bd,d->b'
        else:
            raise ValueError("The outputs tensor should be either 2D or 3D.")
        soft_exp_value = torch.einsum(equation, soft_argmax, self.values)/self.values[-1]
        loss = F.mse_loss(soft_exp_value, targets.float()/self.values[-1])
        return loss
    


class BoundaryLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, logits: Tensor, dist_maps: Tensor) -> Tensor:        
        probs = F.sigmoid(logits)
        multipled = einsum("bkxyz,bkxyz->bkxyz", probs, dist_maps)

        loss = multipled.mean()

        return loss
    
class MaskedBoundaryLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, logits: Tensor, dist_maps: Tensor, mask: Tensor) -> Tensor:        
        probs = F.sigmoid(logits)
        multipled = einsum("bkxyz,bkxyz->bkxyz", probs, dist_maps)
        multipled = multipled[mask]
        loss = multipled.mean()

        return loss

class BoundaryFocalLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.gamma = kwargs["gamma"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, logits: Tensor, dist_maps: Tensor) -> Tensor:        
        probs = F.sigmoid(logits)
        multipled = einsum("bkxyz,bkxyz->bkxyz", probs, dist_maps)

        loss = torch.pow(multipled, self.gamma).mean()

        return loss
        
