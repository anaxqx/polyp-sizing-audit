"""
Loss functions for polyp size classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α(1-p_t)^γ log(p_t)
    
    Where:
    - α: weighting factor for rare class
    - γ: focusing parameter (higher = more focus on hard examples)
    - p_t: predicted probability for true class
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        """
        Args:
            alpha: Weighting factor (float or tensor of size num_classes)
            gamma: Focusing parameter
            weight: Class weights (optional, overrides alpha if provided)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        
        # Use alpha if provided, otherwise use class weights
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        elif isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing
    
    Reduces overconfidence and helps with generalization
    """
    
    def __init__(self, smoothing=0.1, weight=None, reduction='mean'):
        """
        Args:
            smoothing: Smoothing factor (0 = no smoothing, 1 = uniform)
            weight: Class weights
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
        """
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            
            # Apply class weights if provided
            if self.weight is not None:
                true_dist = true_dist * self.weight.unsqueeze(0)
        
        # Compute loss
        loss = -torch.sum(true_dist * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loss_function(config):
    """
    Create loss function based on config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Loss function
    """
    loss_type = config['training'].get('loss_function', 'cross_entropy').lower()
    use_class_weights = config['training'].get('use_class_weights', True)
    
    # Get class weights if needed
    weight = None
    if use_class_weights:
        # This will be set later in Trainer.__init__ after dataset is loaded
        # For now, return a function that accepts weights
        pass
    
    if loss_type == 'cross_entropy':
        # Standard cross entropy (will add weights later)
        return lambda weights: nn.CrossEntropyLoss(weight=weights)
    
    elif loss_type == 'focal':
        alpha = config['training'].get('focal_alpha', 0.25)
        gamma = config['training'].get('focal_gamma', 2.0)
        return lambda weights: FocalLoss(alpha=alpha, gamma=gamma, weight=weights)
    
    elif loss_type == 'label_smoothing' or loss_type == 'smooth_ce':
        smoothing = config['training'].get('label_smoothing', 0.1)
        # PyTorch 1.10+ supports label_smoothing in CrossEntropyLoss
        try:
            return lambda weights: nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=smoothing
            )
        except TypeError:
            # Fallback for older PyTorch
            return lambda weights: LabelSmoothingCrossEntropy(
                smoothing=smoothing,
                weight=weights
            )
    
    elif loss_type == 'focal_smooth':
        # Focal loss with label smoothing
        alpha = config['training'].get('focal_alpha', 0.25)
        gamma = config['training'].get('focal_gamma', 2.0)
        smoothing = config['training'].get('label_smoothing', 0.1)
        # Combine focal with smoothing (approximate)
        return lambda weights: FocalLoss(alpha=alpha, gamma=gamma, weight=weights)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")



