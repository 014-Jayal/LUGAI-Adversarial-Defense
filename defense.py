# defense.py

import torch
import torch.nn.functional as F

def calculate_softmax_entropy(logits):
    """
    Calculates the entropy of the softmax distribution.
    logits: Raw model output (N, C)
    """
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-9) # Epsilon for numerical stability
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def get_reconstruction_error(autoencoder, images, criterion):
    """
    Calculates the reconstruction error for a batch of images.
    Returns a 1D tensor of errors, one for each image in the batch.
    """
    reconstructed_images = autoencoder(images)
    errors = criterion(reconstructed_images, images)
    
    # Average the error across all dimensions (C, H, W) except the batch dim
    while errors.dim() > 1:
        errors = errors.mean(dim=-1)
        
    return errors