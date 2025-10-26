# evaluate_detection.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from model_zoo import BaseCNN, Autoencoder
import config
import attacks
from defense import calculate_softmax_entropy, get_reconstruction_error

def load_data_and_models(device):
    # Load data
    test_images, test_labels = torch.load(config.TEST_FILE)
    test_ds = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Load classifier
    classifier = BaseCNN().to(device)
    classifier.load_state_dict(torch.load(config.BASELINE_MODEL_PATH, map_location=device))
    classifier.eval()
    
    # Load Denoising Autoencoder
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(config.DAE_MODEL_PATH, map_location=device))
    autoencoder.eval()
    
    return test_loader, classifier, autoencoder

def get_detection_features(model, autoencoder, data_loader, device, attack=None):
    model.eval(); autoencoder.eval()
    all_recon_errors, all_entropies = [], []
    recon_criterion = nn.MSELoss(reduction='none')
    
    pbar_desc = "Extracting features (Adversarial)" if attack else "Extracting features (Clean)"
    
    for images, labels in tqdm(data_loader, desc=pbar_desc):
        images, labels = images.to(device), labels.to(device)
        
        if attack:
            with torch.enable_grad():
                images.requires_grad = True
                input_images = attack(images, labels)
        else:
            input_images = images
            
        with torch.no_grad():
            input_images = input_images.detach()
            
            recon_errors = get_reconstruction_error(
                autoencoder, input_images, recon_criterion
            )
            all_recon_errors.append(recon_errors.cpu().numpy())
            
            logits = model(input_images)
            entropies = calculate_softmax_entropy(logits)
            all_entropies.append(entropies.cpu().numpy())
            
    return np.concatenate(all_recon_errors), np.concatenate(all_entropies)

def plot_feature_distributions(clean_errors, adv_errors, clean_entropy, adv_entropy):
    print(f"Plotting feature distributions and saving to {config.DETECTION_PLOT_PATH}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(clean_errors, bins=100, alpha=0.7, label="Clean", color="blue", density=True)
    ax1.hist(adv_errors, bins=100, alpha=0.7, label="Adversarial (FGSM)", color="red", density=True)
    ax1.set_xlabel("Reconstruction Error (MSE)")
    ax1.set_ylabel("Density")
    ax1.set_title("Reconstruction Deviation")
    ax1.legend()
    ax1.set_xlim(0, max(np.percentile(clean_errors, 99.5), np.percentile(adv_errors, 99.5)))
    
    ax2.hist(clean_entropy, bins=100, alpha=0.7, label="Clean", color="blue", density=True)
    ax2.hist(adv_entropy, bins=100, alpha=0.7, label="Adversarial (FGSM)", color="red", density=True)
    ax2.set_xlabel("Softmax Entropy")
    ax2.set_ylabel("Density")
    ax2.set_title("Latent Uncertainty (Entropy)")
    ax2.legend()
    ax2.set_xlim(0, max(np.percentile(clean_entropy, 99.5), np.percentile(adv_entropy, 99.5)))
    
    fig.suptitle("Detection Feature Distributions", fontsize=16)
    plt.tight_layout()
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(config.DETECTION_PLOT_PATH)
    plt.show()
    print("Plot saved.")

def main():
    print(f"Using device: {config.DEVICE}")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    test_loader, model, autoencoder = load_data_and_models(config.DEVICE)
    
    fgsm_attack = attacks.get_fgsm_attack(model, eps=0.3)
    
    clean_errors, clean_entropy = get_detection_features(
        model, autoencoder, test_loader, config.DEVICE, attack=None
    )
    
    adv_errors, adv_entropy = get_detection_features(
        model, autoencoder, test_loader, config.DEVICE, attack=fgsm_attack
    )
    
    print("\nFeature extraction complete.")
    print(f"Clean Error (mean):   {clean_errors.mean():.6f}")
    print(f"Adversarial Error (mean): {adv_errors.mean():.6f}")
    print(f"Clean Entropy (mean):   {clean_entropy.mean():.6f}")
    print(f"Adversarial Entropy (mean): {adv_entropy.mean():.6f}")
    
    plot_feature_distributions(clean_errors, adv_errors, clean_entropy, adv_entropy)
    print("\nDetection feature evaluation complete.")

if __name__ == "__main__":
    main()