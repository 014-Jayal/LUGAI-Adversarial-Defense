# visualize_purification.py

import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from model_zoo import BaseCNN, Autoencoder
import config
import attacks
from visualize import unnormalize, plot_purification_comparison

def load_all_models(device):
    print("Loading models...")
    classifier = BaseCNN().to(device)
    classifier.load_state_dict(torch.load(config.BASELINE_MODEL_PATH, map_location=device))
    classifier.eval()
    
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(config.DAE_MODEL_PATH, map_location=device))
    autoencoder.eval()
    
    return classifier, autoencoder

def get_test_batch(n_images):
    test_images, test_labels = torch.load(config.TEST_FILE)
    test_ds = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_ds, batch_size=n_images, shuffle=True)
    images, labels = next(iter(test_loader))
    return images, labels

def get_predictions(model, images):
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
    return preds

def main():
    N_IMAGES = 5 
    DEVICE = config.DEVICE
    print(f"Using device: {DEVICE}")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    classifier, autoencoder = load_all_models(DEVICE)
    clean_images, true_labels = get_test_batch(N_IMAGES)
    clean_images, true_labels = clean_images.to(DEVICE), true_labels.to(DEVICE)

    fgsm_attack = attacks.get_fgsm_attack(classifier, eps=0.3)

    clean_preds = get_predictions(classifier, clean_images)

    with torch.enable_grad():
        clean_images.requires_grad = True
        adv_images = fgsm_attack(clean_images, true_labels)

    adv_preds = get_predictions(classifier, adv_images)

    with torch.no_grad():
        purified_images = autoencoder(adv_images)
    purified_preds = get_predictions(classifier, purified_images)

    print("Generating plot...")
    plot_purification_comparison(
        clean_images.detach(), adv_images.detach(), purified_images.detach(),
        clean_preds.cpu().numpy(), adv_preds.cpu().numpy(), purified_preds.cpu().numpy(),
        true_labels.cpu().numpy(),
        n=N_IMAGES
    )
    print("\nVisualization complete.")

if __name__ == "__main__":
    main()