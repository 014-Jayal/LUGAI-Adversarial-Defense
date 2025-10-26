# visualize_attacks.py

import torch
import torch.nn.functional as F
import os
from torch.utils.data import TensorDataset, DataLoader

from model_zoo import BaseCNN
import config
import attacks
from visualize import plot_attack_comparison

def load_test_data(test_file, batch_size):
    test_images, test_labels = torch.load(test_file)
    test_ds = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(test_loader))
    return images, labels

def load_model(model_path, device):
    model = BaseCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    return model

def get_predictions(model, images):
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
    return preds

def main():
    print(f"Using device: {config.DEVICE}")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    N_IMAGES = 5
    clean_images, true_labels = load_test_data(config.TEST_FILE, N_IMAGES)
    clean_images, true_labels = clean_images.to(config.DEVICE), true_labels.to(config.DEVICE)
    
    model = load_model(config.BASELINE_MODEL_PATH, config.DEVICE)
    
    clean_preds = get_predictions(model, clean_images)
    
    print("Generating FGSM adversarial examples...")
    fgsm_attack = attacks.get_fgsm_attack(model, eps=0.3)
    
    with torch.enable_grad():
        clean_images.requires_grad = True
        adv_images = fgsm_attack(clean_images, true_labels)
    
    adv_preds = get_predictions(model, adv_images)
    
    print("Generating plot...")
    plot_attack_comparison(
        clean_images.detach(), 
        adv_images.detach(), 
        clean_preds.cpu().numpy(), 
        adv_preds.cpu().numpy(), 
        true_labels.cpu().numpy(),
        n=N_IMAGES
    )
    print("\nVisualization complete.")

if __name__ == "__main__":
    main()