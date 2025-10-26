# evaluate_purification.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import re

from model_zoo import BaseCNN, Autoencoder
import config
import attacks

# Function to read results from the text file
def get_accuracies_from_summary(file_path="results/attack_summary.txt"):
    try:
        with open(file_path, "r") as f:
            content = f.read()
        clean_acc = float(re.search(r"Clean_Accuracy: ([\d\.]+)", content).group(1))
        fgsm_acc = float(re.search(r"FGSM_Accuracy: ([\d\.]+)", content).group(1))
        return clean_acc, fgsm_acc
    except:
        print("Warning: 'results/attack_summary.txt' not found. Using default values.")
        print("Please run 'python evaluate_attacks.py' first for the exact report.")
        return 99.28, 22.97 # Fallback values

def load_all_models(device):
    print("Loading models...")
    classifier = BaseCNN().to(device)
    classifier.load_state_dict(torch.load(config.BASELINE_MODEL_PATH, map_location=device))
    classifier.eval()
    
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(config.DAE_MODEL_PATH, map_location=device))
    autoencoder.eval()
    
    return classifier, autoencoder

def load_test_data(test_file, batch_size):
    print(f"Loading test data from {test_file}...")
    test_images, test_labels = torch.load(test_file)
    print(f"Test tensors shape: {test_images.shape}, {test_labels.shape}")
    
    test_ds = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate_purification(classifier, autoencoder, data_loader, attack, device):
    correct, total = 0, 0
    pbar = tqdm(data_loader, desc="Evaluating Purified Accuracy")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 1. Create adversarial images
        with torch.enable_grad():
            images.requires_grad = True
            adv_images = attack(images, labels)
        
        # 2. Purify the adversarial images
        with torch.no_grad():
            purified_images = autoencoder(adv_images.detach())
        
        # 3. Classify the PURIFIED images
        with torch.no_grad():
            outputs = classifier(purified_images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix(acc=f"{100.*correct/total:.2f}%")

    accuracy = 100. * correct / total
    print(f"Final Purified Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    print(f"Using device: {config.DEVICE}")
    
    CLEAN_ACCURACY, ATTACKED_ACCURACY = get_accuracies_from_summary()
    
    test_loader = load_test_data(config.TEST_FILE, config.BATCH_SIZE)
    classifier, autoencoder = load_all_models(config.DEVICE)
    
    fgsm_attack = attacks.get_fgsm_attack(classifier, eps=0.3)
    
    print("\n" + "="*30)
    print("Evaluating purification performance...")
    purified_acc = evaluate_purification(
        classifier, autoencoder, test_loader, fgsm_attack, config.DEVICE
    )
    
    print("\n" + "="*40)
    print("     FINAL ACCURACY REPORT")
    print("="*40)
    print(f"  Clean Model Accuracy:     {CLEAN_ACCURACY:.2f}%")
    print(f"  Attacked (FGSM) Accuracy: {ATTACKED_ACCURACY:.2f}%")
    print(f"  Purified (DAE) Accuracy:  {purified_acc:.2f}%")
    print("="*40)
    
    recovery_rate = (purified_acc - ATTACKED_ACCURACY) / (CLEAN_ACCURACY - ATTACKED_ACCURACY) * 100
    print(f"\nAccuracy Recovery Rate: {recovery_rate:.2f}%")
    print("\nPurification evaluation complete.")

if __name__ == "__main__":
    main()