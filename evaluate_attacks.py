# evaluate_attacks.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os

from model_zoo import BaseCNN
import config
import attacks

def load_test_data(test_file, batch_size):
    print(f"Loading test data from {test_file}...")
    test_images, test_labels = torch.load(test_file)
    print(f"Test tensors shape: {test_images.shape}, {test_labels.shape}")
    
    test_ds = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader

def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = BaseCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    return model

def evaluate_model(model, data_loader, device, attack=None):
    correct, total = 0, 0
    pbar_desc = "Evaluating (Adversarial)" if attack else "Evaluating (Clean)"
    pbar = tqdm(data_loader, desc=pbar_desc)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        if attack:
            # Enable grad for attack
            images.requires_grad = True
            adv_images = attack(images, labels)
            # Disable grad for forward pass
            with torch.no_grad():
                outputs = model(adv_images)
        else:
            with torch.no_grad():
                outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix(acc=f"{100.*correct/total:.2f}%")

    accuracy = 100. * correct / total
    print(f"Final Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    print(f"Using device: {config.DEVICE}")
    
    test_loader = load_test_data(config.TEST_FILE, config.BATCH_SIZE)
    model = load_model(config.BASELINE_MODEL_PATH, config.DEVICE)
    
    print("\n" + "="*30)
    print("Evaluating on CLEAN test data...")
    clean_acc = evaluate_model(model, test_loader, config.DEVICE, attack=None)
    
    print("\n" + "="*30)
    print("Evaluating on FGSM attack (eps=0.3)...")
    fgsm_attack = attacks.get_fgsm_attack(model, eps=0.3)
    fgsm_acc = evaluate_model(model, test_loader, config.DEVICE, attack=fgsm_attack)
    
    print("\n" + "="*30)
    print("Evaluating on PGD attack (eps=0.3)...")
    pgd_attack = attacks.get_pgd_attack(model, eps=0.3, steps=7)
    pgd_acc = evaluate_model(model, test_loader, config.DEVICE, attack=pgd_attack)
    
    print("\n" + "="*40)
    print("     ADVERSARIAL ATTACK SUMMARY")
    print("="*40)
    print(f"  Clean Model Accuracy:   {clean_acc:.2f}%")
    print(f"  FGSM Attack Accuracy:   {fgsm_acc:.2f}%")
    print(f"  PGD Attack Accuracy:    {pgd_acc:.2f}%")
    print("="*40)
    
    # Save results to a file for later use
    with open("results/attack_summary.txt", "w") as f:
        f.write(f"Clean_Accuracy: {clean_acc:.2f}\n")
        f.write(f"FGSM_Accuracy: {fgsm_acc:.2f}\n")
        f.write(f"PGD_Accuracy: {pgd_acc:.2f}\n")

if __name__ == "__main__":
    main()