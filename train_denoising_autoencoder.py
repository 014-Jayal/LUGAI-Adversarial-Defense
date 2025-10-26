# train_denoising_autoencoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm

# Use simple imports
from model_zoo import BaseCNN, Autoencoder
import config
import attacks

def load_data(train_file, test_file, batch_size):
    print(f"Loading processed data from {config.DATA_PROCESSED_DIR}...")
    train_images, train_labels = torch.load(train_file)
    test_images, test_labels = torch.load(test_file)

    train_ds = TensorDataset(train_images, train_labels)
    test_ds = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_classifier(device):
    print(f"Loading baseline classifier from {config.BASELINE_MODEL_PATH}...")
    classifier = BaseCNN().to(device)
    classifier.load_state_dict(torch.load(config.BASELINE_MODEL_PATH, map_location=device))
    classifier.eval()
    return classifier

def train_one_epoch(autoencoder, classifier_for_attack, attack,
                    train_loader, criterion, optimizer, device):
    autoencoder.train()
    classifier_for_attack.eval()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch")
    
    for clean_images, labels in pbar:
        clean_images, labels = clean_images.to(device), labels.to(device)
        
        # 1. Create adversarial images (INPUT)
        clean_images.requires_grad = True 
        adv_images = attack(clean_images, labels)
        adv_images = adv_images.detach()

        # 2. Train the autoencoder
        optimizer.zero_grad()
        purified_images = autoencoder(adv_images)
        
        # Calculate loss against CLEAN images (TARGET)
        loss = criterion(purified_images, clean_images.detach()) 
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(mse_loss=f"{loss.item():.6f}")
        
    return running_loss / len(train_loader)

def evaluate(autoencoder, classifier_for_attack, attack, 
             test_loader, criterion, device):
    autoencoder.eval()
    classifier_for_attack.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for clean_images, labels in pbar:
            clean_images, labels = clean_images.to(device), labels.to(device)
            
            # 1. Create adversarial images (INPUT)
            with torch.enable_grad():
                clean_images.requires_grad = True
                adv_images = attack(clean_images, labels)
            
            # 2. Get purified images
            purified_images = autoencoder(adv_images)
            
            # 3. Calculate loss against CLEAN images (TARGET)
            loss = criterion(purified_images, clean_images)
            running_loss += loss.item()
            pbar.set_postfix(mse_loss=f"{loss.item():.6f}")

    return running_loss / len(test_loader)

def main():
    print(f"Using device: {config.DEVICE}")
    torch.manual_seed(config.SEED)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    train_loader, test_loader = load_data(
        config.TRAIN_FILE, config.TEST_FILE, config.DAE_BATCH_SIZE
    )
    
    classifier = load_classifier(config.DEVICE)
    autoencoder = Autoencoder().to(config.DEVICE)
    
    attack_to_train_against = attacks.get_fgsm_attack(classifier, eps=0.3)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(autoencoder.parameters(), lr=config.DAE_LEARNING_RATE)
    
    best_test_loss = float('inf')
    
    print("\nStarting Denoising Autoencoder (DAE) training...")
    
    for epoch in range(config.DAE_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.DAE_EPOCHS} ---")
        
        train_loss = train_one_epoch(
            autoencoder, classifier, attack_to_train_against,
            train_loader, criterion, optimizer, config.DEVICE
        )
        print(f"Epoch {epoch+1} Train Loss (MSE): {train_loss:.6f}")
        
        test_loss = evaluate(
            autoencoder, classifier, attack_to_train_against,
            test_loader, criterion, config.DEVICE
        )
        print(f"Epoch {epoch+1} Test Loss (MSE): {test_loss:.6f}")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print(f"New best test loss: {best_test_loss:.6f}. Saving model...")
            torch.save(autoencoder.state_dict(), config.DAE_MODEL_PATH)
            
    print("\nTraining complete.")
    print(f"Best test MSE loss achieved: {best_test_loss:.6f}")
    print(f"Denoising Autoencoder model saved to: {config.DAE_MODEL_PATH}")

if __name__ == "__main__":
    main()