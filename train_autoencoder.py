# train_autoencoder.py
# This script trains the SIMPLE autoencoder (trained on clean images)
# This is the one that FAILED to purify, which is important for your project story.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm

from model_zoo import Autoencoder
import config

def load_data(train_file, test_file, batch_size):
    print(f"Loading processed data from {config.DATA_PROCESSED_DIR}...")
    
    train_images, _ = torch.load(train_file) # No labels needed
    test_images, _ = torch.load(test_file)   # No labels needed
    
    # Create TensorDatasets (input and target are the same)
    train_ds = TensorDataset(train_images, train_images)
    test_ds = TensorDataset(test_images, test_images)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Training Epoch")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets) # Compare output to original input
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(mse_loss=f"{loss.item():.6f}")
        
    return running_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            pbar.set_postfix(mse_loss=f"{loss.item():.6f}")
    return running_loss / len(test_loader)

def main():
    print(f"Training the SIMPLE (Failing) Autoencoder...")
    print(f"Using device: {config.DEVICE}")
    torch.manual_seed(config.SEED)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    train_loader, test_loader = load_data(
        config.TRAIN_FILE, config.TEST_FILE, config.DAE_BATCH_SIZE # Use DAE batch size
    )
    
    model = Autoencoder().to(config.DEVICE)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=config.DAE_LEARNING_RATE)
    
    best_test_loss = float('inf')
    
    for epoch in range(config.DAE_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.DAE_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        print(f"Epoch {epoch+1} Train Loss (MSE): {train_loss:.6f}")
        
        test_loss = evaluate(model, test_loader, criterion, config.DEVICE)
        print(f"Epoch {epoch+1} Test Loss (MSE): {test_loss:.6f}")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print(f"New best test loss: {best_test_loss:.6f}. Saving model...")
            # Save to the "FAILED" path
            torch.save(model.state_dict(), config.FAILED_AE_MODEL_PATH)
            
    print("\nTraining complete.")
    print(f"Best test MSE loss: {best_test_loss:.6f}")
    print(f"Simple autoencoder saved to: {config.FAILED_AE_MODEL_PATH}")

if __name__ == "__main__":
    main()