# train_baseline.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm

# Use simple imports
from model_zoo import BaseCNN
import config

def load_data(train_file, test_file, batch_size):
    """Loads processed .pt data and creates DataLoaders."""
    print(f"Loading processed data from {config.DATA_PROCESSED_DIR}...")
    
    train_images, train_labels = torch.load(train_file)
    test_images, test_labels = torch.load(test_file)

    print(f"Train tensors shape: {train_images.shape}, {train_labels.shape}")
    print(f"Test tensors shape: {test_images.shape}, {test_labels.shape}")

    train_ds = TensorDataset(train_images, train_labels)
    test_ds = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")
        
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix(acc=f"{100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    print(f"Using device: {config.DEVICE}")
    torch.manual_seed(config.SEED)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    train_loader, test_loader = load_data(
        config.TRAIN_FILE, config.TEST_FILE, config.BATCH_SIZE
    )
    
    model = BaseCNN().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_test_acc = 0.0
    print("\nStarting training...")
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, config.DEVICE
        )
        print(f"Epoch {epoch+1} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"New best test accuracy: {best_test_acc:.2f}%. Saving model...")
            torch.save(model.state_dict(), config.BASELINE_MODEL_PATH)
            
    print("\nTraining complete.")
    print(f"Best test accuracy achieved: {best_test_acc:.2f}%")
    print(f"Model saved to: {config.BASELINE_MODEL_PATH}")

if __name__ == "__main__":
    main()