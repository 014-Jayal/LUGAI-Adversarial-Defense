# data_utils.py

import os
import torch
from torchvision import datasets, transforms
import config # Use the new config file

# Directories from config
os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

print("Starting data processing...")

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(config.MNIST_MEAN, config.MNIST_STD)
])

# Load datasets (downloads if not present)
print(f"Downloading training data to {config.DATA_RAW_DIR}...")
train_dataset = datasets.MNIST(root=config.DATA_RAW_DIR, train=True, download=True, transform=transform)
print(f"Downloading test data to {config.DATA_RAW_DIR}...")
test_dataset = datasets.MNIST(root=config.DATA_RAW_DIR, train=False, download=True, transform=transform)

# Function to save dataset as tensors
def save_as_tensors(dataset, save_path):
    images = []
    labels = []
    # Use torch.no_grad() for faster processing
    with torch.no_grad():
        for img, label in dataset:
            images.append(img)
            labels.append(label)
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)
    torch.save((images_tensor, labels_tensor), save_path)
    print(f"Saved {save_path} with {len(dataset)} samples.")

# Save processed datasets
save_as_tensors(train_dataset, config.TRAIN_FILE)
save_as_tensors(test_dataset, config.TEST_FILE)

print(f"✅ MNIST downloaded and processed successfully to {config.DATA_PROCESSED_DIR}!")