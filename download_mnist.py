import os
import torch
from torchvision import datasets, transforms

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST(root=RAW_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=RAW_DIR, train=False, download=True, transform=transform)

    # Manually save processed tensors
    train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    torch.save((train_data, train_labels), os.path.join(PROCESSED_DIR, "training.pt"))

    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    torch.save((test_data, test_labels), os.path.join(PROCESSED_DIR, "test.pt"))

    print(f"✅ MNIST downloaded and processed successfully!")
    print(f"📂 Processed training tensor: {train_data.shape}")
    print(f"📂 Processed test tensor: {test_data.shape}")

if __name__ == "__main__":
    download_mnist()
