# visualize.py

import torch
import matplotlib.pyplot as plt
import os
import config # Use the new config file

def unnormalize(tensor, mean, std):
    """
    Un-normalizes a tensor for plotting.
    """
    tensor = tensor.clone()
    
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    
    if tensor.dim() == 4: # Batch of images
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif tensor.dim() == 3: # Single image
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    
    tensor.mul_(std).add_(mean)
    return torch.clamp(tensor, 0, 1)

def plot_attack_comparison(clean_images, adv_images, clean_preds, adv_preds, true_labels, n=5):
    """
    Plots a grid comparing clean and adversarial images.
    """
    n = min(n, len(clean_images))
    
    clean_images = unnormalize(clean_images, config.MNIST_MEAN, config.MNIST_STD)
    adv_images = unnormalize(adv_images, config.MNIST_MEAN, config.MNIST_STD)
    
    plt.figure(figsize=(10, n * 2.5))
    
    for i in range(n):
        clean_img = clean_images[i].cpu().squeeze()
        adv_img = adv_images[i].cpu().squeeze()
        clean_pred = clean_preds[i]
        adv_pred = adv_preds[i]
        true_label = true_labels[i]
        
        # --- Clean Image ---
        ax = plt.subplot(n, 2, 2*i + 1)
        plt.imshow(clean_img, cmap="gray")
        color = "green" if clean_pred == true_label else "red"
        ax.set_title(f"Clean (True: {true_label})\nPred: {clean_pred}", color=color)
        ax.axis("off")
        
        # --- Adversarial Image ---
        ax = plt.subplot(n, 2, 2*i + 2)
        plt.imshow(adv_img, cmap="gray")
        color = "red" if adv_pred != true_label else "green"
        ax.set_title(f"Adversarial (FGSM)\nPred: {adv_pred}", color=color)
        ax.axis("off")
        
    plt.tight_layout()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(config.ATTACK_VISUALIZATION_PLOT_PATH)
    print(f"\nSaved visualization to {config.ATTACK_VISUALIZATION_PLOT_PATH}")
    plt.show()

def plot_purification_comparison(clean_imgs, adv_imgs, purified_imgs, 
                                 clean_pred, adv_pred, purified_pred, true_label, n):
    """
    Plots a grid comparing clean, adversarial, and purified images.
    """
    clean_imgs = unnormalize(clean_imgs.cpu(), config.MNIST_MEAN, config.MNIST_STD)
    adv_imgs = unnormalize(adv_imgs.cpu(), config.MNIST_MEAN, config.MNIST_STD)
    purified_imgs = unnormalize(purified_imgs.cpu(), config.MNIST_MEAN, config.MNIST_STD)
    
    plt.figure(figsize=(12, n * 3))
    
    for i in range(n):
        ax = plt.subplot(n, 3, 3*i + 1)
        plt.imshow(clean_imgs[i].squeeze(), cmap="gray")
        color = "green" if clean_pred[i] == true_label[i] else "red"
        ax.set_title(f"Clean (True: {true_label[i]})\nPred: {clean_pred[i]}", color=color)
        ax.axis("off")
        
        ax = plt.subplot(n, 3, 3*i + 2)
        plt.imshow(adv_imgs[i].squeeze(), cmap="gray")
        color = "red" # Always red to highlight the attack
        ax.set_title(f"Adversarial (FGSM)\nPred: {adv_pred[i]}", color=color)
        ax.axis("off")
        
        ax = plt.subplot(n, 3, 3*i + 3)
        plt.imshow(purified_imgs[i].squeeze(), cmap="gray")
        color = "green" if purified_pred[i] == true_label[i] else "red"
        ax.set_title(f"Purified (DAE Output)\nPred: {purified_pred[i]}", color=color)
        ax.axis("off")
        
    plt.tight_layout()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(config.PURIFICATION_PLOT_PATH)
    print(f"\nSaved visualization to {config.PURIFICATION_PLOT_PATH}")
    plt.show()