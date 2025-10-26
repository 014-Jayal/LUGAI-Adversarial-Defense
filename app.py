# app.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# Use simple imports
import config
import attacks
from model_zoo import BaseCNN, Autoencoder
from visualize import unnormalize
from defense import calculate_softmax_entropy, get_reconstruction_error

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="LUGAI Demo",
    page_icon="🛡️",
    layout="wide", # Use wide layout for better spacing
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models(device):
    """Loads all models and returns them."""
    print("Loading models...")
    classifier = BaseCNN().to(device)
    classifier.load_state_dict(torch.load(config.BASELINE_MODEL_PATH, map_location=device))
    classifier.eval()

    purifier = Autoencoder().to(device)
    purifier.load_state_dict(torch.load(config.DAE_MODEL_PATH, map_location=device))
    purifier.eval()

    print("Models loaded successfully.")
    return classifier, purifier

@st.cache_data # Cache the data loading as well
def load_test_data():
    """Loads the test dataset for image selection."""
    test_images, test_labels = torch.load(config.TEST_FILE)
    return test_images, test_labels

def get_predictions(model, images_tensor):
    """Gets predictions and softmax probabilities."""
    with torch.no_grad():
        logits = model(images_tensor)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        conf = torch.max(probs, dim=1).values
    return logits, preds, conf

# --- Main App ---

# --- Header ---
st.title("🛡️ LUGAI: Self-Healing AI Defense Demo")
st.markdown("---") # Add a horizontal line
st.markdown(
    """
    This interactive demo showcases LUGAI, a two-stage framework designed to protect neural networks
    from adversarial attacks in real-time.

    **👈 Use the sidebar** to select an MNIST digit and apply an attack. Watch LUGAI detect the threat
    and attempt to purify the image to restore the correct prediction!
    """
)
st.markdown("---") # Add another horizontal line

# --- Load Models and Data ---
DEVICE = config.DEVICE
try:
    classifier, purifier = load_models(DEVICE)
    test_images, test_labels = load_test_data()
except FileNotFoundError as e:
    st.error(f"🚨 **Error loading files:** {e}. Please ensure you have run the necessary setup scripts (`data_utils.py`, `train_baseline.py`, `train_denoising_autoencoder.py`) first.")
    st.stop()


# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
st.sidebar.markdown("Select an image and attack parameters.")

image_index = st.sidebar.slider("Select Image Index:", 0, 9999, 10, key="image_slider")
attack_type = st.sidebar.selectbox("Select Attack Type:", ["None", "FGSM", "PGD"], key="attack_select")

# Display selected image preview in sidebar
st.sidebar.subheader("Selected Image:")
clean_img_sidebar, _ = test_images[image_index], test_labels[image_index]
# Unnormalize for display
plot_clean_sidebar = unnormalize(clean_img_sidebar.cpu(), config.MNIST_MEAN, config.MNIST_STD).squeeze()
st.sidebar.image(plot_clean_sidebar.numpy(), width='stretch')

# Attack parameters (conditionally shown)
epsilon = 0.3 # Default value
if attack_type == "FGSM":
    epsilon = st.sidebar.slider("FGSM Epsilon (ε):", 0.05, 0.5, 0.3, 0.05, key="fgsm_eps")
elif attack_type == "PGD":
    epsilon = st.sidebar.slider("PGD Epsilon (ε):", 0.05, 0.5, 0.3, 0.05, key="pgd_eps")

# Get the selected clean image and label for processing
clean_img, true_label = test_images[image_index], test_labels[image_index]
clean_img = clean_img.unsqueeze(0).to(DEVICE) # Add batch dim
true_label = true_label.unsqueeze(0).to(DEVICE)

# --- Attack Generation ---
input_img = clean_img # Start with clean image
attack_applied = "None"
if attack_type == "FGSM":
    atk = attacks.get_fgsm_attack(classifier, eps=epsilon)
    with torch.enable_grad(): # Enable grad for attack generation
        clean_img.requires_grad = True
        input_img = atk(clean_img, true_label)
    attack_applied = f"FGSM (ε={epsilon})"
elif attack_type == "PGD":
    atk = attacks.get_pgd_attack(classifier, eps=epsilon, steps=7)
    with torch.enable_grad(): # Enable grad for attack generation
        clean_img.requires_grad = True
        input_img = atk(clean_img, true_label)
    attack_applied = f"PGD (ε={epsilon})"

input_img = input_img.detach() # Detach after attack generation

# --- LUGAI Pipeline ---
with torch.no_grad(): # No gradients needed for inference
    # 1. Get initial prediction on the (potentially attacked) input
    logits_adv, pred_adv, conf_adv = get_predictions(classifier, input_img)

    # --- STAGE 1: DETECTION ---
    recon_criterion = nn.MSELoss(reduction='none') # Need per-sample error
    recon_error = get_reconstruction_error(purifier, input_img, recon_criterion).item()
    entropy = calculate_softmax_entropy(logits_adv).item()

    # Apply detection thresholds
    # --- FIX: Removed extra 'S' ---
    is_adversarial = (recon_error > config.RECON_THRESHOLD) or (entropy > config.ENTROPY_THRESHOLD)

    # --- STAGE 2: PURIFICATION ---
    if is_adversarial:
        purified_img = purifier(input_img)
        logits_final, pred_final, conf_final = get_predictions(classifier, purified_img)
        purification_applied = True
    else:
        # If not detected, purification is skipped
        purified_img = input_img
        logits_final, pred_final, conf_final = logits_adv, pred_adv, conf_adv
        purification_applied = False

# --- Display Results ---
st.subheader("📊 LUGAI Pipeline Visualization")

# Un-normalize images for plotting
plot_clean = unnormalize(clean_img.detach().cpu(), config.MNIST_MEAN, config.MNIST_STD).squeeze()
plot_input = unnormalize(input_img.detach().cpu(), config.MNIST_MEAN, config.MNIST_STD).squeeze()
plot_purified = unnormalize(purified_img.detach().cpu(), config.MNIST_MEAN, config.MNIST_STD).squeeze()

# Create columns for layout
col1, col2, col3 = st.columns(3, gap="medium")

# --- Column 1: Original Image ---
with col1:
    st.markdown("#### 1. Original Image")
    # --- FIX: Updated width parameter ---
    st.image(plot_clean.numpy(), caption=f"Ground Truth: {true_label.item()}", width='stretch')
    logits_clean, pred_clean, conf_clean = get_predictions(classifier, clean_img.detach())
    st.metric(label="Classifier Prediction (Clean)", value=pred_clean.item())
    st.write(f"Confidence: {conf_clean.item():.2f}")

# --- Column 2: Input to LUGAI ---
with col2:
    st.markdown(f"#### 2. Input ({attack_applied})")
    # --- FIX: Updated width parameter ---
    st.image(plot_input.numpy(), caption="Image fed into LUGAI", width='stretch')

    pred_adv_val = pred_adv.item()
    if pred_adv_val == true_label.item():
        st.success(f"✔️ Initial Prediction: {pred_adv_val}")
    else:
        st.error(f"❌ Initial Prediction: {pred_adv_val}")
    st.write(f"Confidence: {conf_adv.item():.2f}")
    st.markdown("---")
    st.markdown("**Detection Features:**")
    st.write(f"- Reconstruction Error: `{recon_error:.4f}`")
    st.write(f"- Softmax Entropy: `{entropy:.4f}`")

# --- Column 3: LUGAI Output ---
with col3:
    st.markdown("#### 3. LUGAI Output")

    # Detection Status Box
    with st.container(border=True): # Add a border around detection info
        st.markdown("**Stage 1: Detection Result**")
        if is_adversarial:
            st.warning("⚠️ **Threat Detected!** Proceeding to Purification.")
            st.caption(f"(Recon Error > {config.RECON_THRESHOLD} OR Entropy > {config.ENTROPY_THRESHOLD})")
        else:
            st.success("✅ **Input Appears Clean.** Purification skipped.")
            st.caption(f"(Recon Error ≤ {config.RECON_THRESHOLD} AND Entropy ≤ {config.ENTROPY_THRESHOLD})")

    # Purification Image (show original if skipped)
    # --- FIX: Updated width parameter ---
    st.image(plot_purified.numpy(), caption="Image after LUGAI Processing", width='stretch')

    # Final Prediction Box
    with st.container(border=True): # Add a border around final prediction
        st.markdown("**Stage 2: Final Prediction**")
        if purification_applied:
            st.caption("Prediction based on the *purified* image.")
        else:
            st.caption("Prediction based on the *original input* (purification skipped).")

        pred_final_val = pred_final.item()
        if pred_final_val == true_label.item():
            st.success(f"✔️ Final Prediction: {pred_final_val}")
        else:
            st.error(f"❌ Final Prediction: {pred_final_val}")
        st.write(f"Confidence: {conf_final.item():.2f}")

st.markdown("---")
st.caption("LUGAI Project | Jayal Shah, Sakshi Makwana, Mayank Jangid")