# config.py

import torch
import os

# --- Project Paths ---
# Use os.path.join for cross-platform compatibility
DATA_DIR = "data"
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = "models"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# --- Data Files ---
TRAIN_FILE = os.path.join(DATA_PROCESSED_DIR, "training.pt")
TEST_FILE = os.path.join(DATA_PROCESSED_DIR, "test.pt")

# --- Model Files ---
BASELINE_MODEL_NAME = "cnn_classifier.pth"
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, BASELINE_MODEL_NAME)

# The "failed" simple autoencoder (for the notebook story)
FAILED_AE_MODEL_NAME = "autoencoder.pth"
FAILED_AE_MODEL_PATH = os.path.join(MODEL_DIR, FAILED_AE_MODEL_NAME)

# The "successful" denoising autoencoder
DAE_MODEL_NAME = "denoising_autoencoder.pth"
DAE_MODEL_PATH = os.path.join(MODEL_DIR, DAE_MODEL_NAME)

# --- Data Preprocessing ---
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
SEED = 42

# --- Denoising Autoencoder Training Hyperparameters ---
DAE_LEARNING_RATE = 1e-3
DAE_EPOCHS = 20
DAE_BATCH_SIZE = 128

# --- Detection Thresholds (for app.py) ---
# These are example thresholds based on your plots.
RECON_THRESHOLD = 0.4 
ENTROPY_THRESHOLD = 0.1 

# --- Figure Names ---
DETECTION_PLOT_NAME = "detection_feature_distributions.png"
DETECTION_PLOT_PATH = os.path.join(FIGURES_DIR, DETECTION_PLOT_NAME)
PURIFICATION_PLOT_NAME = "purification_visualization.png"
PURIFICATION_PLOT_PATH = os.path.join(FIGURES_DIR, PURIFICATION_PLOT_NAME)
ATTACK_VISUALIZATION_PLOT_PATH = os.path.join(FIGURES_DIR, "attack_visualization.png")