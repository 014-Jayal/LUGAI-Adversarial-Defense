<h1 align="center">🛡️ LUGAI — Latent Uncertainty Guided Adversary Intervention</h1>

<h3 align="center">A Real-Time Self-Healing Framework for Adversarial Defense in Deep Neural Networks</h3>

<p align="center">
Project Group No. 30
</p>

<p align="center">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch">
<img src="https://img.shields.io/badge/Adversarial%20AI-Security-blue">
<img src="https://img.shields.io/badge/Defense-Self--Healing-green">
<img src="https://img.shields.io/badge/Dataset-MNIST-orange">
<img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit">
<img src="https://img.shields.io/badge/Status-Completed-success">
</p>

---

## Overview

LUGAI (Latent Uncertainty Guided Adversary Intervention) is a self-healing adversarial defense framework that protects deep neural networks against maliciously crafted inputs.

The framework combines:

- Adversarial Threat Detection
- Input Purification
- Prediction Recovery
- Real-Time Visualization

Unlike conventional defenses that only detect attacks, LUGAI actively restores corrupted inputs using a Denoising Autoencoder (DAE), enabling the classifier to recover from adversarial degradation.

---

## Project Showcase

<table>
<tr>
<td align="center" width="33%">
<b>Live Detection & Purification Demo</b><br><br>
<img src="assets/lugai_demo.png" width="100%"><br>
Real-time visualization of the complete defense pipeline.
</td>

<td align="center" width="33%">
<b>Visual Proof of Purification</b><br><br>
<img src="assets/purification_results.png" width="100%"><br>
Clean → Attacked → Healed image recovery.
</td>

<td align="center" width="33%">
<b>Detection Feature Analysis</b><br><br>
<img src="assets/detection_histograms.png" width="100%"><br>
Clear separation of clean and adversarial samples.
</td>
</tr>
</table>

---

## System Architecture

<p align="center">
<img src="assets/lugai_architecture.png" width="900">
</p>

```text
User Input
    │
    ▼
Attack Generator
(FGSM / PGD / DeepFool)
    │
    ▼
Stage 1: Threat Detection
 ├─ Softmax Entropy
 └─ Reconstruction Error
    │
    ▼
Threat Decision
    │
    ▼
Stage 2: Purification
(Denoising Autoencoder)
    │
    ▼
CNN Classifier
    │
    ▼
Recovered Prediction
```

---

## How LUGAI Works

### Stage 1 — Detection

LUGAI analyzes every incoming image using two signals:

- **Softmax Entropy** (model uncertainty)
- **Reconstruction Error** (distance from clean data manifold)

If either metric exceeds predefined thresholds, the image is flagged as adversarial.

### Stage 2 — Purification

Flagged inputs are passed through a Denoising Autoencoder (DAE), which removes adversarial perturbations and reconstructs a cleaner version of the image.

### Stage 3 — Recovery

The purified image is classified again, restoring the original prediction whenever possible.

---

## Experimental Results

| Metric | Accuracy (%) |
|----------|----------|
| Clean Model Accuracy | 99.28 |
| Attacked Accuracy (FGSM ε=0.3) | 22.97 |
| Purified Accuracy (LUGAI) | 93.02 |
| Recovery Rate | 91.80 |

---

## Key Features

- Dual-signal adversarial detection
- Entropy-based uncertainty estimation
- Reconstruction-error validation
- Denoising Autoencoder purification
- FGSM, PGD and DeepFool support
- Real-time Streamlit dashboard
- Interactive attack simulation
- Accuracy recovery benchmarking

---

## Technology Stack

### Core
- Python 3.11
- PyTorch
- Torchattacks
- Streamlit

### Visualization
- Matplotlib
- Seaborn

### Data & Evaluation
- NumPy
- Scikit-Learn
- TQDM

---

## Installation

```bash
git clone https://github.com/014-Jayal/LUGAI-Adversarial-Defense.git
cd LUGAI-Adversarial-Defense

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset Preparation

```bash
python data_utils.py
```

This downloads and preprocesses the MNIST dataset.

---

## Training

### Train CNN

```bash
python train_baseline.py
```

### Train Denoising Autoencoder

```bash
python train_denoising_autoencoder.py
```

---

## Evaluation

```bash
python evaluate_attacks.py
python evaluate_detection.py
python evaluate_purification.py
```

---

## Run Demo

```bash
streamlit run app.py
```

---

## Repository Structure

```text
LUGAI-Adversarial-Defense/
├── assets/
│   ├── lugai_demo.png
│   ├── purification_results.png
│   ├── detection_histograms.png
│   └── lugai_architecture.png
├── data/
├── models/
├── results/
├── app.py
├── attacks.py
├── defense.py
├── model_zoo.py
├── config.py
├── train_baseline.py
├── train_denoising_autoencoder.py
├── evaluate_attacks.py
├── evaluate_detection.py
├── evaluate_purification.py
└── README.md
```

---

## Research Contributions

- Self-healing adversarial defense framework
- Dual-signal threat detection
- Automatic input purification
- Real-time interactive defense dashboard
- >91% recovery of lost accuracy

---

## Future Work

- CIFAR-10 evaluation
- GTSRB evaluation
- Adaptive attack resistance
- U-Net purification models
- ROC-based threshold optimization
- Edge deployment

---

## Team

### Jayal Shah
### Sakshi Makwana
### Mayank Jangid

**Supervisor:** Dr. Sanjay B. Sonar

---

## Citation

```bibtex
@misc{lugai2026,
title={LUGAI: Latent Uncertainty Guided Adversary Intervention},
author={Jayal Shah and Sakshi Makwana and Mayank Jangid},
year={2026}
}
```
