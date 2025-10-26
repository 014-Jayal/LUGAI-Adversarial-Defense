# attacks.py

import torchattacks
import config # Use the new config file

def get_fgsm_attack(model, eps=0.3):
    atk = torchattacks.FGSM(model, eps=eps)
    atk.set_normalization_used(mean=config.MNIST_MEAN, std=config.MNIST_STD)
    return atk

def get_pgd_attack(model, eps=0.3, alpha=2/255, steps=7):
    atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    atk.set_normalization_used(mean=config.MNIST_MEAN, std=config.MNIST_STD)
    return atk

def get_deepfool_attack(model, steps=50, overshoot=0.02):
    atk = torchattacks.DeepFool(model, steps=steps, overshoot=overshoot)
    atk.set_normalization_used(mean=config.MNIST_MEAN, std=config.MNIST_STD)
    return atk