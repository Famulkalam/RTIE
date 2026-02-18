"""
RTIE — Robustness Testing Module

Tests model resilience under perturbations:
    - Gaussian noise (σ = 5, 10, 15)
    - Temperature shift (±2°C, ±5°C)
    - Gaussian blur (kernel 3, 5, 7)
    - Rotation (±10°, ±20°)

Goal: <5% accuracy degradation.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import config
from model import RTIEModel
from dataset import get_dataloaders


# Normalization constants (EfficientNet standard)
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def load_model(device):
    model = RTIEModel(pretrained=False).to(device)
    ckpt = torch.load(os.path.join(config.MODEL_DIR, "best_model.pth"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def denormalize(tensor):
    """Convert normalized tensor (B, C, H, W) to [0, 1] range."""
    # tensor: (B, 3, H, W)
    mean = MEAN.view(1, 3, 1, 1).to(tensor.device)
    std = STD.view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


def normalize(tensor):
    """Convert [0, 1] tensor to normalized range."""
    mean = MEAN.view(1, 3, 1, 1).to(tensor.device)
    std = STD.view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std


def apply_noise(images, sigma):
    """Add Gaussian noise to image tensor."""
    # Unnormalize to [0, 1]
    x = denormalize(images)
    
    # Add noise (sigma is in 0-255 scale)
    noise = torch.randn_like(x) * (sigma / 255.0)
    x = torch.clamp(x + noise, 0, 1)
    
    # Renormalize
    return normalize(x)


def apply_temp_shift(images, shift):
    """Simulate temperature shift by adjusting pixel intensity."""
    # Unnormalize
    x = denormalize(images)
    
    # Apply shift (approximate: shift / 100 of the range)
    delta = shift / 100.0
    x = torch.clamp(x + delta, 0, 1)
    
    # Renormalize
    return normalize(x)


def apply_blur(images, kernel_size):
    """Apply Gaussian blur to images."""
    # Unnormalize
    x = denormalize(images)
    
    # Apply blur
    blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1.0, 2.0))
    x = blur(x)
    
    # Renormalize
    return normalize(x)


def apply_rotation(images, angle):
    """Rotate images by given angle."""
    # Unnormalize
    x = denormalize(images)
    
    # To PIL for rotation
    rotated = []
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    
    for img in x:
        # img is (3, H, W) in [0, 1]
        pil_img = to_pil(img.cpu())
        pil_img = pil_img.rotate(angle, fillcolor=0)
        rotated.append(to_tensor(pil_img))
    
    x_rot = torch.stack(rotated).to(x.device)
    
    # Renormalize
    return normalize(x_rot)


def evaluate_with_perturbation(model, test_loader, perturbation_fn, param, device):
    """Evaluate model accuracy under a specific perturbation."""
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, physics, labels, _ in test_loader:
            images = images.to(device)
            physics = physics.to(device)
            
            # Apply perturbation
            if param is not None:
                images = perturbation_fn(images, param)
            else:
                images = perturbation_fn(images, None)  # Baseline identity

            outputs = model(images, physics)
            preds = outputs["logits"].argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return accuracy_score(all_labels, all_preds)


def run_robustness_tests():
    """Run all perturbation tests and generate report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(device)
    _, _, test_loader = get_dataloaders()
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    # Baseline accuracy
    print("Computing baseline accuracy...")
    baseline_acc = evaluate_with_perturbation(model, test_loader, lambda x, _: x, None, device)
    print(f"  Baseline: {baseline_acc:.4f}")

    results = {"baseline": baseline_acc}

    # ── Perturbation Tests ──
    perturbations = [
        ("Gaussian Noise", apply_noise, config.ROBUSTNESS_NOISE_SIGMAS, "σ"),
        ("Temperature Shift", apply_temp_shift, config.ROBUSTNESS_TEMP_SHIFTS, "°C"),
        ("Gaussian Blur", apply_blur, config.ROBUSTNESS_BLUR_KERNELS, "kernel"),
        ("Rotation", apply_rotation, config.ROBUSTNESS_ROTATIONS, "°"),
    ]

    all_results = {}
    for name, fn, params, unit in perturbations:
        print(f"\nTesting: {name}")
        test_results = {}
        for p in tqdm(params, desc=f"  {name}"):
            acc = evaluate_with_perturbation(model, test_loader, fn, p, device)
            degradation = (baseline_acc - acc) * 100
            test_results[f"{p}{unit}"] = {
                "accuracy": acc,
                "degradation_pct": degradation,
            }
            status = "✓" if degradation < 5 else "✗"
            print(f"    {p:>5}{unit}: Acc={acc:.4f} | Δ={degradation:+.2f}% {status}")

        all_results[name] = test_results

    results["perturbations"] = all_results

    # ── Generate Chart ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RTIE — Robustness Test Results", fontsize=16, fontweight="bold")

    for ax, (name, fn, params, unit) in zip(axes.flatten(), perturbations):
        accs = [all_results[name][f"{p}{unit}"]["accuracy"] for p in params]
        degs = [all_results[name][f"{p}{unit}"]["degradation_pct"] for p in params]
        labels = [f"{p}{unit}" for p in params]

        bars = ax.bar(labels, accs, color=["#4CAF50" if d < 5 else "#F44336" for d in degs], alpha=0.8)
        ax.axhline(y=baseline_acc, color="blue", linestyle="--", alpha=0.5, label=f"Baseline ({baseline_acc:.3f})")
        ax.axhline(y=baseline_acc - 0.05, color="red", linestyle=":", alpha=0.5, label="5% threshold")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(max(0, min(accs) - 0.1), 1.0)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(config.REPORT_DIR, "robustness_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nRobustness chart → {chart_path}")

    # Save results JSON
    import json
    json_path = os.path.join(config.REPORT_DIR, "robustness_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON → {json_path}")

    return results


if __name__ == "__main__":
    run_robustness_tests()
