"""
RTIE — Field Integration Simulation

Simulates real-world deployment variability:
    - Different radiator aspect ratios (tall, wide, square)
    - Camera distance changes (zoom in/out)
    - Thermal sensor scale drift (calibration shifts)

Demonstrates deployment awareness and model resilience.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
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


def apply_aspect_ratio(images, target_ratio):
    """Simulate different radiator aspect ratios via asymmetric resize."""
    # Unnormalize
    x = denormalize(images)
    
    h_ratio, w_ratio = target_ratio
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    result = []
    for img in x:
        pil = to_pil(img.cpu())
        # Resize to intermediate aspect ratio, then back to 224x224
        intermediate_h = int(224 * h_ratio / 64)
        intermediate_w = int(224 * w_ratio / 32)
        pil = pil.resize((intermediate_w, intermediate_h), Image.BICUBIC)
        pil = pil.resize((224, 224), Image.BICUBIC)
        result.append(to_tensor(pil))
    
    x_aug = torch.stack(result).to(x.device)
    return normalize(x_aug)


def apply_scale(images, scale_factor):
    """Simulate camera distance by scaling and center-cropping."""
    # Unnormalize
    x = denormalize(images)
    
    result = []
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    for img in x:
        pil = to_pil(img.cpu())
        w, h = pil.size
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        pil = pil.resize((new_w, new_h), Image.BICUBIC)

        # Center crop back to 224x224
        left = max(0, (new_w - 224) // 2)
        top = max(0, (new_h - 224) // 2)
        right = left + 224
        bottom = top + 224

        if new_w < 224 or new_h < 224:
            # Pad if smaller
            padded = Image.new("RGB", (224, 224), (0, 0, 0))
            paste_x = max(0, (224 - new_w) // 2)
            paste_y = max(0, (224 - new_h) // 2)
            padded.paste(pil, (paste_x, paste_y))
            pil = padded
        else:
            pil = pil.crop((left, top, right, bottom))

        result.append(to_tensor(pil))
    
    x_aug = torch.stack(result).to(x.device)
    return normalize(x_aug)


def apply_thermal_drift(images, temp_shift, contrast_scale):
    """Simulate sensor calibration differences."""
    # Unnormalize
    x = denormalize(images)
    
    # Shift intensity (temperature offset in [0,1] space)
    shifted = x + (temp_shift / 100.0)
    
    # Scale contrast around mean
    mean = shifted.mean(dim=(1, 2, 3), keepdim=True)
    scaled = mean + (shifted - mean) * contrast_scale
    
    x = torch.clamp(scaled, 0, 1)
    
    # Renormalize
    return normalize(x)


def evaluate_perturbation(model, test_loader, transform_fn, device):
    """Evaluate accuracy under a transformation."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, physics, labels, _ in test_loader:
            images = images.to(device)
            physics = physics.to(device)
            
            # Apply transformation
            images = transform_fn(images)
            ### IMPORTANT: Do not double-normalize if transform_fn handles it
            
            outputs = model(images, physics)
            preds = outputs["logits"].argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)


def run_field_simulation():
    """Run all field integration simulations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(device)
    _, _, test_loader = get_dataloaders()
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    # Baseline
    print("Computing baseline accuracy...")
    baseline = evaluate_perturbation(model, test_loader, lambda x: x, device)
    print(f"Baseline accuracy: {baseline:.4f}\n")

    results = {"baseline": baseline, "simulations": {}}

    # ── Test 1: Aspect Ratios ──
    print("Testing aspect ratio variations...")
    ar_results = {}
    for ratio in config.FIELD_ASPECT_RATIOS:
        label = f"{ratio[0]}x{ratio[1]}"
        acc = evaluate_perturbation(model, test_loader,
                                     lambda x, r=ratio: apply_aspect_ratio(x, r), device)
        deg = (baseline - acc) * 100
        ar_results[label] = {"accuracy": acc, "degradation_pct": deg}
        print(f"  {label}: Acc={acc:.4f} (Δ={deg:+.2f}%)")
    results["simulations"]["aspect_ratio"] = ar_results

    # ── Test 2: Camera Distance ──
    print("\nTesting camera distance (scale) variations...")
    scale_results = {}
    for sf in config.FIELD_SCALE_FACTORS:
        label = f"{sf:.1f}x"
        acc = evaluate_perturbation(model, test_loader,
                                     lambda x, s=sf: apply_scale(x, s), device)
        deg = (baseline - acc) * 100
        scale_results[label] = {"accuracy": acc, "degradation_pct": deg}
        print(f"  {label}: Acc={acc:.4f} (Δ={deg:+.2f}%)")
    results["simulations"]["camera_distance"] = scale_results

    # ── Test 3: Thermal Drift ──
    print("\nTesting thermal scale drift...")
    drift_results = {}
    for ts in config.FIELD_TEMP_SHIFTS:
        for cs in config.FIELD_CONTRAST_SCALES:
            label = f"shift={ts}°C,contrast={cs}"
            acc = evaluate_perturbation(model, test_loader,
                                         lambda x, t=ts, c=cs: apply_thermal_drift(x, t, c), device)
            deg = (baseline - acc) * 100
            drift_results[label] = {"accuracy": acc, "degradation_pct": deg}
            print(f"  {label}: Acc={acc:.4f} (Δ={deg:+.2f}%)")
    results["simulations"]["thermal_drift"] = drift_results

    # ── Generate Chart ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("RTIE — Field Integration Simulation Results", fontsize=16, fontweight="bold")

    # Aspect ratio chart
    ax = axes[0]
    labels_ar = list(ar_results.keys())
    accs_ar = [ar_results[k]["accuracy"] for k in labels_ar]
    ax.bar(labels_ar, accs_ar, color="#2196F3", alpha=0.8)
    ax.axhline(y=baseline, color="green", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax.set_title("Aspect Ratio Variations", fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Scale chart
    ax = axes[1]
    labels_sc = list(scale_results.keys())
    accs_sc = [scale_results[k]["accuracy"] for k in labels_sc]
    ax.bar(labels_sc, accs_sc, color="#FF9800", alpha=0.8)
    ax.axhline(y=baseline, color="green", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax.set_title("Camera Distance Variations", fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Thermal drift chart
    ax = axes[2]
    labels_td = list(drift_results.keys())
    accs_td = [drift_results[k]["accuracy"] for k in labels_td]
    # Use shorter labels
    short_labels = [f"T{ts}C{cs}" for ts in config.FIELD_TEMP_SHIFTS for cs in config.FIELD_CONTRAST_SCALES]
    ax.barh(short_labels, accs_td, color="#9C27B0", alpha=0.8)
    ax.axvline(x=baseline, color="green", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax.set_title("Thermal Scale Drift", fontweight="bold")
    ax.set_xlabel("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(config.REPORT_DIR, "field_simulation.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nField simulation chart → {chart_path}")

    # Save JSON
    json_path = os.path.join(config.REPORT_DIR, "field_simulation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON → {json_path}")

    return results


if __name__ == "__main__":
    run_field_simulation()
