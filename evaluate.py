"""
RTIE — Evaluation Module

Generates:
    - Classification report (per-class precision, recall, F1)
    - Confusion matrix heatmap
    - Efficiency regression MAE and R²
    - Expected Calibration Error (ECE)
    - Calibration curve (reliability diagram)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
    accuracy_score,
)
from tqdm import tqdm

import config
from model import RTIEModel
from dataset import get_dataloaders


def load_model(device):
    """Load best trained model."""
    model = RTIEModel(pretrained=False).to(device)
    ckpt_path = os.path.join(config.MODEL_DIR, "best_model.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_f1']:.4f})")
    return model


class ModelWithTemperature(torch.nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling.
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        # Optimize log(T) to ensure T > 0
        self.log_temperature = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images, physics):
        outputs = self.model(images, physics)
        return outputs

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.log_temperature.exp().unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader, device):
        """
        Tune the temperature of the model (using the validation set).
        """
        self.eval()
        nll_criterion = torch.nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for images, physics, labels, _ in valid_loader:
                images = images.to(device)
                physics = physics.to(device)
                outputs = self.model(images, physics)
                logits_list.append(outputs["logits"])
                labels_list.append(labels.to(device))
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before Temperature Scaling - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: Optimize the temperature parameter
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.log_temperature.exp().item())
        print('After Temperature Scaling - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    """
    def __init__(self, n_bins=15):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def compute_ece(confidences, predictions, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) manually for reporting.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= low) & (confidences < high)
        n_bin = mask.sum()

        if n_bin > 0:
            bin_acc = (predictions[mask] == labels[mask]).mean()
            bin_conf = confidences[mask].mean()
            ece += abs(bin_acc - bin_conf) * (n_bin / len(confidences))
            bin_data.append({
                "bin_lower": low, "bin_upper": high,
                "accuracy": float(bin_acc), "confidence": float(bin_conf),
                "count": int(n_bin),
            })
        else:
            bin_data.append({
                "bin_lower": low, "bin_upper": high,
                "accuracy": 0, "confidence": 0, "count": 0,
            })

    return float(ece), bin_data


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("RTIE — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {save_path}")


def plot_calibration_curve(bin_data, ece, save_path):
    """Plot reliability diagram (calibration curve)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Reliability diagram
    accs = [b["accuracy"] for b in bin_data]
    confs = [b["confidence"] for b in bin_data]
    counts = [b["count"] for b in bin_data]
    bin_centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bin_data]

    ax1.bar(bin_centers, accs, width=0.08, alpha=0.6, label="Actual accuracy", color="#2196F3")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.scatter(confs, accs, color="red", s=50, zorder=5, label="Bin centers")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax1.set_ylabel("Fraction of Positives", fontsize=11)
    ax1.set_title(f"Calibration Curve (ECE = {ece:.4f})", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Histogram of confidence scores
    ax2.bar(bin_centers, counts, width=0.08, color="#4CAF50", alpha=0.7)
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Confidence Distribution", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Calibration curve → {save_path}")


def evaluate():
    """Run full evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(device)
    _, _, test_loader = get_dataloaders()

    os.makedirs(config.REPORT_DIR, exist_ok=True)

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    all_eff_pred = []
    all_eff_true = []
    all_confidence = []

    print("\nEvaluating on test set...")
    
    # Apply Temperature Scaling
    print("Performing Temperature Scaling...")
    _, val_loader, _ = get_dataloaders()
    scaled_model = ModelWithTemperature(model)
    scaled_model.to(device)
    scaled_model.set_temperature(val_loader, device)

    with torch.no_grad():
        for images, physics, labels, efficiencies in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            physics = physics.to(device)

            outputs = scaled_model(images, physics)  # Forward pass
            
            # Use temperature scaled logits for probabilities
            logits = outputs["logits"]
            scaled_logits = scaled_model.temperature_scale(logits)
            probs = F.softmax(scaled_logits, dim=-1)

            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_eff_pred.extend(outputs["efficiency"].cpu().numpy())
            all_eff_true.extend(efficiencies.numpy())
            all_confidence.extend(probs.max(dim=-1).values.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_eff_pred = np.array(all_eff_pred)
    all_eff_true = np.array(all_eff_true)
    all_confidence = np.array(all_confidence)

    # ── Classification Report ──
    report = classification_report(
        all_labels, all_preds,
        target_names=config.CLASS_NAMES,
        digits=4,
    )
    accuracy = accuracy_score(all_labels, all_preds)

    # ── Efficiency Metrics ──
    eff_mae = mean_absolute_error(all_eff_true, all_eff_pred)
    eff_r2 = r2_score(all_eff_true, all_eff_pred)

    # ── ECE ──
    ece, bin_data = compute_ece(all_confidence, all_preds, all_labels)

    # ── Write Report ──
    report_text = (
        "=" * 60 + "\n"
        "RTIE — Evaluation Report\n"
        "=" * 60 + "\n\n"
        f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n"
        "Classification Report:\n" + report + "\n\n"
        f"Efficiency Regression:\n"
        f"  MAE: {eff_mae:.2f}\n"
        f"  R²:  {eff_r2:.4f}\n\n"
        f"Expected Calibration Error (ECE): {ece:.4f}\n"
    )
    print(report_text)

    report_path = os.path.join(config.REPORT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Report → {report_path}")

    # ── Plots ──
    plot_confusion_matrix(
        all_labels, all_preds,
        os.path.join(config.REPORT_DIR, "confusion_matrix.png"),
    )
    plot_calibration_curve(
        bin_data, ece,
        os.path.join(config.REPORT_DIR, "calibration_curve.png"),
    )

    # Save results as JSON
    results = {
        "accuracy": float(accuracy),
        "efficiency_mae": float(eff_mae),
        "efficiency_r2": float(eff_r2),
        "ece": float(ece),
        "calibration_bins": bin_data,
    }
    with open(os.path.join(config.REPORT_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    evaluate()
