"""
RTIE — Grad-CAM Explainability

Generates Grad-CAM heatmap overlays for each fault class,
showing where EfficientNet-B0 focuses its attention.

Uses pytorch-grad-cam targeting the final convolutional layer.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from model import RTIEModel
from dataset import get_dataloaders


def load_model(device):
    model = RTIEModel(pretrained=False).to(device)
    ckpt = torch.load(os.path.join(config.MODEL_DIR, "best_model.pth"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


class RTIEModelWrapper(torch.nn.Module):
    """Wrapper to make Grad-CAM work with our multi-input model."""
    def __init__(self, model, physics_features):
        super().__init__()
        self.model = model
        self.physics_features = physics_features

    def forward(self, x):
        outputs = self.model(x, self.physics_features)
        return outputs["logits"]


def get_target_layer(model):
    """Get the last convolutional layer of EfficientNet-B0."""
    # EfficientNet-B0 via timm: final conv block is model.backbone.conv_head
    # or we can use the last block in model.backbone.blocks
    try:
        # timm EfficientNet structure
        target_layer = model.backbone.conv_head
    except AttributeError:
        # Fallback: last block
        target_layer = model.backbone.blocks[-1]
    return target_layer


def generate_gradcam_visuals():
    """Generate Grad-CAM overlays for sample images per class."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(device)
    _, _, test_loader = get_dataloaders()

    os.makedirs(config.GRADCAM_DIR, exist_ok=True)

    # Collect samples per class (5 per class)
    class_samples = {i: [] for i in range(config.NUM_CLASSES)}
    target_count = 5

    for images, physics, labels, _ in test_loader:
        for i in range(len(labels)):
            cls = labels[i].item()
            if len(class_samples[cls]) < target_count:
                class_samples[cls].append((
                    images[i:i+1],
                    physics[i:i+1],
                    labels[i].item(),
                ))
        # Check if we have enough
        if all(len(v) >= target_count for v in class_samples.values()):
            break

    print(f"Generating Grad-CAM visuals...")

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        samples = class_samples[class_idx]
        print(f"\n  Class: {class_name} ({len(samples)} samples)")

        fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
        if len(samples) == 1:
            axes = axes.reshape(2, 1)

        for j, (img_tensor, phys_tensor, label) in enumerate(samples):
            img_tensor = img_tensor.to(device)
            phys_tensor = phys_tensor.to(device)

            # Create wrapper for Grad-CAM
            wrapper = RTIEModelWrapper(model, phys_tensor)
            target_layer = get_target_layer(model)

            cam = GradCAM(model=wrapper, target_layers=[target_layer])

            # Generate CAM
            targets = [ClassifierOutputTarget(class_idx)]
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0]  # first (only) image in batch

            # Prepare original image for overlay
            img_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_np = np.clip(img_np, 0, 1).astype(np.float32)

            # Create overlay
            overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            # Plot original
            axes[0, j].imshow(img_np)
            axes[0, j].set_title(f"Original", fontsize=10)
            axes[0, j].axis("off")

            # Plot Grad-CAM overlay
            axes[1, j].imshow(overlay)
            axes[1, j].set_title(f"Grad-CAM", fontsize=10)
            axes[1, j].axis("off")

        fig.suptitle(f"Grad-CAM — {class_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(config.GRADCAM_DIR, f"gradcam_{class_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"    Saved → {save_path}")

    print(f"\nAll Grad-CAM visuals saved to {config.GRADCAM_DIR}")


if __name__ == "__main__":
    generate_gradcam_visuals()
