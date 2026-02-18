"""
RTIE — Multi-Task Model Architecture

EfficientNet-B0 backbone (1280-dim) + Physics Feature Fusion (10-dim)
with Monte Carlo Dropout and uncertainty-based escalation.

Outputs:
    1. Fault classification (5 classes)
    2. Efficiency score (0-100, regression)
    3. Confidence score (0-1, calibrated)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

import config


class MCDropout(nn.Dropout):
    """Monte Carlo Dropout — stays active during inference."""
    def forward(self, x):
        return F.dropout(x, self.p, training=True)  # always on


class RTIEModel(nn.Module):
    """
    Radiator Thermal Intelligence Engine — Multi-Task Model

    Architecture:
        EfficientNet-B0 → 1280-dim → concat(physics 10-dim)
        → FC(1290→512) → ReLU → MCDropout
        → FC(512→256) → ReLU → MCDropout
        → 3 heads: classification, efficiency, confidence
    """

    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        physics_dim=config.NUM_PHYSICS_FEATURES,
        backbone_name=config.BACKBONE,
        dropout_rate=config.MC_DROPOUT_RATE,
        pretrained=True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone: EfficientNet-B0 ──
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0  # remove classifier
        )
        backbone_dim = config.BACKBONE_DIM  # 1280 for EfficientNet-B0

        # ── Fusion Layers ──
        fusion_dim = backbone_dim + physics_dim  # 1290
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.HIDDEN_DIM_1),
            nn.BatchNorm1d(config.HIDDEN_DIM_1),
            nn.ReLU(inplace=True),
            MCDropout(p=dropout_rate),
            nn.Linear(config.HIDDEN_DIM_1, config.HIDDEN_DIM_2),
            nn.BatchNorm1d(config.HIDDEN_DIM_2),
            nn.ReLU(inplace=True),
            MCDropout(p=dropout_rate),
        )

        # ── Head 1: Fault Classification ──
        self.classification_head = nn.Linear(config.HIDDEN_DIM_2, num_classes)

        # ── Head 2: Efficiency Regression ──
        self.efficiency_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM_2, 1),
            nn.Sigmoid(),  # output 0-1, scale to 0-100 later
        )

        # ── Head 3: Confidence Score ──
        self.confidence_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM_2, 1),
            nn.Sigmoid(),  # calibrated 0-1
        )

    def forward(self, images, physics_features):
        """
        Forward pass.

        Args:
            images: (B, 3, 224, 224) tensor
            physics_features: (B, 10) tensor

        Returns:
            dict with keys: logits, efficiency, confidence
        """
        # Extract backbone features
        backbone_feats = self.backbone(images)  # (B, 1280)

        # Fuse with physics features
        fused = torch.cat([backbone_feats, physics_features], dim=1)  # (B, 1290)
        fused = self.fusion(fused)  # (B, 256)

        # Multi-task outputs
        logits = self.classification_head(fused)           # (B, 5)
        efficiency = self.efficiency_head(fused) * 100     # (B, 1) scaled 0-100
        confidence = self.confidence_head(fused)           # (B, 1) 0-1

        return {
            "logits": logits,
            "efficiency": efficiency.squeeze(-1),
            "confidence": confidence.squeeze(-1),
        }

    @torch.no_grad()
    def predict_with_uncertainty(self, images, physics_features, T=config.MC_FORWARD_PASSES):
        """
        Monte Carlo Dropout inference — T stochastic forward passes.

        Returns:
            dict with mean predictions, uncertainty, and escalation status.
        """
        # Note: We do NOT call self.train() here because that would enable BatchNorm training,
        # which fails for batch_size=1. The custom MCDropout layer ensures dropout is valid
        # even in eval mode.

        all_logits = []
        all_efficiency = []
        all_confidence = []

        for _ in range(T):
            out = self.forward(images, physics_features)
            all_logits.append(F.softmax(out["logits"], dim=-1))
            all_efficiency.append(out["efficiency"])
            all_confidence.append(out["confidence"])

        # Stack: (T, B, ...)
        logits_stack = torch.stack(all_logits)       # (T, B, 5)
        eff_stack = torch.stack(all_efficiency)       # (T, B)
        conf_stack = torch.stack(all_confidence)      # (T, B)

        # Mean predictions
        mean_probs = logits_stack.mean(dim=0)         # (B, 5)
        mean_efficiency = eff_stack.mean(dim=0)       # (B,)
        mean_confidence = conf_stack.mean(dim=0)      # (B,)

        # Uncertainty: std dev of class probabilities
        uncertainty_std = logits_stack.std(dim=0).mean(dim=-1)  # (B,)

        # Predicted class
        predicted_class = mean_probs.argmax(dim=-1)   # (B,)

        # Escalation status
        needs_review = (
            (uncertainty_std > config.UNCERTAINTY_STD_THRESHOLD) |
            (mean_confidence < config.CONFIDENCE_THRESHOLD)
        )
        status = ["requires_manual_review" if r else "auto_approved" for r in needs_review]

        return {
            "predicted_class": predicted_class,
            "class_probs": mean_probs,
            "efficiency_score": mean_efficiency,
            "confidence": mean_confidence,
            "uncertainty_std": uncertainty_std,
            "status": status,
        }


if __name__ == "__main__":
    # Quick shape verification
    model = RTIEModel(pretrained=False)
    images = torch.randn(2, 3, 224, 224)
    physics = torch.randn(2, config.NUM_PHYSICS_FEATURES)

    out = model(images, physics)
    print("Forward pass shapes:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    # MC Dropout inference
    result = model.predict_with_uncertainty(images, physics, T=5)
    print("\nMC Dropout inference:")
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable:    {trainable:,}")
