"""
RTIE — Multi-Task Training Pipeline

Trains the RTIE model with:
    - CrossEntropyLoss (classification)
    - MSELoss (efficiency regression)
    - BCELoss (confidence calibration)
    - AdamW optimizer with cosine annealing
    - Early stopping on val loss
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import config
from model import RTIEModel
from dataset import get_dataloaders


def compute_confidence_target(logits, labels, smoothing=0.1):
    """
    Generate confidence targets: high confidence when prediction matches label.
    Uses label smoothing to prevent overconfident targets.
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float()
    # Smooth: correct=0.9, incorrect=0.1
    targets = correct * (1.0 - smoothing) + (1 - correct) * smoothing
    return targets


class RTIETrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Model
        self.model = RTIEModel(pretrained=True).to(self.device)

        # Loss functions
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.efficiency_loss_fn = nn.MSELoss()
        self.confidence_loss_fn = nn.BCELoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )

        # Data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            batch_size=config.BATCH_SIZE
        )

        # Logging
        self.history = {"train": [], "val": []}
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.REPORT_DIR, exist_ok=True)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        for images, physics, labels, efficiencies in pbar:
            images = images.to(self.device)
            physics = physics.to(self.device)
            labels = labels.to(self.device)
            efficiencies = efficiencies.to(self.device)

            # Forward
            outputs = self.model(images, physics)

            # Compute losses
            loss_class = self.class_loss_fn(outputs["logits"], labels)
            loss_eff = self.efficiency_loss_fn(outputs["efficiency"], efficiencies)

            # Confidence target: based on whether prediction is correct
            conf_targets = compute_confidence_target(outputs["logits"].detach(), labels)
            loss_conf = self.confidence_loss_fn(outputs["confidence"], conf_targets)

            total = (
                loss_class
                + config.LAMBDA_EFFICIENCY * loss_eff
                + config.LAMBDA_CONFIDENCE * loss_conf
            )

            # Backward
            self.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += total.item()
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{total.item():.4f}")

        # Epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return {"loss": avg_loss, "accuracy": acc, "f1": f1}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for images, physics, labels, efficiencies in self.val_loader:
            images = images.to(self.device)
            physics = physics.to(self.device)
            labels = labels.to(self.device)
            efficiencies = efficiencies.to(self.device)

            outputs = self.model(images, physics)

            loss_class = self.class_loss_fn(outputs["logits"], labels)
            loss_eff = self.efficiency_loss_fn(outputs["efficiency"], efficiencies)
            conf_targets = compute_confidence_target(outputs["logits"], labels)
            loss_conf = self.confidence_loss_fn(outputs["confidence"], conf_targets)

            total = (
                loss_class
                + config.LAMBDA_EFFICIENCY * loss_eff
                + config.LAMBDA_CONFIDENCE * loss_conf
            )
            total_loss += total.item()

            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

        return {
            "loss": avg_loss, "accuracy": acc, "f1": f1,
            "precision": precision, "recall": recall,
        }

    def train(self):
        start_epoch = 0
        if self.args.resume:
            checkpoint_path = os.path.join(config.MODEL_DIR, "best_model.pth")
            if os.path.exists(checkpoint_path):
                print(f"Resuming from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                self.best_val_f1 = checkpoint["val_f1"]
                print(f"Resumed from epoch {start_epoch} (Best Val F1: {self.best_val_f1:.4f})")
            else:
                print(f"Checkpoint not found at {checkpoint_path}, starting from scratch.")

        print(f"\n{'='*60}")
        print(f"RTIE Training — {self.args.epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch, self.args.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            lr = self.scheduler.get_last_lr()[0]
            print(
                f"  Train — Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}"
            )
            print(
                f"  Val   — Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | LR: {lr:.6f}"
            )

            # Checkpointing — best val F1
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.patience_counter = 0
                save_path = os.path.join(config.MODEL_DIR, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_f1": val_metrics["f1"],
                    "val_accuracy": val_metrics["accuracy"],
                }, save_path)
                print(f"  ★ New best model saved (F1={val_metrics['f1']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\n  Early stopping at epoch {epoch+1} (patience={config.EARLY_STOPPING_PATIENCE})")
                    break

            print()

        # Save training history
        history_path = os.path.join(config.REPORT_DIR, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete!")
        print(f"  Best Val F1: {self.best_val_f1:.4f}")
        print(f"  History: {history_path}")
        print(f"  Model: {os.path.join(config.MODEL_DIR, 'best_model.pth')}")

        return self.history


def main():
    parser = argparse.ArgumentParser(description="RTIE Training")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Training epochs")
    parser.add_argument("--subset", type=int, default=None, help="Use subset of data for quick test")
    parser.add_argument("--resume", action="store_true", help="Resume from best_model.pth")
    args = parser.parse_args()

    trainer = RTIETrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
