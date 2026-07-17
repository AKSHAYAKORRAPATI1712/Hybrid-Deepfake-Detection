# -*- coding: utf-8 -*-
"""
Optimized Deepfake Detector with EMA + Warmup + Evaluation Reports
- Stable version for Windows (no multiprocessing freeze)
- Added: post-training metrics and visualization (confusion matrix, classification report, training curves)
- Results saved under: training_reports/
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ===================== CONFIG =====================
REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"
CACHE_DIR = "processed_faces"
REPORT_DIR = "training_reports"

N_FRAMES = 16
BATCH_SIZE = 8
EPOCHS = 12
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_PATH = "checkpoint_latest.pth"
BEST_MODEL_PATH = "best_detector_ema.pth"

FORCE_RETRAIN = False
USE_EMA_AFTER = 2
USE_CUTMIX_AFTER = 2
USE_FOCAL_LOSS = True
BACKBONE = 'efficientnet-b3'

os.makedirs(os.path.join(CACHE_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "fake"), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ===================== Preprocessing =====================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

mtcnn = MTCNN(keep_all=False, device='cpu')  # safer for Windows


def extract_and_cache(video_path, save_path, n_frames=N_FRAMES):
    try:
        if os.path.exists(save_path):
            return torch.load(save_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.zeros(n_frames, 3, 224, 224)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return torch.zeros(n_frames, 3, 224, 224)
        idxs = np.linspace(0, total - 1, n_frames, dtype=int)
        frames, fid = [], 0
        while True:
            ret, f = cap.read()
            if not ret:
                break
            if fid in idxs:
                try:
                    f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    face = mtcnn(f_rgb)
                    if face is not None:
                        f = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    frames.append(transform(f))
                except Exception:
                    pass
            fid += 1
        cap.release()
        if len(frames) == 0:
            return torch.zeros(n_frames, 3, 224, 224)
        while len(frames) < n_frames:
            frames.append(frames[-1])
        tensor = torch.stack(frames[:n_frames])
        torch.save(tensor, save_path)
        return tensor
    except Exception:
        return torch.zeros(n_frames, 3, 224, 224)


# ===================== Dataset =====================
class CachedDeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, cache_dir):
        self.real_videos = list(Path(real_dir).glob("*.mp4"))
        self.fake_videos = list(Path(fake_dir).glob("*.mp4"))
        self.items = [(str(v), 0) for v in self.real_videos] + [(str(v), 1) for v in self.fake_videos]
        self.cache_dir = cache_dir
        print(f" Found {len(self.real_videos)} real, {len(self.fake_videos)} fake videos")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        subdir = "real" if label == 0 else "fake"
        cache_path = os.path.join(self.cache_dir, subdir, Path(path).stem + ".pt")
        frames = extract_and_cache(path, cache_path)
        return frames, torch.tensor(label, dtype=torch.float32)


# ===================== Model, EMA, Losses =====================
class DeepfakeDetector(nn.Module):
    def __init__(self, backbone_name=BACKBONE, pretrained=True, tconv_channels=512, dropout=0.1):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(backbone_name)
        self.feat_dim = self.backbone._fc.in_features
        self.temporal = nn.Sequential(
            nn.Conv1d(self.feat_dim, tconv_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(tconv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(tconv_channels, tconv_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(tconv_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(tconv_channels, 1))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat_map = self.backbone.extract_features(x)
        feat = F.adaptive_avg_pool2d(feat_map, 1).view(B, T, self.feat_dim)
        feat = feat.permute(0, 2, 1)
        t = self.temporal(feat)
        t = self.pool(t).view(B, -1)
        return self.classifier(t)


class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ===================== Training / Validation =====================
def train_epoch(model, loader, opt, crit, cutmix, scaler, ema, device, use_cutmix=False, use_ema=False):
    model.train()
    total_loss, correct = 0, 0
    for x, y in tqdm(loader, desc='Train', leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x).squeeze(1)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if use_ema:
            ema.update(model)
        total_loss += loss.item()
        preds = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == y).sum().item()
    return total_loss / len(loader), 100 * correct / len(loader.dataset)


def validate(model, loader, crit, device, collect_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Val', leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(1)
            loss = crit(out, y)
            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += x.size(0)
            if collect_preds:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / (total + 1e-12)
    acc = 100.0 * correct / (total + 1e-12)
    if collect_preds:
        return avg_loss, acc, np.array(all_preds), np.array(all_labels)
    return avg_loss, acc


# ===================== Main =====================
def main():
    # Toggle: also save the best base (non-EMA) model when validation improves
    SAVE_BEST_BASE = True
    BEST_BASE_PATH = "best_detector_base.pth"

    print(f" Using device: {DEVICE}")
    ds = CachedDeepfakeDataset(REAL_DIR, FAKE_DIR, CACHE_DIR)
    n = len(ds)
    if n == 0:
        print("No videos found in given directories. Exiting.")
        return

    # --- Split dataset (80/20) ---
    train_size = int(0.8 * n)
    val_size = n - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    # --- Build a WeightedRandomSampler for balanced training batches ---
    all_labels = [int(lbl.item()) for _, lbl in ds]
    train_indices = train_ds.indices if hasattr(train_ds, "indices") else list(range(train_size))
    train_labels = [all_labels[i] for i in train_indices]
    class_sample_count = np.array([train_labels.count(t) for t in [0, 1]])
    if class_sample_count.sum() == 0:
        print("Warning: no samples found in training subset.")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    else:
        weights_per_class = 1.0 / (class_sample_count + 1e-12)
        samples_weights = np.array([weights_per_class[t] for t in train_labels])
        sampler = torch.utils.data.WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f" Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Model, EMA, Loss, Optimizer, Scheduler, Scaler ---
    model = DeepfakeDetector().to(DEVICE)
    ema = ModelEMA(model)
    crit = FocalLoss() if USE_FOCAL_LOSS else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=2, factor=0.5)
    scaler = torch.cuda.amp.GradScaler()

    # --- training tracking ---
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_acc = 0.0

    # --- Training loop ---
    for e in range(1, EPOCHS + 1):
        print(f"\n Epoch {e}/{EPOCHS}")

        # Train (enable ema updates)
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, crit, None, scaler, ema, DEVICE, use_ema=True)

        # Validate (base model)
        val_loss, val_acc = validate(model, val_loader, crit, DEVICE)
        scheduler.step(val_acc)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {tr_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # ------------------ Save checkpoints ------------------
        ckpt = {
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.ema.state_dict() if hasattr(ema, "ema") else None,
            'optimizer_state_dict': opt.state_dict(),
            'val_acc': val_acc,
            'train_loss': tr_loss,
            'val_loss': val_loss
        }
        torch.save(ckpt, CHECKPOINT_PATH)
        print(f" Latest checkpoint saved to {CHECKPOINT_PATH}")

        # Save best models (EMA and optional base) when validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            # Save EMA model weights (recommended)
            if hasattr(ema, "ema"):
                torch.save(ema.ema.state_dict(), BEST_MODEL_PATH)
                print(f" New best EMA model saved (Val Acc: {best_acc:.2f}%) → {BEST_MODEL_PATH}")
            # Optionally save base (non-EMA) model weights
            if SAVE_BEST_BASE:
                torch.save(model.state_dict(), BEST_BASE_PATH)
                print(f" New best BASE model saved (Val Acc: {best_acc:.2f}%) → {BEST_BASE_PATH}")

    # ===================== Final Evaluation (load best model if available) =====================
    print("\n Generating final evaluation report using best model (EMA preferred)...")

    # Prefer EMA best for evaluation; fall back to saved base best if EMA not available; else in-memory ema; else base in-memory
    eval_model = None
    if os.path.exists(BEST_MODEL_PATH):
        eval_model = DeepfakeDetector().to(DEVICE)
        eval_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        eval_model.eval()
        print(f" Loaded best EMA model from {BEST_MODEL_PATH}")
    elif SAVE_BEST_BASE and os.path.exists(BEST_BASE_PATH):
        eval_model = DeepfakeDetector().to(DEVICE)
        eval_model.load_state_dict(torch.load(BEST_BASE_PATH, map_location=DEVICE))
        eval_model.eval()
        print(f" EMA not found — loaded best BASE model from {BEST_BASE_PATH}")
    else:
        try:
            eval_model = ema.ema
            print("Using in-memory EMA model for final evaluation.")
        except Exception:
            eval_model = model
            print("EMA not available, using base model for final evaluation.")

    # Collect preds & labels for final report
    val_loss, val_acc, preds, labels = validate(eval_model, val_loader, crit, DEVICE, collect_preds=True)

    # Convert predictions and labels to ints (0/1)
    preds = preds.astype(int)
    labels = labels.astype(int)

    # Print validation summary in desired format
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation Accuracy (%): {val_acc:.4f}\n")

    # Print counts to quickly see predicted distribution
    pred_counts = np.bincount(preds, minlength=2)
    true_counts = np.bincount(labels, minlength=2)
    print(f"Predicted counts - Real: {pred_counts[0]}, Fake: {pred_counts[1]}")
    print(f"True counts      - Real: {true_counts[0]}, Fake: {true_counts[1]}\n")

    # Classification report (zero_division=0 to avoid warnings when a class has no predictions)
    report = classification_report(labels, preds, target_names=["Real", "Fake"], digits=4, zero_division=0)
    print(report)

    # Save classification report
    report_path = os.path.join(REPORT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Validation Loss: {val_loss:.6f}\n")
        f.write(f"Validation Accuracy (%): {val_acc:.4f}\n\n")
        f.write(report)
    print(f" Classification report saved to {report_path}")

    # Confusion matrix and plot
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix (Best Model)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f" Confusion matrix saved to {cm_path}")

    # Training curves plot
    print("\n Saving training curves...")
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Training Acc")
    plt.plot(epochs, val_accs, label="Validation Acc")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    curves_path = os.path.join(REPORT_DIR, "training_curves.png")
    plt.savefig(curves_path)
    plt.close()
    print(f" Training curves saved to {curves_path}")

if _name_ == "__main__":
    main()

