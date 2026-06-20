"""
train.py  —  Traffic Sign Recognition (Indian Dataset)
======================================================
EfficientNet-B0 fine-tuned with two-phase schedule.

Post-training analysis (runs automatically after training):
  - Confusion matrix  → saved as PNG + CSV
  - Per-class accuracy report  → printed + saved as CSV
  - Top-N confused pairs  → printed to help diagnose errors

These tell you whether remaining errors are fixable (data/augmentation)
or inherent (classes that genuinely look alike).
"""

import csv
import os
import time
import collections
import math
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive — safe in Colab / headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = "/content/drive/MyDrive/TrafficSign/Indian-Traffic Sign-Dataset/Images"
MODEL_DIR       = "/content/drive/MyDrive/TrafficSign/models"
MODEL_PATH      = os.path.join(MODEL_DIR, "efficientnet_b0_traffic_sign.pth")
LOG_PATH        = os.path.join(MODEL_DIR, "training_log.csv")
CONF_MAT_PNG    = os.path.join(MODEL_DIR, "confusion_matrix.png")
CONF_MAT_CSV    = os.path.join(MODEL_DIR, "confusion_matrix.csv")
PER_CLASS_CSV   = os.path.join(MODEL_DIR, "per_class_accuracy.csv")

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE           = 128
BATCH_SIZE           = 64       # fits T4 at 128x128; drop to 32 if you hit CUDA OOM
EPOCHS               = 60       # extended epochs for lower LR fine-tuning
PHASE1_EPOCHS        = 5        # head-only warm-up (backbone frozen)
LR_PHASE1            = 1e-3
LR_PHASE2            = 2e-5      # lower LR for delicate fine-tuning
WEIGHT_DECAY         = 1e-4
VAL_SPLIT            = 0.15
EARLY_STOP_PAT       = 15       # increased patience for slow improvements
GRAD_CLIP            = 2.0
NUM_WORKERS          = 2
SEED                 = 42
LABEL_SMOOTHING      = 0.05     # proven value from original run; 0.1 was unjustified
MIXUP_ALPHA          = 0.2      # EfficientNet pretrained features are strong; 0.4 fights them
USE_WEIGHTED_SAMPLER = True
COSINE_T0            = 10
COSINE_T_MULT        = 2
TOP_CONFUSED_N       = 15       # print this many worst confused pairs

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ── Transforms ─────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 12, IMAGE_SIZE + 12)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.05),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.88, 1.12), shear=6),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.06),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ── Dataset helpers ────────────────────────────────────────────────────────────
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        return self.transform(Image.open(path).convert("RGB")), label

    def get_labels(self):
        return [self.subset.dataset.samples[i][1] for i in self.subset.indices]


def stratified_split(dataset, val_split, seed):
    rng = torch.Generator().manual_seed(seed)
    class_indices = collections.defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label in sorted(class_indices):
        idxs    = class_indices[label]
        perm    = torch.randperm(len(idxs), generator=rng).tolist()
        idxs    = [idxs[i] for i in perm]
        n_val_c = max(1, int(len(idxs) * val_split))
        val_indices.extend(idxs[:n_val_c])
        train_indices.extend(idxs[n_val_c:])
    return train_indices, val_indices


def make_weighted_sampler(dataset):
    labels        = dataset.get_labels()
    class_counts  = collections.Counter(labels)
    
    # 0.75 exponent gives stronger weight to rare classes than sqrt (0.50)
    class_weights = {c: 1.0 / (n ** 0.75) for c, n in class_counts.items()}
    
    # Apply specific boost multipliers to specified hard class folders
    try:
        class_names = dataset.subset.dataset.classes
        boost_map = {
            "49": 6.0,  # Level crossing countdown marker (rarest: ~58 images)
            "47": 3.0,  # Level crossing countdown marker (~144 images)
            "48": 3.0,  # Level crossing countdown marker (~168 images)
            "50": 3.0,  # Level crossing countdown marker (~164 images)
            "23": 2.5,  # Turn left
            "24": 2.5,  # Turn right
            "36": 2.5,  # Side road junction
            "37": 2.5,  # Side road junction
            "42": 2.5,  # Staggered side road junction
            "43": 2.5,  # Staggered side road junction
            "52": 2.0,  # Bus stop
        }
        for folder_name, multiplier in boost_map.items():
            if folder_name in class_names:
                idx = class_names.index(folder_name)
                if idx in class_weights:
                    class_weights[idx] *= multiplier
    except Exception as e:
        print(f"Warning: Could not apply sampler class boosts ({e})")
        
    sample_w      = [class_weights[l] for l in labels]
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


def build_loaders(data_dir, val_split):
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names  = base_dataset.classes
    num_classes  = len(class_names)

    train_indices, val_indices = stratified_split(base_dataset, val_split, SEED)

    train_set = TransformSubset(Subset(base_dataset, train_indices), train_transform)
    val_set   = TransformSubset(Subset(base_dataset, val_indices),   val_transform)

    kw = dict(
        batch_size         = BATCH_SIZE,
        num_workers        = NUM_WORKERS,
        pin_memory         = (device.type == "cuda"),
        persistent_workers = (NUM_WORKERS > 0),
    )
    if USE_WEIGHTED_SAMPLER:
        train_loader = DataLoader(train_set, sampler=make_weighted_sampler(train_set), **kw)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **kw)

    val_loader = DataLoader(val_set, shuffle=False, **kw)

    print(f"  Dataset   : {data_dir}")
    print(f"  Classes   : {num_classes}")
    print(f"  Train     : {len(train_indices):,}  ({100*(1-val_split):.0f}%)")
    print(f"  Val       : {len(val_indices):,}  ({100*val_split:.0f}%)")
    return train_loader, val_loader, num_classes, class_names


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def freeze_backbone(model):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier")


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# ── Mixup ──────────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha):
    lam   = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Numerically stable multiclass Focal Loss.
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(logp).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        logpt = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * logpt
        
        if self.label_smoothing > 0:
            smooth_loss = -logp.mean(dim=-1)
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ── AMP (Automatic Mixed Precision) ────────────────────────────────────────────
# 20–50% faster on T4, less VRAM, identical accuracy.
_use_amp = (device.type == "cuda")
_scaler  = torch.amp.GradScaler(enabled=_use_amp)


# ── Train / eval loop ──────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, use_mixup=False):
    training = optimizer is not None
    model.train() if training else model.eval()

    running_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=_use_amp):
                    if use_mixup and MIXUP_ALPHA > 0:
                        mx, ya, yb, lam = mixup_data(images, labels, MIXUP_ALPHA)
                        out  = model(mx)
                        loss = mixed_criterion(criterion, out, ya, yb, lam)
                    else:
                        out  = model(images)
                        loss = criterion(out, labels)
                _scaler.scale(loss).backward()
                _scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                _scaler.step(optimizer)
                _scaler.update()
                with torch.no_grad():
                    clean = model(images) if (use_mixup and MIXUP_ALPHA > 0) else out
                    correct += clean.max(1)[1].eq(labels).sum().item()
            else:
                with torch.amp.autocast(device_type=device.type, enabled=_use_amp):
                    out  = model(images)
                    loss = criterion(out, labels)
                correct += out.max(1)[1].eq(labels).sum().item()

            running_loss += loss.item() * images.size(0)
            total        += labels.size(0)

    return running_loss / total, 100.0 * correct / total


# ── Confusion matrix + per-class analysis ─────────────────────────────────────
def collect_predictions(model, loader):
    """Return (all_labels, all_preds) numpy arrays over the full loader."""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            preds  = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def compute_confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Saves a normalised confusion matrix PNG.
    For large class counts (>20) we omit axis tick labels to keep it readable —
    the CSV always has the full detail.
    """
    num_classes = len(class_names)
    cm_norm     = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig_side = max(10, num_classes * 0.4)
    fig, ax  = plt.subplots(figsize=(fig_side, fig_side))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix (normalised by true class)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)

    if num_classes <= 30:
        ticks = np.arange(num_classes)
        ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels(class_names, fontsize=7)
        # Annotate cells only for small matrices
        if num_classes <= 20:
            thresh = cm_norm.max() / 2.0
            for i, j in itertools.product(range(num_classes), range(num_classes)):
                val = cm_norm[i, j]
                if val > 0.01:          # skip near-zero cells to reduce clutter
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if val > thresh else "black")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Predicted label (tick labels omitted — see CSV)", fontsize=10)
        ax.set_ylabel("True label (tick labels omitted — see CSV)",      fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix PNG : {save_path}")


def save_confusion_matrix_csv(cm, class_names, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ pred"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())
    print(f"  Confusion matrix CSV : {save_path}")


def per_class_report(cm, class_names, save_path):
    """
    Compute and print per-class precision, recall, F1, support.
    Saves a CSV sorted by recall ascending (worst classes first).
    """
    num_classes = len(class_names)
    rows = []
    for i in range(num_classes):
        tp      = cm[i, i]
        fn      = cm[i, :].sum() - tp          # missed in true row
        fp      = cm[:, i].sum() - tp          # false predictions in pred col
        support = cm[i, :].sum()

        recall    = tp / (tp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        rows.append({
            "class":     class_names[i],
            "support":   int(support),
            "recall_%":  round(100 * recall,    2),
            "precision_%": round(100 * precision, 2),
            "f1_%":      round(100 * f1,        2),
            "tp":        int(tp),
            "fp":        int(fp),
            "fn":        int(fn),
        })

    # Sort worst-first by recall
    rows_sorted = sorted(rows, key=lambda r: r["recall_%"])

    # Print summary table
    print()
    print("  ── Per-class accuracy (sorted worst → best recall) ──")
    hdr = f"  {'Class':<35} {'Support':>8} {'Recall':>8} {'Prec':>8} {'F1':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows_sorted:
        flag = "  ◄ POOR" if r["recall_%"] < 70 else ""
        print(
            f"  {r['class']:<35} {r['support']:>8} "
            f"{r['recall_%']:>7.1f}% {r['precision_%']:>7.1f}% {r['f1_%']:>7.1f}%"
            f"{flag}"
        )

    # Save CSV
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"\n  Per-class CSV        : {save_path}")

    return rows_sorted


def top_confused_pairs(cm, class_names, n=15):
    """
    Print the N most-confused off-diagonal (true→predicted) pairs.
    These are the pairs most worth investigating (more data, relabelling, etc.)
    """
    num_classes = len(class_names)
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)

    print()
    print(f"  ── Top-{n} confused pairs (true → predicted, by count) ──")
    print(f"  {'Count':>6}   {'True class':<35} → {'Predicted class'}")
    print("  " + "-" * 70)
    for count, true_cls, pred_cls in pairs[:n]:
        print(f"  {count:>6}   {true_cls:<35}   {pred_cls}")


def run_analysis(model, val_loader, class_names):
    """Full post-training analysis: confusion matrix + per-class report."""
    print()
    print("=" * 72)
    print("  Post-Training Analysis")
    print("=" * 72)
    print("  Collecting predictions on validation set…")

    labels, preds = collect_predictions(model, val_loader)
    num_classes   = len(class_names)
    cm            = compute_confusion_matrix(labels, preds, num_classes)

    # Overall accuracy (sanity check)
    overall = 100.0 * (labels == preds).mean()
    print(f"  Overall val accuracy  : {overall:.2f}%  (from raw predictions, no label smoothing)")

    # Plots + CSVs
    plot_confusion_matrix(cm, class_names, CONF_MAT_PNG)
    save_confusion_matrix_csv(cm, class_names, CONF_MAT_CSV)

    # Per-class report
    per_class_rows = per_class_report(cm, class_names, PER_CLASS_CSV)

    # Top confused pairs — the most actionable diagnostic
    top_confused_pairs(cm, class_names, n=TOP_CONFUSED_N)

    # Summary: how many classes are struggling?
    poor = [r for r in per_class_rows if r["recall_%"] < 70]
    ok   = [r for r in per_class_rows if 70 <= r["recall_%"] < 90]
    good = [r for r in per_class_rows if r["recall_%"] >= 90]

    print()
    print("  ── Class health summary ──")
    print(f"  ≥ 90% recall  (good)  : {len(good):>3} classes")
    print(f"  70–90% recall (ok)    : {len(ok):>3} classes")
    print(f"  < 70% recall  (poor)  : {len(poor):>3} classes  ◄ focus here")

    if poor:
        print()
        print("  Classes with < 70% recall — likely causes:")
        print("  1. Too few training samples  → collect more images")
        print("  2. Visual ambiguity with another class  → see confused pairs above")
        print("  3. Labelling errors in the dataset  → audit those folders")

    print("=" * 72)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def _save_checkpoint(model, optimizer, epoch, val_acc, val_loss, class_names, num_classes, phase="full"):
    torch.save(
        {
            "epoch":       epoch,
            "phase":       phase,        # "head" or "full" — needed for resume
            "state_dict":  model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_acc":     val_acc,
            "val_loss":    val_loss,
            "class_names": class_names,
            "num_classes": num_classes,
            "image_size":  IMAGE_SIZE,
            "backbone":    "efficientnet_b0",
        },
        MODEL_PATH,
    )


def _make_row(epoch, tl, ta, vl, va, lr, t, phase):
    return {
        "epoch": epoch, "phase": phase,
        "train_loss": round(tl, 6), "train_acc": round(ta, 4),
        "val_loss":   round(vl, 6), "val_acc":   round(va, 4),
        "lr":         round(lr, 8), "time_s":    round(t,  2),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 72)
    print("  Traffic Sign Recognition — EfficientNet-B0 Fine-Tuning")
    print("=" * 72)
    print(f"  Device      : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Image size  : {IMAGE_SIZE}px  |  Batch: {BATCH_SIZE}")
    print(f"  Epochs      : {EPOCHS}  (Phase 1 head-only: {PHASE1_EPOCHS})")
    print(f"  LR P1/P2    : {LR_PHASE1} / {LR_PHASE2}  |  WD: {WEIGHT_DECAY}")
    print(f"  Label smooth: {LABEL_SMOOTHING}  |  Mixup α: {MIXUP_ALPHA}")
    print(f"  Val split   : {VAL_SPLIT:.0%}  |  seed: {SEED}")
    print("-" * 72)

    train_loader, val_loader, num_classes, class_names = build_loaders(DATA_DIR, VAL_SPLIT)
    print("-" * 72)

    model     = build_model(num_classes).to(device)
    criterion = FocalLoss(gamma=2.0, label_smoothing=LABEL_SMOOTHING)

    print(f"  Backbone    : EfficientNet-B0 (ImageNet pretrained)")
    print(f"  Parameters  : {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 72)

    best_val_acc, patience_count = 0.0, 0
    history: list[dict] = []

    # ── Resume from checkpoint if available ───────────────────────────────────
    # The checkpoint stores the epoch and phase it was saved at, so we can
    # skip already-completed epochs and restore optimiser state exactly.
    resume_epoch = 0          # last completed epoch (0 = fresh start)
    resume_phase = "head"     # which phase we were in when saved
    if os.path.exists(MODEL_PATH):
        print(f"\n[!] Checkpoint found: {MODEL_PATH}")
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            best_val_acc  = ckpt.get("val_acc",  0.0)
            resume_epoch  = ckpt.get("epoch",    0)
            resume_phase  = ckpt.get("phase",    "head")
            print(f"    Resumed from epoch {resume_epoch}  "
                  f"phase={resume_phase}  best_val_acc={best_val_acc:.2f}%")
            # Reload history from log so the CSV stays continuous
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        history.append({
                            "epoch":      int(row["epoch"]),
                            "phase":      row["phase"],
                            "train_loss": float(row["train_loss"]),
                            "train_acc":  float(row["train_acc"]),
                            "val_loss":   float(row["val_loss"]),
                            "val_acc":    float(row["val_acc"]),
                            "lr":         float(row["lr"]),
                            "time_s":     float(row["time_s"]),
                        })
        except Exception as e:
            print(f"    Could not load checkpoint ({e}) — starting from scratch.")
            resume_epoch = 0
            resume_phase = "head"

    header = (f"{'Epoch':>7}  {'T-Loss':>8}  {'T-Acc':>7}  "
              f"{'V-Loss':>8}  {'V-Acc':>7}  {'LR':>10}  {'Time':>7}  {'Phase':>7}")
    sep = "-" * len(header)

    # ── Phase 1: head only ────────────────────────────────────────────────────
    # Skip entirely if we already finished Phase 1 before the interruption.
    phase1_start = resume_epoch + 1 if resume_phase == "head" else PHASE1_EPOCHS + 1

    print(f"\n  ── Phase 1: head-only ({PHASE1_EPOCHS} epochs) ──")
    freeze_backbone(model)
    print(f"  Trainable : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}  (backbone frozen)")

    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)
    sch1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt1, T_0=PHASE1_EPOCHS, T_mult=1, eta_min=1e-6)

    # Fast-forward scheduler to match the resumed epoch so LR is correct
    for _e in range(1, phase1_start):
        sch1.step(_e)

    if phase1_start > PHASE1_EPOCHS:
        print(f"  Phase 1 already complete — skipping.")
    else:
        print(f"  Resuming from epoch {phase1_start}.")
        print(header); print(sep)

    for epoch in range(phase1_start, PHASE1_EPOCHS + 1):
        t0 = time.perf_counter()
        tl, ta = run_epoch(model, train_loader, criterion, opt1,  use_mixup=False)
        vl, va = run_epoch(model, val_loader,   criterion, None,  use_mixup=False)
        sch1.step(epoch)
        elapsed = time.perf_counter() - t0
        lr      = sch1.get_last_lr()[0]

        if va > best_val_acc:
            best_val_acc, patience_count = va, 0
            _save_checkpoint(model, opt1, epoch, va, vl, class_names, num_classes, phase="head")
        else:
            patience_count += 1

        mark = " ✓" if va == best_val_acc else ""
        print(f"{epoch:>6}/{EPOCHS}  {tl:>8.4f}  {ta:>6.2f}%  {vl:>8.4f}  {va:>6.2f}%  {lr:>10.6f}  {elapsed:>6.1f}s  {'head':>7}{mark}")
        history.append(_make_row(epoch, tl, ta, vl, va, lr, elapsed, "head"))

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    # If we crashed mid-phase-2, resume_phase=="full" and we skip ahead.
    phase2_start = (resume_epoch + 1
                    if resume_phase == "full"
                    else PHASE1_EPOCHS + 1)

    print(f"\n  ── Phase 2: full fine-tuning (epochs {PHASE1_EPOCHS+1}–{EPOCHS}) ──")
    unfreeze_all(model)
    print(f"  Trainable : {sum(p.numel() for p in model.parameters()):,}  (all layers)")

    opt2 = optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
    sch2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=COSINE_T0, T_mult=COSINE_T_MULT, eta_min=1e-7)

    # Fast-forward scheduler
    for _e in range(1, phase2_start - PHASE1_EPOCHS):
        sch2.step(_e)

    patience_count = 0
    if phase2_start <= EPOCHS:
        print(f"  Resuming from epoch {phase2_start}.")
    print(header); print(sep)

    for epoch in range(phase2_start, EPOCHS + 1):
        t0 = time.perf_counter()
        tl, ta = run_epoch(model, train_loader, criterion, opt2, use_mixup=True)
        vl, va = run_epoch(model, val_loader,   criterion, None, use_mixup=False)
        sch2.step(epoch - PHASE1_EPOCHS)
        elapsed = time.perf_counter() - t0
        lr      = sch2.get_last_lr()[0]

        if va > best_val_acc:
            best_val_acc, patience_count = va, 0
            _save_checkpoint(model, opt2, epoch, va, vl, class_names, num_classes, phase="full")
        else:
            patience_count += 1

        mark = " ✓" if va == best_val_acc else ""
        print(f"{epoch:>6}/{EPOCHS}  {tl:>8.4f}  {ta:>6.2f}%  {vl:>8.4f}  {va:>6.2f}%  {lr:>10.6f}  {elapsed:>6.1f}s  {'full':>7}{mark}")
        history.append(_make_row(epoch, tl, ta, vl, va, lr, elapsed, "full"))

        if patience_count >= EARLY_STOP_PAT:
            print(f"\n  Early stopping — no improvement for {EARLY_STOP_PAT} epochs.")
            break

    # ── Save training log (full history including resumed rows) ──────────────
    if history:
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    print("=" * 72)
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Model saved       : {MODEL_PATH}")
    print(f"  Training log      : {LOG_PATH}")

    # ── Load best checkpoint for analysis ────────────────────────────────────
    print("\n  Loading best checkpoint for post-training analysis…")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    run_analysis(model, val_loader, class_names)


if __name__ == "__main__":
    main()