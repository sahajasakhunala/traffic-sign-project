import csv
import os
import time
import collections
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from .model import TrafficSignCNN

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "/content/drive/MyDrive/TrafficSign/Indian-Traffic Sign-Dataset/Images"
MODEL_DIR  = "/content/drive/MyDrive/TrafficSign/models"
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_sign_cnn.pth")
LOG_PATH   = os.path.join(MODEL_DIR, "training_log.csv")

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE        = 64
BATCH_SIZE        = 64
LR                = 1e-3            # Standard Adam LR — warmup ramps into it
EPOCHS            = 10              # Initial test run — bump to 40 after verifying paths/GPU
WARMUP_EPOCHS     = 3               # Linear warmup for stable early training
VAL_SPLIT         = 0.15
EARLY_STOP_PAT    = 10              # Patient — cosine LR improves late
GRAD_CLIP         = 2.0             # Moderate clip
NUM_WORKERS       = 0               # T4 Colab has enough CPU cores to feed GPU
SEED              = 42
LABEL_SMOOTHING   = 0.05            # Mild smoothing — just enough for noisy labels
MIXUP_ALPHA       = 0.2             # Gentle mixup — regularises without destroying signal
USE_WEIGHTED_SAMPLER = True         # Fix class-imbalance

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ── Transforms ─────────────────────────────────────────────────────────────────
# Balanced augmentation: enough variety for robustness without destroying
# the signal that the model needs to learn from.
#
# REMOVED vs previous version:
#   - GaussianBlur        → Was smearing small sign details (text, arrows)
#   - RandomGrayscale     → Colour IS a key feature for traffic signs
#   - RandomPerspective   → Redundant with RandomAffine; combined effect was too harsh
#   - RandomErasing       → Redundant with Mixup
#   - CutMix              → Redundant with Mixup; combined effect was over-regularizing
#
# KEPT / TUNED:
#   - Moderate ColorJitter → Handles lighting variation without extreme distortion
#   - RandomAffine         → Position/scale/shear variation for realistic viewing angles
#   - RandomCrop from oversize → Natural spatial jitter
#   - Very low HorizontalFlip → Most traffic signs are directional
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 8, IMAGE_SIZE + 8)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.05),           # Very rare — signs are directional
    transforms.RandomRotation(degrees=15),              # Moderate tilt
    transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.10),
        scale=(0.90, 1.10),
        shear=5,
    ),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ── Dataset helpers ────────────────────────────────────────────────────────────
class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies an independent transform."""

    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        image = Image.open(path).convert("RGB")
        return self.transform(image), label

    def get_labels(self) -> list[int]:
        """Return all labels — needed for WeightedRandomSampler."""
        return [self.subset.dataset.samples[i][1] for i in self.subset.indices]


def stratified_split(
    dataset:   datasets.ImageFolder,
    val_split: float,
    seed:      int,
) -> tuple[list[int], list[int]]:
    rng = torch.Generator().manual_seed(seed)
    class_indices: dict[int, list[int]] = collections.defaultdict(list)
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


def make_weighted_sampler(dataset: TransformSubset) -> WeightedRandomSampler:
    """Over-sample minority classes using inverse-frequency weights with sqrt damping."""
    labels          = dataset.get_labels()
    class_counts    = collections.Counter(labels)
    # sqrt damping: pure inverse-frequency is too aggressive for mildly imbalanced data
    class_weights   = {cls: 1.0 / math.sqrt(count) for cls, count in class_counts.items()}
    sample_weights  = [class_weights[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )
    return sampler


def build_loaders(
    data_dir: str,
    val_split: float,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names  = base_dataset.classes
    num_classes  = len(class_names)

    train_indices, val_indices = stratified_split(base_dataset, val_split, SEED)

    train_set = TransformSubset(Subset(base_dataset, train_indices), train_transform)
    val_set   = TransformSubset(Subset(base_dataset, val_indices),   val_transform)

    _loader_kwargs = dict(
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
        persistent_workers = False,
    )

    if USE_WEIGHTED_SAMPLER:
        sampler      = make_weighted_sampler(train_set)
        train_loader = DataLoader(train_set, sampler=sampler, **_loader_kwargs)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **_loader_kwargs)

    val_loader = DataLoader(val_set, shuffle=False, **_loader_kwargs)

    print(f"  Dataset   : {data_dir}")
    print(f"  Classes   : {num_classes}")
    print(f"  Train     : {len(train_indices):,}  ({100*(1-val_split):.0f}%)")
    print(f"  Val       : {len(val_indices):,}  ({100*val_split:.0f}%)")
    return train_loader, val_loader, num_classes, class_names


# ── Mixup helper ───────────────────────────────────────────────────────────────
# Only Mixup is used — CutMix was redundant and together they over-regularized.
# With alpha=0.2 the mixing is gentle: most of the time lam ≈ 0.8–1.0, meaning
# one image dominates and the other adds subtle noise.
def mixup_data(x, y, alpha: float, device: torch.device):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index      = torch.randperm(batch_size, device=device)
    mixed_x    = lam * x + (1 - lam) * x[index]
    y_a, y_b   = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── LR schedule ────────────────────────────────────────────────────────────────
# Linear warmup → cosine annealing to eta_min.
def get_lr(epoch: int, warmup_epochs: int, total_epochs: int, base_lr: float) -> float:
    eta_min = 1e-6
    if epoch <= warmup_epochs:
        # Linear warmup from eta_min to base_lr
        return eta_min + (base_lr - eta_min) * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * progress)) / 2


# ── Train / evaluate loops ─────────────────────────────────────────────────────
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device:    torch.device,
    epoch:     int = 0,
    use_mixup: bool = False,
) -> tuple[float, float]:
    """
    One full pass over `loader`.  Pass optimizer=None for eval mode.

    Key fix: training accuracy is computed on the CLEAN (un-mixed) predictions.
    Previous code compared mixed-input predictions against original labels,
    producing a misleadingly low training accuracy (~44%).
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

                if use_mixup and MIXUP_ALPHA > 0:
                    mixed_images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA, device)
                    outputs = model(mixed_images)
                    loss    = mixed_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                # ── Accuracy on CLEAN inputs ──────────────────────────────────
                # Run a quick forward on the original (un-mixed) images to get
                # a meaningful training accuracy metric.
                with torch.no_grad():
                    if use_mixup and MIXUP_ALPHA > 0:
                        clean_outputs = model(images)
                    else:
                        clean_outputs = outputs
                    _, predicted = clean_outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

            running_loss += loss.item() * images.size(0)
            total        += labels.size(0)

    return running_loss / total, 100.0 * correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 72)
    print("  Traffic Sign Recognition — Training (Indian Dataset)")
    print("=" * 72)
    print(f"  Device      : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Epochs      : {EPOCHS}  (warmup: {WARMUP_EPOCHS})")
    print(f"  LR          : {LR}  |  weight_decay : 1e-4")
    print(f"  Grad clip   : {GRAD_CLIP}  |  early-stop patience : {EARLY_STOP_PAT}")
    print(f"  Label smooth: {LABEL_SMOOTHING}  |  Mixup α: {MIXUP_ALPHA}")
    print(f"  Val split   : {VAL_SPLIT:.0%}  |  seed : {SEED}")
    print("-" * 72)

    train_loader, val_loader, num_classes, class_names = build_loaders(DATA_DIR, VAL_SPLIT)
    print("-" * 72)

    model     = TrafficSignCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # AdamW — decoupled weight decay works better than L2 reg in Adam
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc   = 0.0
    patience_count = 0
    history: list[dict] = []

    # Print parameter count for transparency
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters  : {total_params:,}  (trainable: {train_params:,})")
    print("-" * 72)

    header = (
        f"{'Epoch':>7}  {'T-Loss':>8}  {'T-Acc':>7}  "
        f"{'V-Loss':>8}  {'V-Acc':>7}  {'LR':>10}  {'Time':>7}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()

        # Set LR with warmup + cosine annealing
        current_lr = get_lr(epoch, WARMUP_EPOCHS, EPOCHS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Mixup only after warmup — let the model learn basic features first
        use_mixup = (epoch > WARMUP_EPOCHS)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, use_mixup=use_mixup,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device,
            epoch=epoch, use_mixup=False,
        )

        elapsed = time.perf_counter() - t0

        # ── Checkpoint ──────────────────────────────────────────────────────────
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "state_dict":  model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_acc":     val_acc,
                    "val_loss":    val_loss,
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "image_size":  IMAGE_SIZE,
                },
                MODEL_PATH,
            )
        else:
            patience_count += 1

        marker = " ✓" if improved else ""
        print(
            f"{epoch:>6}/{EPOCHS}"
            f"  {train_loss:>8.4f}"
            f"  {train_acc:>6.2f}%"
            f"  {val_loss:>8.4f}"
            f"  {val_acc:>6.2f}%"
            f"  {current_lr:>10.6f}"
            f"  {elapsed:>6.1f}s"
            f"{marker}"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    4),
            "lr":         round(current_lr, 8),
            "time_s":     round(elapsed,    2),
        })

        if patience_count >= EARLY_STOP_PAT:
            print(f"\n  Early stopping triggered — no improvement for {EARLY_STOP_PAT} epochs.")
            break

    # ── Save log ──────────────────────────────────────────────────────────────
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("=" * 72)
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Model saved       : {MODEL_PATH}")
    print(f"  Training log      : {LOG_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()