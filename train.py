import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import collections

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import TrafficSignCNN

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join("data", "GTSRB", "GTSRB_Training")
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_sign_cnn.pth")
LOG_PATH   = os.path.join(MODEL_DIR, "training_log.csv")

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE       = 64
BATCH_SIZE       = 64
LR               = 1e-3
EPOCHS           = 15
VAL_SPLIT        = 0.15          # 15 % of data held out for validation
EARLY_STOP_PAT   = 5             # stop if val acc doesn't improve for N epochs
GRAD_CLIP        = 2.0           # max gradient norm
NUM_WORKERS      = min(4, os.cpu_count() or 1)
SEED             = 42

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True   # faster convolutions once input size is fixed


# ── Transforms ─────────────────────────────────────────────────────────────────
# Augmentation only for training; validation gets deterministic preprocessing.
# NOTE: RandomHorizontalFlip intentionally excluded — flipping traffic signs
#       (e.g. Keep Right, Curve Left) produces semantically invalid images.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=15),
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
        mean=[0.3337, 0.3064, 0.3171],         # GTSRB channel statistics
        std=[0.2672, 0.2564, 0.2629],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.3337, 0.3064, 0.3171],
        std=[0.2672, 0.2564, 0.2629],
    ),
])


# ── Dataset helpers ────────────────────────────────────────────────────────────
class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies an independent transform — avoids data leakage
    that would occur if both splits shared the same augmented ImageFolder."""

    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # subset.dataset already applied its own transform; we need raw PIL images.
        # Load fresh from disk via the underlying ImageFolder.
        path, _ = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def stratified_split(
    dataset:   datasets.ImageFolder,
    val_split: float,
    seed:      int,
) -> tuple[list[int], list[int]]:
    """Per-class stratified split — preserves class-frequency ratios in both sets.

    GTSRB is highly imbalanced (class sizes range from ~180 to ~2250 samples).
    random_split would under-represent rare classes in validation by chance;
    stratified split guarantees each class contributes val_split% to val exactly.
    """
    rng = torch.Generator().manual_seed(seed)

    # Group indices by class label.
    class_indices: dict[int, list[int]] = collections.defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label in sorted(class_indices):
        idxs     = class_indices[label]
        # Shuffle within each class for unbiased selection.
        perm     = torch.randperm(len(idxs), generator=rng).tolist()
        idxs     = [idxs[i] for i in perm]
        n_val_c  = max(1, int(len(idxs) * val_split))   # at least 1 val sample/class
        val_indices.extend(idxs[:n_val_c])
        train_indices.extend(idxs[n_val_c:])

    return train_indices, val_indices


def build_loaders(
    data_dir: str,
    val_split: float,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    """Stratified train/val split with independent per-split transforms.

    Returns (train_loader, val_loader, num_classes, class_names).
    """
    # Load once without transform — TransformSubset applies per-split transforms.
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names  = base_dataset.classes   # saved into checkpoint so inference needs no hard-coded map
    num_classes  = len(class_names)

    train_indices, val_indices = stratified_split(base_dataset, val_split, SEED)

    train_set = TransformSubset(Subset(base_dataset, train_indices), train_transform)
    val_set   = TransformSubset(Subset(base_dataset, val_indices),   val_transform)

    _loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    train_loader = DataLoader(train_set, shuffle=True,  **_loader_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, **_loader_kwargs)

    print(f"  Dataset  : {data_dir}")
    print(f"  Classes  : {num_classes}")
    print(f"  Train    : {len(train_indices):,}  ({100*(1-val_split):.0f}%)")
    print(f"  Val      : {len(val_indices):,}  ({100*val_split:.0f}%)")
    return train_loader, val_loader, num_classes, class_names


# ── Train / evaluate loops ─────────────────────────────────────────────────────
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device:    torch.device,
) -> tuple[float, float]:
    """One full pass over `loader`.  Pass optimizer=None for eval mode."""
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

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 72)
    print("  Traffic Sign Recognition — Training")
    print("=" * 72)
    print(f"  Device      : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  LR          : {LR}  |  weight_decay : 1e-4")
    print(f"  Grad clip   : {GRAD_CLIP}  |  early-stop patience : {EARLY_STOP_PAT}")
    print(f"  Val split   : {VAL_SPLIT:.0%}  |  seed : {SEED}")
    print("-" * 72)

    train_loader, val_loader, num_classes, class_names = build_loaders(DATA_DIR, VAL_SPLIT)
    print("-" * 72)

    model     = TrafficSignCNN().to(device)
    # Label smoothing kept at 0.05 (not 0.1) — less aggressive for 43-class problem
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc    = 0.0
    patience_count  = 0
    history: list[dict] = []

    header = (
        f"{'Epoch':>7}  {'T-Loss':>8}  {'T-Acc':>7}  "
        f"{'V-Loss':>8}  {'V-Acc':>7}  {'LR':>10}  {'Time':>7}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device)

        elapsed    = time.perf_counter() - t0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ── Checkpoint ──────────────────────────────────────────────────────────
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(
                {
                    "epoch":        epoch,
                    "state_dict":   model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "val_acc":      val_acc,
                    "val_loss":     val_loss,
                    "class_names":  class_names,   # inference needs no hard-coded mapping
                    "num_classes":  num_classes,
                    "image_size":   IMAGE_SIZE,
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

        # ── CSV log ──────────────────────────────────────────────────────────────
        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    4),
            "lr":         round(current_lr, 8),
            "time_s":     round(elapsed,    2),
        })

        # ── Early stopping ───────────────────────────────────────────────────────
        if patience_count >= EARLY_STOP_PAT:
            print(f"\n  Early stopping triggered — no improvement for {EARLY_STOP_PAT} epochs.")
            break

    # ── Save log ─────────────────────────────────────────────────────────────────
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # ── Summary ──────────────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Model saved       : {MODEL_PATH}")
    print(f"  Training log      : {LOG_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
