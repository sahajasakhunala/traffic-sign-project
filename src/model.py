import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Squeeze-and-Excitation block ───────────────────────────────────────────────
# UPDATED: SE recalibrates each channel's importance after every conv block.
# On Indian roads, signs share colours with surroundings (red soil, green trees)
# so learning *which* channels matter per image is a meaningful accuracy boost.
class SEBlock(nn.Module):
    """Channel-wise attention: squeeze global info, excite informative channels."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ── Convolutional block ────────────────────────────────────────────────────────
# UPDATED: Double-conv pattern (like VGG mini-block) extracts richer features
# before downsampling. BatchNorm after each conv stabilises training on the
# noisy, uneven Indian dataset (mixed lighting, varying image quality).
class ConvBlock(nn.Module):
    """Two Conv→BN→ReLU layers followed by optional SE attention and MaxPool."""

    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        *,
        pool:    bool = True,
        use_se:  bool = True,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se   = SEBlock(out_ch, se_reduction) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.se(self.conv(x)))


# ── Main model ─────────────────────────────────────────────────────────────────
class TrafficSignCNN(nn.Module):
    """
    Improved CNN for Indian road sign recognition.

    Key upgrades over the baseline:
      • Double-conv blocks     — richer feature extraction per stage
      • BatchNorm everywhere   — stable training on noisy, small datasets
      • SE channel attention   — suppresses irrelevant channels (background colours)
      • Deeper classifier head — gradual bottleneck avoids abrupt info loss
      • BN1d in classifier     — stabilises the fully-connected stack
      • Lighter dropout (0.4 / 0.3 staged) — less destructive than flat 0.5

    Input:  (B, 3, H, W)  — H, W ≥ 32; trained at 64×64.
    Output: (B, num_classes) raw logits.
    """

    def __init__(self, num_classes: int = 46):
        super().__init__()

        # ── Feature extractor ──────────────────────────────────────────────────
        # Block 1: no SE (features too low-level to benefit from channel weighting)
        # Blocks 2-3: SE attention for channel recalibration
        # 64×64 → 32×32 → 16×16 → 8×8
        self.block1 = ConvBlock( 3,  32, pool=True, use_se=False)
        self.block2 = ConvBlock(32,  64, pool=True, use_se=True)
        self.block3 = ConvBlock(64, 128, pool=True, use_se=True)

        # ── Spatial pooling ────────────────────────────────────────────────────
        # UPDATED: (4, 4) instead of (8, 8) — reduces parameter count in the
        # first FC layer from 128*8*8=8192 to 128*4*4=2048, cutting overfit risk.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Classifier head ────────────────────────────────────────────────────
        # UPDATED: Two-layer bottleneck (2048→512→256) with BN1d and staged
        # dropout. BN1d is especially effective here as it normalises the
        # activations entering each linear layer, accelerating convergence.
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, num_classes),
        )

        # ── Weight initialisation ──────────────────────────────────────────────
        # UPDATED: Kaiming He init for conv/linear layers + zero BN bias.
        # Gives better gradient flow from epoch 1, especially important during
        # the LR warmup phase used in train.py.
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    model       = TrafficSignCNN(num_classes=46)
    dummy_input = torch.randn(4, 3, 64, 64)   # batch of 4
    output      = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Output shape  : {output.shape}")          # (4, 46)
    print(f"Total params  : {total_params:,}")
    print(f"Trainable     : {train_params:,}")

    # Verify gradient flow through SE blocks
    loss = output.sum()
    loss.backward()
    print("Backward pass : OK")