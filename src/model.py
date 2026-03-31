"""
model.py — U-Net Architecture for Seismic Segmentation
=======================================================
Implements a standard encoder–decoder U-Net with skip connections,
optimized for 9-class pixel-wise classification of seismic sections.

Architecture Overview:
    ┌──────────────────────────────────────────────────────────────┐
    │  Input (1, 256, 256)                                         │
    │   ↓                                                          │
    │  Encoder:  64 → 128 → 256 → 512  (4 downsampling blocks)    │
    │   ↓                                                          │
    │  Bottleneck: 1024 channels at 16×16 resolution               │
    │   ↓                                                          │
    │  Decoder:  512 → 256 → 128 → 64  (4 upsampling blocks)      │
    │   ↓                                                          │
    │  Output Head: 1×1 conv → 9 classes at 256×256                │
    └──────────────────────────────────────────────────────────────┘

Each encoder/decoder block:
    Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU

Skip connections concatenate encoder features with decoder features
to preserve fine spatial detail (critical for thin fault/horizon lines).

Usage:
    model = UNet(in_channels=1, num_classes=9)
    output = model(images)  # (B, 9, 256, 256)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Double convolution block: the fundamental building unit of U-Net.

        Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU

    Uses padding=1 to preserve spatial dimensions through the block.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder stage: ConvBlock followed by 2×2 MaxPool downsampling.

    Returns both the conv features (for skip connection) and the
    downsampled output (for the next encoder stage).

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels (doubles at each encoder stage).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns
        -------
        features : torch.Tensor
            Full-resolution features (saved for skip connection).
        downsampled : torch.Tensor
            2× downsampled output for the next stage.
        """
        features = self.conv(x)
        downsampled = self.pool(features)
        return features, downsampled


class DecoderBlock(nn.Module):
    """
    Decoder stage: Upsample → Concatenate skip → ConvBlock.

    Uses transposed convolution for learned upsampling (2× spatial).

    Parameters
    ----------
    in_ch : int
        Input channels (from previous decoder stage).
    skip_ch : int
        Channels from the corresponding encoder skip connection.
    out_ch : int
        Output channels.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_ch, in_ch // 2, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Low-resolution input from previous decoder/bottleneck.
        skip : torch.Tensor
            High-resolution features from corresponding encoder stage.

        Returns
        -------
        torch.Tensor
            Decoded features at 2× the input resolution.
        """
        x = self.upsample(x)

        # Handle potential size mismatch from odd-dimensioned inputs
        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = nn.functional.pad(
                x, [diff_w // 2, diff_w - diff_w // 2,
                     diff_h // 2, diff_h - diff_h // 2]
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net Model
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net segmentation model for seismic interpretation.

    Architecture:
        Encoder:     1 → 64 → 128 → 256 → 512   (4 stages)
        Bottleneck:  512 → 1024                    (deepest level)
        Decoder:     1024 → 512 → 256 → 128 → 64  (4 stages)
        Head:        64 → num_classes              (1×1 convolution)

    The model processes 256×256 grayscale patches and produces a
    per-pixel classification map with `num_classes` channels.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale seismic).
    num_classes : int
        Number of output segmentation classes (default: 9).

    Example
    -------
    >>> model = UNet(in_channels=1, num_classes=9)
    >>> x = torch.randn(4, 1, 256, 256)
    >>> out = model(x)
    >>> out.shape
    torch.Size([4, 9, 256, 256])
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 9) -> None:
        super().__init__()

        # ── Encoder pathway ──
        self.enc1 = EncoderBlock(in_channels, 64)    # 256 → 128
        self.enc2 = EncoderBlock(64, 128)             # 128 → 64
        self.enc3 = EncoderBlock(128, 256)             # 64  → 32
        self.enc4 = EncoderBlock(256, 512)             # 32  → 16

        # ── Bottleneck ──
        self.bottleneck = ConvBlock(512, 1024)         # 16 × 16

        # ── Decoder pathway ──
        self.dec4 = DecoderBlock(1024, 512, 512)       # 16  → 32
        self.dec3 = DecoderBlock(512, 256, 256)        # 32  → 64
        self.dec2 = DecoderBlock(256, 128, 128)        # 64  → 128
        self.dec1 = DecoderBlock(128, 64, 64)          # 128 → 256

        # ── Classification head ──
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize convolutional weights with Kaiming (He) initialization
        and batch norm parameters with standard values.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes, H, W).
        """
        # Encoder — extract features at 4 resolutions
        skip1, x = self.enc1(x)   # skip1: (B,  64, 256, 256)
        skip2, x = self.enc2(x)   # skip2: (B, 128, 128, 128)
        skip3, x = self.enc3(x)   # skip3: (B, 256,  64,  64)
        skip4, x = self.enc4(x)   # skip4: (B, 512,  32,  32)

        # Bottleneck — deepest representation
        x = self.bottleneck(x)     # (B, 1024, 16, 16)

        # Decoder — upsample and fuse with encoder skip connections
        x = self.dec4(x, skip4)    # (B, 512,  32,  32)
        x = self.dec3(x, skip3)    # (B, 256,  64,  64)
        x = self.dec2(x, skip2)    # (B, 128, 128, 128)
        x = self.dec1(x, skip1)    # (B,  64, 256, 256)

        # Classification head
        logits = self.head(x)      # (B, 9, 256, 256)

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Returns
    -------
    int
        Total trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: UNet) -> None:
    """
    Print a concise summary of the U-Net model.

    Shows total parameters and estimated memory footprint for
    a single forward pass (useful for VRAM planning).
    """
    n_params = count_parameters(model)
    # Rough estimate: 4 bytes per param (float32)
    mem_mb = n_params * 4 / (1024 ** 2)

    print(f"\n{'═' * 50}")
    print(f"U-Net Model Summary")
    print(f"{'═' * 50}")
    print(f"  Input:       (1, 256, 256) grayscale")
    print(f"  Output:      (9, 256, 256) class logits")
    print(f"  Parameters:  {n_params:,}")
    print(f"  Weight size: {mem_mb:.1f} MB")
    print(f"{'═' * 50}\n")
