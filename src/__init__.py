# =============================================================================
# Seismic Fault & Horizon Segmentation
# =============================================================================
# Deep learning pipeline for automated fault and horizon detection in 2D
# seismic sections using U-Net image segmentation.
#
# Dataset:       Netherlands F3 Block (North Sea)
# Architecture:  U-Net (encoder-decoder with skip connections)
# Output:        9-class pixel-wise segmentation
#
# Classes:
#   0 = Background       4 = FS6          8 = Top Foresets
#   1 = Fault            5 = FS7
#   2 = FS4              6 = FS8
#   3 = MFS4             7 = Shallow
#
# Primary execution target: Google Colab (T4 GPU, 15GB VRAM)
# =============================================================================

__version__ = "1.0.0"
__author__ = "Seismic Segmentation Project"

NUM_CLASSES = 9
CLASS_NAMES = [
    "Background", "Fault", "FS4", "MFS4", "FS6",
    "FS7", "FS8", "Shallow", "Top Foresets",
]
