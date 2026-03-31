# 🌊 Seismic Segmentation: Fault & Horizon Identification 🌎

A professional deep learning pipeline for automated seismic interpretation. This project implements a high-performance **U-Net** architecture to segment both stratigraphic horizons and structural faults simultaneously from 2D seismic sections and 3D volumes.

---

## 🚀 Key Features

*   **9-Class Segmentation**: Identifies Background, Faults, and 7 distinct geological horizons (FS4, MFS4, FS6, FS7, FS8, Shallow, Top Foresets).
*   **U-Net Architecture**: Multi-scale feature extraction with skip connections to preserve thin-line geological details.
*   **Hybrid Loss Function**: Combines **Weighted Cross-Entropy** (to address class imbalance) and **Dice Loss** (to optimize spatial overlap).
*   **Flexible Data Support**: Native support for pre-processed `.npy` slices and raw `.segy` seismic volumes.
*   **Performance**: Achieved **87.8% Mean Dice** on the Netherlands F3 Block benchmark.
*   **Optimized Inference**: Supports sliding-window prediction on arbitrarily sized seismic images.

---

## 🔬 Methodology: How It Works

The assistant treats seismic interpretation as a **semantic segmentation** problem. Unlike traditional methods that treat horizons and faults separately, this model is trained to identify both in a single forward pass:

1.  **Dual-Task Learning**: The model identifies 7 continuous **horizons** (rock layers) and **structural faults** (vertical cracks) simultaneously. 
2.  **U-Net Architecture**: We use a symmetric encoder-decoder network. The **encoder** extracts deep geological patterns, while the **decoder** projects them back to pixel-level coordinates. **Skip connections** ensure that thin features (like a 1-pixel wide fault) are not lost during processing.
3.  **Countering Imbalance**: In seismic data, background pixels dominate >90% of the image. We use a **Composite Loss (Weighted CE + Dice)** to penalize the model heavily if it misses a rare fault or a precise horizon line.
4.  **Sliding Window Inference**: For 3D volumes or large 2D sections, the model processes the data in overlapping 256×256 patches, which are then stitched back together to ensure seamless global interpretation.

---

## 🖼️ Example Prediction (Inline 0651)

The following compares expert geological labeling with the full-volume AI prediction:

| Ground Truth (Geologist Labels) | U-Net Prediction |
| :---: | :---: |
| <img src="assets/inline_0651_gt.png" width="400"/> | <img src="assets/inline_0651_pred.png" width="400"/> |

---

## 📊 Dataset: Netherlands F3 Block

The model is trained on the **Netherlands Offshore F3 Block** dataset (North Sea).
- **Scale**: 3D seismic volume (approx. 1.5 GB).
- **Labels**: Expert-annotated masks for 8 geological classes.
- **Challenges**: Contains complex salt domes and thinning horizons, providing a rigorous test for AI performance.

> **📥 Download Dataset:** Due to large file sizes, the data is hosted externally on Google Drive. 
> 
> **👉 [Download the Dataset Here](https://drive.google.com/drive/folders/1e7wqu7mvjAWOI8OWH8weBftrtohabysl?usp=sharing)**
> 
> The Drive contains two folders:
> 1. `data/`: Contains the processed slices and generated masks needed to train or run predictions. **Place this folder in your project root before running the code.**
> 2. `F3_Demo_2023/`: Contains the complete, raw original dataset downloaded from the official site (for reference or manual reprocessing).
> 
> *(Note: The pre-trained model `best_model.pth` is already included directly within this repository in the `checkpoints/` folder).*

*   **Framework**: [PyTorch 2.0+](https://pytorch.org/) (with Mixed Precision / AMP)
*   **Numerical Processing**: [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/)
*   **Visualization**: [Matplotlib](https://matplotlib.org/)
*   **Seismic IO**: [segyio](https://github.com/equinor/segyio)
*   **Training Infrastructure**: Optimized for Google Colab T4 (16GB VRAM) and local consumer GPUs.

---

## 📂 Project Structure

```text
Fault_detection/
├── 📁 src/               # Core engine: dataset, model, loss, and training loops
├── 📁 notebooks/         # Interactive research and visualization
├── 📁 checkpoints/       # Pre-trained model weights (best_model.pth)
├── 📁 data/              # Dataset directory (slices/ and masks/)
├── 📁 outputs/           # Prediction results and performance metrics
├── 📄 requirements.txt    # Production dependencies
└── 📄 README.md          # Project documentation
```

---

## 📖 Usage Guide

### 1. Installation
Ensure you have Python 3.8+ installed, then set up the environment:
```bash
pip install -r requirements.txt
```

### 2. Single-Image Prediction
Run the trained model on any seismic `.npy` or `.segy` file to generate a colored overlay:
```bash
# Predict on a pre-processed .npy slice
python -m src.predict --input data/slices/inline_0138_train.npy --save_mask

# Predict on a raw SEG-Y volume (processes first 5 inlines)
python -m src.predict --input path/to/seismic.sgy --max_inlines 5
```

### 3. Model Evaluation
Compute quantitative metrics (Dice, IoU) on the test set:
```bash
python -m src.evaluate --data_dir data/ --checkpoint checkpoints/best_model.pth
```

### 4. Training
Initialize a new training run from scratch or continue from a checkpoint:
```bash
python src/train.py --data_dir data/ --epochs 50 --batch_size 16
```

---

## 📊 Dataset Attribution
This project utilizes the **Netherlands Offshore F3 Block** dataset, a public seismic volume from the North Sea. We acknowledge the contribution of the geological research community in providing this essential resource for machine learning development.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information (if applicable).
