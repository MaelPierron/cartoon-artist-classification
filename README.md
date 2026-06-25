# Cartoon Artist Classification

**Deep learning vs machine learning for illustration style recognition — classifying cartoon images by artist.**


## Overview

Can a model tell which artist drew a cartoon panel? This project compares a **fine-tuned ResNet-18** against classical machine learning approaches (**SVM with HOG/LBP/color features, with and without PLS dimensionality reduction**) on a 5-class cartoon artist classification task. Visualization includes PCA and t-SNE projections of the learned feature space.


## Dataset

- **5 classes**: Ghibli, One Piece, Project Moon, Steins;Gate, Tower of God (zero-shot test class)
- **~300 images per class**, ~1500 total
- Dataset not publicly versioned — available locally


## Approach

Two pipelines compared:

| Pipeline | Method | Details |
|----------|--------|---------|
| **Deep Learning** | ResNet-18 (fine-tuned) | Pretrained on ImageNet, last FC layer replaced, trained to convergence, Adam lr=1e-4, input size 384×384 |
| **Classical ML** | SVM (RBF kernel, C=10) | Features: HOG, LBP, RGB color histograms → SelectKBest (top 15) |
| **Classical ML + dim. reduction** | SVM + PLS | Same features, dimensionality reduced via Partial Least Squares |


## Results

*Accuracy values to be restored from saved outputs.*

| Model | Accuracy | Notes |
|-------|----------|-------|
| ResNet-18 (fine-tuned) | — | Best performer |
| SVM (RBF) | — | — |
| SVM + PLS | — | — |

**Confusion matrices:**

| ResNet-18 | SVM | SVM + PLS |
|-----------|-----|-----------|
| ![](path/to/resnet_confusion.png) | ![](path/to/svm_confusion.png) | ![](path/to/svm_pls_confusion.png) |

**Feature space visualization:**

| t-SNE 2D | PCA 2D |
|----------|--------|
| ![](path/to/tsne.png) | ![](path/to/pca.png) |


## Installation

```bash
git clone https://github.com/MaelPierron/cartoon-artist-classification.git
cd cartoon-artist-classification
pip install -r requirements.txt
```

Dataset must be placed in `dataset/train/` and `dataset/val/` with class subfolders. The zero-shot test class goes in `dataset5/train/`.

## Usage

1. **Train ResNet-18**: `python ia.py`
2. **Extract features for ML**: `python extract_feature5.py`
3. **Train SVM**: `python svm.py`
4. **Train SVM + PLS**: `python svm_pls.py`
5. **Visualize feature space**: `python read_csv_tnse.py` (t-SNE 2D), `python read_csv_acp_3d.py` (PCA)


## Repository Structure

```
├── ia.py                  # ResNet-18 training & evaluation
├── svm.py                 # SVM classification
├── svm_pls.py             # SVM + PLS classification
├── extract_feature5.py    # HOG + LBP + color histogram extraction
├── detection_features.py  # Additional feature extraction
├── crop.py                # Image preprocessing / cropping
├── read_csv_tnse.py       # t-SNE 2D visualization
├── read_csv_acp_3d.py     # PCA 3D visualization
├── tsne_3d.py             # t-SNE 3D visualization
├── read_csv.py            # CSV data loading utilities
├── test_ia.py             # Model testing
├── test_generalisation.py # Generalization evaluation
├── test.jpg               # Sample test image
└── requirements.txt
```


## Limitations

- Dataset size (~300/class) limits generalization
- Zero-shot class (Tower of God) used for qualitative evaluation only
- Feature extraction pipeline depends on local dataset structure


## Authors

- Mael Pierron — [GitHub](https://github.com/MaelPierron) — mael.pierron@laposte.net
- Liza Belkheir
- Jean-Max Agogue

CPE Lyon — S8, 2026
