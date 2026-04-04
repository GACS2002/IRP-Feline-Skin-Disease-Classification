# Hybrid ViT-ML Framework for Feline Skin Disease Classification

**Individual Research Project | BSc (Hons) Artificial Intelligence & Data Science**  
Informatics Institute of Technology (IIT) affiliated with Robert Gordon University (RGU)

---

## Overview

This is a hybrid diagnostic framework that combines a pretrained **DeiT-Small Vision Transformer** as a frozen feature extractor with classical machine learning classifiers for feline skin disease classification. The framework was designed to operate under the data-constrained conditions characteristic of veterinary AI applications, where large-scale annotated datasets are unavailable.

The system classifies feline skin images into four categories:
- **Flea Allergy**
- **Health** (No Observed Pathology)
- **Ringworm**
- **Scabies**

The optimal pipeline — **RF + PCA** — achieved **85.81% test accuracy** on a clinician-validated dataset of 973 images, outperforming a ResNet50 CNN baseline by 8.11 percentage points under identical data conditions.

---

## Repository Structure

```
IRP-Feline-Skin-Disease-Classification/
│
├── models/                         
│   ├── deit_feature_extractor.pth  # Frozen DeiT-Small weights
│   ├── pca_transformer.pkl         # Fitted PCA transformer
│   ├── svm_pca_model.pkl           # Final SVM + PCA classifier
│   ├── mi_selected_idx.pkl         # MI selected feature indices
│   └── boruta_selected_features.pkl# Boruta selected feature indices
│
├── Data_Preprocessing.ipynb        # Dataset splitting (70/15/15 per class)
├── Feature_extraction.ipynb        # DeiT-Small feature extraction → .npy arrays
├── Feature_Selection.ipynb         # PCA, Mutual Information, Boruta selection
├── Classification_SVM_1.ipynb      # SVM classifier (PCA, MI, Boruta)
├── Classification_RF_1.ipynb       # Random Forest classifier (PCA, MI, Boruta)
├── Classification_XGB_1.ipynb      # XGBoost classifier (PCA, MI, Boruta)
├── CNN_compare.ipynb               # ResNet50 CNN baseline (early stopping)
│
├── app.py                          # Flask backend for web prototype
├── index.html                      # Frontend UI for FeliDerm prototype
├── requirements.txt                # Python dependencies
│
├── CAT SKIN DISEASE.zip            # Original dataset (pre-validation)
├── CAT SKIN DISEASE After Validation.zip  # Clinician-validated dataset (973 images)
│
├── X_train_phaseA.npy              # Raw DeiT embeddings - train
├── X_val_phaseA.npy                # Raw DeiT embeddings - val
├── X_test_phaseA.npy               # Raw DeiT embeddings - test
├── X_train_pca.npy                 # PCA features - train
├── X_val_pca.npy                   # PCA features - val
├── X_test_pca.npy                  # PCA features - test
├── X_train_mi.npy                  # MI features - train
├── X_val_mi.npy                    # MI features - val
├── X_test_mi.npy                   # MI features - test
├── X_train_boruta.npy              # Boruta features - train
├── X_val_boruta.npy                # Boruta features - val
├── X_test_boruta.npy               # Boruta features - test
├── y_train_phaseA.npy              # Labels - train
├── y_val_phaseA.npy                # Labels - val
└── y_test_phaseA.npy               # Labels - test
```

---

## Pipeline Architecture

```
Input Image
    ↓
DeiT-Small (frozen, pretrained on ImageNet)
    → 384-dimensional embedding
    ↓
Feature Selection
    ├── PCA          → 197 features (95% variance retained)
    ├── Mutual Info  → 260 features (adaptive k search)
    └── Boruta       → 250 features (all-relevant selection)
    ↓
Classification
    ├── SVM (RBF kernel)
    ├── Random Forest
    └── XGBoost
    ↓
Predicted Class + Confidence Score
```

---

## Results Summary

| Combination | Val Acc | Test Acc | Gap  | Macro F1 |
|-------------|---------|----------|------|----------|
| **RF + PCA**| 92.47% | **85.81%**| 6.66%| 0.86 |
| SVM + PCA   | 94.52% | 85.81%   | 8.71% | 0.86 |
| SVM + MI    | 95.21% | 85.14%   | 10.07%| 0.85 |
| RF + MI     | 91.78% | 85.14%   | 6.64% | 0.85 |
| XGB + MI    | 90.41% | 84.46%   | 5.95% | 0.84 |
| XGB + Boruta| 90.41% | 83.78%   | 6.63% | 0.83 |
| RF + Boruta | 90.41% | 83.78%   | 6.63% | 0.83 |
| SVM + Boruta| 93.84% | 83.11%   | 10.73%| 0.83 |
| XGB + PCA   | 92.47% | 82.43%   | 10.04%| 0.82 |
| ResNet50 (CNN baseline) | 84.93% | 77.70% | 7.23% | — |

RF + PCA selected as the optimal configuration for prototype deployment.

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/GACS2002/IRP-Feline-Skin-Disease-Classification.git
cd IRP-Feline-Skin-Disease-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebooks in order
If running the full pipeline from scratch:
1. `Data_Preprocessing.ipynb` — split the dataset
2. `Feature_extraction.ipynb` — extract DeiT embeddings
3. `Feature_Selection.ipynb` — apply PCA / MI / Boruta
4. `Classification_SVM_1.ipynb` / `Classification_RF_1.ipynb` / `Classification_XGB_1.ipynb` — train and evaluate
5. `CNN_compare.ipynb` — train ResNet50 baseline

If using the pre-extracted `.npy` arrays and saved models, skip to step 4.

---

## Running the Prototype

```bash
python app.py
```

> **Note:** Ensure the `models/` folder contains all saved artefacts before running the prototype.

---

## Dataset

The dataset is derived from the **Cat Skin Disease V3** dataset available on [Kaggle](https://www.kaggle.com). The original dataset of 1,000 images was clinically validated, resulting in the removal of 27 clinically inconsistent images. The final validated dataset of **973 images** across four classes was used for all experiments.

- `CAT SKIN DISEASE.zip` — original unvalidated dataset
- `CAT SKIN DISEASE After Validation.zip` — clinician-validated dataset used in all experiments

---

## Class Mapping

ImageFolder assigns classes alphabetically:

| Index | Class |
|-------|-------|
| 0 | Flea_Allergy |
| 1 | Health |
| 2 | Ringworm |
| 3 | Scabies |

---

## Acknowledgements

- Supervisor: **Narmada Balasooriya**, Informatics Institute of Technology
- Dataset: [Cat Skin Disease V3 — Kaggle](https://www.kaggle.com/datasets/vekified/cat-skin-disease-v3)
- DeiT-Small pretrained model via [timm](https://github.com/huggingface/pytorch-image-models)

---

*This repository is submitted as part of the Individual Research Project for BSc (Hons) Artificial Intelligence & Data Science at IIT affiliated with RGU.*
