# Transfer Learning for Image Classification under Data Scarcity

This repository contains the empirical implementation and experimental analysis for **Deep Learning Take-Home Assignment — Session 05: Advanced CNNs and Transfer Learning**. The project systematically explores different transfer learning adaptation paradigms under a small-data regime using a 10-class subset of the Oxford-IIIT Pet dataset.

---

## 👤 Author Information
* **Name** : Richtjhie Hartawan Agusta
* **Student ID (NIM)** : 2802529102
* **Course** : Deep Learning (Session 05: Advanced CNNs & Transfer Learning)

---

## 📌 Project Overview
This study empirically investigates five distinct training configurations across modern Convolutional Neural Network (CNN) architectures:
1. **B1: Baseline from Scratch** — ResNet-18 initialized with random weights (`weights=None`), serving as the lower-bound baseline.
2. **B2: Feature Extraction** — Pretrained ResNet-18 with the entire backbone frozen (`requires_grad=False`), training only a newly initialized linear classifier head.
3. **B3: Progressive Fine-Tuning** — Pretrained ResNet-18 with the final residual block (`layer4`) and `fc` unfrozen, trained using a differential learning rate ($10^{-4}$ for `layer4`, $10^{-3}$ for `fc`).
4. **B4: Data Augmentation Control Study** — Controlled comparison evaluating the model trained without augmentation vs. with active stochastic data augmentation.
5. **B5: Efficiency-Oriented Architecture** — Pretrained EfficientNet-B0 fine-tuned on top layers (`features[7..8]` + classifier) to evaluate computational trade-offs against ResNet-18.

### Dataset & Stratified Splitting
* **Dataset**: Oxford-IIIT Pet (10 selected classes: 5 cat breeds and 5 dog breeds, 1,993 images total).
* **Class Names**: *Abyssinian, Bengal, Birman, Persian, Ragdoll* (Cats); *Chihuahua, Boxer, English Cocker Spaniel, Pomeranian, Pug* (Dogs).
* **Data Partitioning**: Stratified split preserving class distributions:
  * **Train Set**: 1,395 images (70.0%)
  * **Validation Set**: 299 images (15.0%)
  * **Test Set**: 299 images (15.0%)

---

## 📊 Experimental Results

All experiments were executed under standardized protocols: input dimension $224 \times 224$ (ImageNet normalization), **AdamW** optimizer with `weight_decay=1e-2`, batch size 32, fixed random seed (42), and **Early Stopping** monitoring validation loss with `patience=4`:

| Experiment Condition | Trainable Parameters | Total Parameters | Learning Rate Group | Best Epoch | Training Time (s) | Val Accuracy (%) | Test Accuracy (%) | Test Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1: Scratch** | 11,181,642 (100.0%) | 11,181,642 | 1.0e-03 | 7 | 70.09 | 38.46 | 35.12 | 0.3205 |
| **B2: Feature_Ext** | 5,130 (0.05%) | 11,181,642 | 1.0e-03 | 13 | 93.77 | 92.64 | 93.98 | 0.9395 |
| **B3: Fine_Tuning** | 8,398,858 (75.11%) | 11,181,642 | 1.0e-04 / 1.0e-03 | 3 | 38.76 | 94.65 | 93.31 | 0.9336 |
| **B4: No_Augment** | 8,398,858 (75.11%) | 11,181,642 | 1.0e-04 / 1.0e-03 | 5 | 30.10 | 94.98 | 95.32 | 0.9531 |
| **B4: With_Augment** | 8,398,858 (75.11%) | 11,181,642 | 1.0e-04 / 1.0e-03 | 3 | 37.78 | 94.65 | 93.31 | 0.9336 |
| **B5: EfficientNet** | 1,142,202 (28.41%) | 4,020,358 | 1.0e-04 / 1.0e-03 | 15 | 102.90 | 94.31 | 94.98 | 0.9498 |

---

## 🔍 Key Empirical Insights

1. **Extreme Performance Disparity (Scratch vs. Transfer Learning)**:  
   Training ResNet-18 from scratch on 1,395 images yields a dismal **Test Macro F1 of 0.3205**, suffering from extreme gradient noise due to an unfavorable parameter-to-sample ratio (~8,000 : 1). Transfer learning paradigms jump immediately to **Macro F1 > 0.93**, leveraging rich inductive priors inherited from ImageNet.
2. **Mitigation of Overfitting via Data Augmentation**:  
   Without augmentation (B4 No-Aug), the model exhibits severe sample memorization: training accuracy reaches **100.00%** with a training loss of **0.0083**, creating a wide **generalization gap of +5.02 pp** ($100.00\% - 94.98\%$). Active data augmentation effectively dampens this gap down to **+1.48 pp** (a ~70% reduction) while attaining the lowest validation loss (**0.1616**).
3. **Feature Attribution & Grad-CAM Analysis**:  
   Grad-CAM heatmaps extracted from `layer4` reveal that prior to fine-tuning (B2), activations are diffuse and frequently distracted by background contextual pixels. Following progressive fine-tuning (B3), model attention shifts and condenses tightly onto domain-discriminative anatomical markers (eyes, muzzle, facial contour, and breed-specific fur markings).

---

## 📁 Repository Structure
```text
.
├── TF_Learning_Experiment.ipynb   # Complete, fully executed Jupyter Notebook with outputs
├── requirements.txt               # List of required Python dependencies
├── LICENSE                        # MIT License
├── README.md                      # Comprehensive project documentation
└── logs/                          # Directory containing raw metric logs and generated plots
    ├── metrics_summary.csv        # Consolidated experimental summary in CSV format
    ├── metrics_summary.json       # Consolidated experimental summary in JSON format
    ├── epoch_histories.json       # Full per-epoch loss and accuracy histories
    ├── loss_curves_B1_B2_B3.png   # Comparative loss curves (B1, B2, B3)
    ├── augmentation_gap_B4.png    # Generalization gap curve (B4 With-Aug vs. No-Aug)
    ├── failure_analysis_grid.png  # Qualitative failure analysis visualization
    └── gradcam_before_after_FT.png# Comparative Grad-CAM visualization on layer4
```

---

## 🚀 Installation & Reproduction

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/transfer-learning-oxford-pets-.git
   cd transfer-learning-oxford-pets-
   ```

2. **Install Dependencies**:
   Ensure you are using Python 3.10+:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Execute all cells end-to-end in Google Colab (recommended: free T4 GPU runtime) or locally via Jupyter Notebook:
   ```bash
   jupyter notebook TF_Learning_Experiment.ipynb
   ```

---

## 📜 License
This project is licensed under the terms of the **MIT License**. See the [LICENSE](LICENSE) file for details.