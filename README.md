# COL828 Assignment 1 - Fine-Tuning & Prompt-Tuning for CLIP

## Overview
This project implements and evaluates various fine-tuning and prompt-tuning strategies for the Vision Transformer (ViT) backbone of OpenAI's CLIP model in an image classification task. The experiments include zero-shot inference, linear probing, shallow and deep Visual Prompt Tuning (VPT), dual VPT, and full fine-tuning. Metrics such as training and test accuracy, convergence speed, and trainable parameters are analyzed for comparison.

---

## Table of Contents
- [Requirements](#requirements)
- [Running Instructions](#running-instructions)
- [Experiment Setup](#experiment-setup)
- [Experiments and Results](#experiments-and-results)
  - [E1: Zero-Shot Inference](#e1-zero-shot-inference)
  - [E2: Linear Probing](#e2-linear-probing)
  - [E3: Shallow Visual Prompt Tuning (VPT)](#e3-shallow-visual-prompt-tuning-vpt)
  - [E4: Deep Visual Prompt Tuning (VPT)](#e4-deep-visual-prompt-tuning-vpt)
  - [E5: Dual VPT](#e5-dual-vpt)
  - [E6: Full Fine-Tuning](#e6-full-fine-tuning)
- [Comparative Analysis](#comparative-analysis)
- [Conclusion](#conclusion)

---

## Requirements
To run the experiments, install the following dependencies:
- `torch`
- `transformers`
- `datasets`
- `PIL`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install dependencies via:
```bash
pip install torch transformers datasets Pillow scikit-learn matplotlib seaborn

# Experiment Setup

## Model:
OpenAI CLIP with ViT-B/16 backbone

## Dataset:
"aggr8/brain_mri_train_test_split" (classes: glioma, meningioma, no tumor, pituitary)

## Training Configuration:
- **Optimizer:** Adam
- **Learning Rate:** 1e-3
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 500
- **Batch Size:** 32

## Evaluation Metrics:
- Accuracy (Training and Test)
- Loss Curves (Convergence Behavior)
- Convergence Speed (epochs to optimal performance)
- Trainable Parameters (model complexity)

---

# Experiments and Results

## E1: Zero-Shot Inference
- **Objective:** Baseline performance of the pre-trained CLIP model without tuning.
- **Trainable Parameters:** 0
- **Test Accuracy:** 29%

### Classification Report:
| Class       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Glioma       | 0.34      | 0.74   | 0.47     | 91      |
| Meningioma   | 0.22      | 0.36   | 0.27     | 76      |
| No Tumor     | 0.00      | 0.00   | 0.00     | 71      |
| Pituitary    | 0.00      | 0.00   | 0.00     | 82      |
| **Overall**  | **0.29**  | -      | -        | 320     |

---

## E2: Linear Probing
- **Objective:** Fine-tune only the linear classification head.
- **Trainable Parameters:** ~2,000
- **Test Accuracy:** 91%

### Observations:
- Significant improvement over zero-shot inference.
- Rapid convergence within 200 epochs.

---

## E3: Shallow Visual Prompt Tuning (VPT)
- **Objective:** Add learnable tokens to the input of the ViT backbone, freezing other parameters.
- **Trainable Parameters:** ~3,000
- **Test Accuracy:** 88%

### Observations:
- Comparable to linear probing.
- Convergence achieved within 300 epochs with minimal computational overhead.

---

## E4: Deep Visual Prompt Tuning (VPT)
- **Objective:** Add learnable tokens at each layer of the ViT model.
- **Trainable Parameters:** ~11,000
- **Test Accuracy:** 92%

### Observations:
- Further improved performance, slower convergence (400 epochs).
- Increased computational cost due to deeper integration.

---

## E5: Dual VPT (Visual + Textual Prompt Tuning)
- **Objective:** Apply prompt tuning to both visual and textual branches of CLIP.
- **Trainable Parameters:** ~5,000
- **Test Accuracy:** 100%

### Observations:
- Best accuracy among all experiments (100%).
- Convergence within 2 epochs.

---

## E6: Full Fine-Tuning
- **Objective:** Fine-tune all parameters of the ViT backbone.
- **Trainable Parameters:** ~100 million
- **Test Accuracy:** 95%

### Observations:
- Achieved the highest test accuracy (95%).
- Fast convergence (10 epochs) but at the cost of significant computational requirements.

---

# Comparative Analysis

| Experiment   | Trainable Parameters | Test Accuracy | Convergence Speed (Epochs) |
|--------------|-----------------------|---------------|----------------------------|
| E1 (Zero-Shot) | 0                    | 29%           | N/A                        |
| E2 (Linear Head) | ~2,000              | 91%           | 200                        |
| E3 (Shallow VPT) | ~3,000              | 88%           | 300                        |
| E4 (Deep VPT)   | ~11,000              | 92%           | 400                        |
| E5 (Dual VPT)   | ~5,000               | 100%          | 2                          |
| E6 (Full Fine-Tuning) | ~100M          | 95%           | 10                         |

### Key Insights:
- **Zero-Shot (E1):** Baseline performance.
- **Linear Probing (E2):** Effective with minimal parameters.
- **Shallow vs Deep VPT:** Deep VPT improves accuracy at the cost of slower convergence.
- **Dual VPT (E5):** Superior due to multi-modal tuning.
- **Full Fine-Tuning (E6):** Maximizes performance but has the highest computational demand.

---

# Conclusion
The experiments illustrate the trade-off between model complexity and performance:
- **Visual Prompt Tuning (VPT)** offers a middle ground by enhancing frozen models with minimal overhead.
- **Dual VPT** achieves exceptional accuracy by leveraging both visual and textual prompts.
- **Full fine-tuning** yields the highest performance but requires significant computational resources.

Future work could include:
- Regularization techniques to mitigate overfitting.
- Data augmentation to improve model robustness.
