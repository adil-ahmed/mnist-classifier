# MNIST Digit Classifier

A modular deep learning project built from scratch using **PyTorch** to classify handwritten digits from the **MNIST** dataset.

---

## What We Did
- Built an **end-to-end digit classifier** using two architectures:
  - **MLP (Multi-Layer Perceptron)** — a fully connected baseline
  - **Simple CNN** — a convolutional model for better feature extraction
- Designed a **modular structure** with separate scripts for data, models, and training.
- Implemented a clean **training–evaluation pipeline** and logged metrics.
- Visualized **loss and accuracy curves** for both architectures.

---

## Why We Did It
To learn how real-world ML/DL pipelines are built —  
understanding not just how to train a model, but how to structure code for scalability, experiment tracking, and reproducibility.

---

## How It Works
1. **Data Loading:** MNIST dataset with PyTorch `DataLoader`  
2. **Model Definition:** MLP and CNN models (`src/models.py`)  
3. **Training Loop:** Custom training and evaluation (`src/train.py`)  
4. **Results:** Saved models, metrics, and plots in `results/`

---

## We Achieved
| Model | Accuracy | Epochs | Optimizer |
|--------|-----------|---------|------------|
| MLP | ~97% | 5 | Adam |
| CNN | ~98% | 8 | AdamW |

---

## Tech Stack
- Python • PyTorch • Matplotlib • NumPy • Jupyter Lab

---
## Possible Future Exploration

1. Compare different optimizers (SGD, RMSProp, Adam, AdamW)
2. Add Dropout and Batch Normalization
3. Implement a confusion matrix and classification report
4. Experiment with FashionMNIST and CIFAR-10

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
jupyter lab

