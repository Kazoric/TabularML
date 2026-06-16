# TabularML

TabularML is a lightweight, modular PyTorch-based framework designed to streamline the creation, training, and evaluation of deep neural networks tailored for tabular data.

The architecture strictly decouples data handling, neural network architectures, and the core training mechanics, making it highly reusable and scalable.

## 🚀 Features

- **Modular training pipeline** with clean separation of responsibilities:
  - `Trainer`: handles training loop, validation, metrics, scheduler
  - `Predictor`: for single-batch or multi-batch inference
  - `Visualizer`: for plotting loss and metrics over epochs
- Built-in support for metrics:
  - **Accuracy**, **Balanced Accuracy**, **ROC AUC**
  - Implemented using **pure PyTorch** (no `sklearn`)
- Configurable schedulers and optimizers
- Loss and metrics plotted after training
- Easily extensible to new models

## 🧠 Supported Architectures

- FullyConnected

Each model file defines:
- `Architecture`: the PyTorch `nn.Module` class
- `Model`: a subclass of `ModelBase` that builds the architecture

## 🛠️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Kazoric/TabularML.git
cd TabularML

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements
pip install -r requirements.txt
```

## 📂 Project Structure

```bash
.
├── core/                       # Framework engine
│   ├── model_base.py           # Base Model class orchestrating training and inference
│   ├── trainer.py              # Training loop (forward/backward, validation, metrics)
│   ├── predictor.py            # Inference logic (evaluation on unseen data)
│   ├── config.py               # Hyperparameter and configuration management
│   ├── visualizer.py           # Generation of learning curves and metrics plots
│   └── metrics.py              # Native PyTorch implementations of F1, Precision, Recall, Accuracy, ROC AUC
│
├── data/
│   ├── preprocessing.py        # Preprocessing function
│   └── data_loader.py          # PyTorch Dataset and DataLoader setups for tabular data
│
├── models/                     # Custom neural network architectures
│   └── FullyConnected.py       # Example: Standard Multi-Layer Perceptron (MLP)
│
├── experiments/                # Local storage for weights, metrics, and plots (Git ignored)
│
└── README.md                   # Project documentation
```

## ## Working with Custom Datasets

The framework is highly flexible, but because tabular datasets often require specific cleaning, encoding, or feature scaling, **you must update the preprocessing pipeline when switching to a new dataset**.

Before running your training, make sure to adapt the `preprocessing` logic located inside `data/preprocessing.py` to handle:
* **Categorical Mapping:** Updating label encoders or dictionary mappings to match your new target or feature columns.
* **Missing Value Imputation:** Adjusting how null values are filled based on the nature of the new data.
* **Feature Scaling:** Modifying standardizers or normalization techniques ($MinMax$, $Z-score$) depending on the feature distributions.

> ⚠️ **Important:** Failing to update the preprocessing steps to match your new dataset's schema will likely result in shape mismatches or embedding errors during the PyTorch forward pass.