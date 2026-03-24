# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import SequentialLR, LinearLR
import numpy as np
from sklearn.model_selection import KFold
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Tuple, Callable

# Import custom modules
from core.trainer import Trainer
# from core.predictor import Predictor
# from core.checkpoint import CheckpointManager
from core.config import ModelConfig, TrainerConfig

class TabularModel(ABC):
    def __init__(
        self,
        model_config: ModelConfig,
    ) -> None:
        """
        Initialize the Model object.
        
        Args:
            config (ModelConfig): Configuration object containing model parameters
        
        Returns:
            None
        """

        # Set device to CPU or GPU if available
        self.device = model_config.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.build_model().to(self.device)

        # Initialize trainer
        self.trainer = Trainer(
            model = self.model,
            model_config = model_config,
        )

        # Initialize predictor
        # self.predictor = Predictor(self.model, self.device)

    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Abstract method to build the model. Must be implemented by subclasses.
        
        Returns:
            nn.Module: The built model
        """
        pass

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10) -> None:
        """
        Train the model for a specified number of epochs.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader (optional)
            epochs (int): Number of epochs to train for
        """
        
        self.trainer.train(train_loader, val_loader, epochs)

    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model on a validation set.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Model performance metric (e.g., accuracy or loss)
        """
        
        return self.trainer.evaluate(val_loader)

    def cross_validation_train(self, train_loader: DataLoader, epochs: int = 10, k: int = 5) -> None:
        """
        Train the model for a specified number of epochs using k-fold cross-validation.
        
        Args:
            train_loader (DataLoader): Training data loader
            epochs (int): Number of epochs to train for
            k (int): Number of folds for cross-validation
        """
        train_dataset = train_loader.dataset
        batch_size = train_loader.batch_size

        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        losses = []
        all_scores = {}

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            self.model = self.build_model().to(self.device)
            self.trainer.reset(self.model)
            print(f"Fold {fold+1}/{k}")
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            self.train(train_loader, val_loader, epochs)
            loss = self.trainer.get_final_metrics()['val_loss']
            losses.append(loss)
            fold_metrics = self.trainer.get_final_metrics()['valid_metrics']
            print(f"Fold {fold+1} metrics:")
            for metric_name, value in fold_metrics.items():
                if metric_name not in all_scores:
                    all_scores[metric_name] = []
                all_scores[metric_name].append(value.item())
                print(f"{metric_name}: {value.item()}")
            print("\n")
        
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        print(f"Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        for metric_name, values in all_scores.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")