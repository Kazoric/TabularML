# Import necessary libraries
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Optional

from core import metrics
from core.metrics import METRICS_REGISTRY
from core.config import ModelConfig

class Trainer:
    """
    Class for training a PyTorch model.

    """
    
    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        """
        Initialize the Trainer object.

        """
        
        # Set model and optimizer attributes
        self.model = model

        self.criterion = model_config.criterion

        self.metrics = model_config.metrics

        self.lr = model_config.lr

        self.optimizer_params = model_config.optimizer_params
        self.optimizer = model_config.optimizer(self.model.parameters(), lr=self.lr, **self.optimizer_params)

        self.scheduler_params = model_config.scheduler_params
        self.scheduler = model_config.scheduler(self.optimizer, **self.scheduler_params)
        
        # Set device attribute
        self.device = model_config.device

        # Set number of classes
        self.output_dim = model_config.output_dim

        # Initialize loss and metric tracking lists
        self.train_loss = []
        self.valid_loss = []

        # Initialize learning rate tracking lists
        self.lr_history = []

        self.save = model_config.save

        # Initialize best validation loss and start epoch attributes
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        self.best_epoch_metrics: dict = {}
        
        # Initialize train and validation metric tracking dictionaries
        self.train_metrics = {name: [] for name in self.metrics}
        self.valid_metrics = {name: [] for name in self.metrics}

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 10) -> None:
        """
        Train the model on a given dataset.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader (optional)
            epochs (int): Number of training epochs
        
        Returns:
            None
        """
        
        # Iterate over training epochs
        for epoch in range(self.start_epoch, epochs):

            # Store learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            
            # Set model to training mode
            self.model.train()
            
            # Initialize running loss and prediction tracking lists
            running_loss = 0.0
            all_outputs = []
            all_labels = []

            # Iterate over training data loader
            for X_cat, X_num, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                
                # Move input to specified device
                X_cat, X_num, y = X_cat.to(self.device, non_blocking=True), X_num.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                # Zero gradients and make predictions
                self.optimizer.zero_grad()
                outputs = self.model(X_cat, X_num)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                # Update running loss and prediction tracking lists
                running_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(y)

            # Update train loss and metric tracking lists
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            self.train_loss.append(running_loss)

            # Compute and store metrics
            metric_outputs = self._compute_metrics(all_labels, all_outputs)
            for name, value in metric_outputs.items():
                self.train_metrics[name].append(value.cpu())

            # Print training metrics
            metrics_str = " | ".join(f"{name}: {value:.4f}" for name, value in metric_outputs.items())
            print(f"{'Train':<12} | Loss: {running_loss:.4f} | {metrics_str} | Learning Rate: {current_lr:.4f}")

            # Evaluate model on validation set if available
            if val_loader:
                val_loss = self.evaluate(val_loader)

                # Save best model if validation loss improves
                if self.save and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    last_train_metrics = {
                        name: self.train_metrics[name][-1] for name in self.metrics
                    }

                    last_valid_metrics = {
                        name: self.valid_metrics[name][-1] for name in self.metrics
                    }
                    self.best_epoch_metrics = {
                        "epoch": epoch + 1,
                        "train_loss": running_loss,
                        "val_loss": val_loss,
                        "train_metrics": last_train_metrics,
                        "valid_metrics": last_valid_metrics
                    }
                    # if self.save_checkpoint:
                    #     self.save_checkpoint(epoch + 1, val_loss)
            
            # Update learning rate scheduler if available
            if self.scheduler is not None:
                self.scheduler.step()

            print()

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model on a given dataset.
        
        Args:
            data_loader (DataLoader): Data loader to use for evaluation
        
        Returns:
            float: Validation loss
        """
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize running loss and prediction tracking lists
        running_loss = 0.0
        all_outputs = []
        all_labels = []

        # Iterate over data loader
        with torch.no_grad():
            for X_cat, X_num, y in data_loader:
                
                # Move input to specified device
                X_cat, X_num, y = X_cat.to(self.device, non_blocking=True), X_num.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                # Make predictions
                outputs = self.model(X_cat, X_num)
                loss = self.criterion(outputs, y)

                # Update running loss and prediction tracking lists
                running_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(y)

        # Update validation loss and metric tracking lists
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        self.valid_loss.append(running_loss)

        # Compute and store metrics
        metric_outputs = self._compute_metrics(all_labels, all_outputs)
        for name, value in metric_outputs.items():
            self.valid_metrics[name].append(value.cpu())

        # Print validation metrics
        metrics_str = " | ".join(f"{name}: {value:.4f}" for name, value in metric_outputs.items())
        print(f"{'Validation':<12} | Loss: {running_loss:.4f} | {metrics_str}")
        
        return running_loss

    def _compute_metrics(self, y_true, y_pred):
        metric_outputs = {}

        for metric_name in self.metrics:
            if metric_name not in METRICS_REGISTRY:
                raise ValueError(f"Metric '{metric_name}' not registered")

            metric_fn = METRICS_REGISTRY[metric_name]
            metric_outputs[metric_name] = metric_fn(y_true, y_pred)
        return metric_outputs
    
    def get_final_metrics(self) -> dict:
        return self.best_epoch_metrics

    def reset(self, model: nn.Module) -> None:
        """
        Reset the Trainer state for a new fold or new training session.
        
        Args:
            model (nn.Module): A fresh model instance to attach to this Trainer.
        """
        self.model = model.to(self.device)

        # Recreate optimizer
        self.optimizer = self.optimizer.__class__(
            self.model.parameters(),
            lr=self.lr,
            **self.optimizer_params
        )

        # Recreate scheduler
        if self.scheduler_params:
            self.scheduler = self.scheduler.__class__(self.optimizer, **self.scheduler_params)

        # Reset losses and metrics
        self.train_loss = []
        self.valid_loss = []
        self.lr_history = []

        self.train_metrics = {name: [] for name in self.metrics}
        self.valid_metrics = {name: [] for name in self.metrics}

        # Reset best validation tracking
        self.best_val_loss = float('inf')
        self.best_epoch_metrics = {}
        self.start_epoch = 0