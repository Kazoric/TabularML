from jinja2.tests import test_filter
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass(kw_only=True)
class ModelConfig:
    input_dim: int
    output_dim: int
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    optimizer_params: Optional[dict] = field(default_factory=dict)
    metrics: Optional[Callable] = field(default_factory=dict)
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    task: str = "classification"  # "classification" or "regression"
    device: str = "cuda"
    save: bool = True
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    scheduler_params: Optional[dict] = field(default_factory=dict)

@dataclass(kw_only=True)
class TrainerConfig:
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    device: str = "cuda"
    save_checkpoints: bool = True

    # model: torch.nn.Module
    # optimizer: torch.optim.Optimizer
    # criterion: torch.nn.Module
    # device: str
    # save: bool = False
    # checkpoint_fn: Callable[[int, float], None] = None
    # scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    # metrics: Optional[dict] = None
    # num_classes: Optional[int] = None

@dataclass(kw_only=True)
class DataConfig:
    train_file: str
    test_file: str
    categorical_features: List[str]
    numeric_features: List[str]
    target: str
    batch_size: int = 1024