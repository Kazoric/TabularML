import torch
from torch import nn

from dataclasses import dataclass, field
from typing import List, Tuple, Callable

from core.model_base import TabularModel
from core.config import ModelConfig

class FullyConnected(nn.Module):
    def __init__(self, embedding_dim, n_numerical_features, output_dim):
        super(FullyConnected, self).__init__()


        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, dim) for num_embeddings, dim in embedding_dim
        ])

        n_embeddings_output = sum(dim for _, dim in embedding_dim)
        n_inputs = n_embeddings_output + n_numerical_features
        
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )

        self.output = nn.Sigmoid()

    def forward(self, X_cat, X_num):
        cat_embeddings = [embedding(X_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        cat_out = torch.cat(cat_embeddings, dim=1)

        x = torch.cat([cat_out, X_num], dim=1)
        x = self.layers(x)
        x = self.output(x)
        return x
        

class FullyConnectedModel(TabularModel):
    """Model wrapper for FullyConnected."""

    def __init__(
        self,
        config,
    ):
        self.name = "FullyConnected"
        
        embedding_dim = [
            (n, min(50, (n + 1) // 2))
            for n in config.cat_cardinalities
        ]

        n_numerical_features = config.n_numeric_features
        output_dim = 1
        
        # This dictionary is the "recipe" to recreate the model instance later.
        self.params = {
            "embedding_dim": embedding_dim,
            "n_numerical_features": n_numerical_features,
            "output_dim": output_dim,
        }

        # Initialize base class
        super().__init__(model_config=config)

    def build_model(self):
        print(
            f"Building FullyConnected model."
        )
        
        # Pass the stored params and the calculated dims to the architecture
        return FullyConnected(
            embedding_dim=self.params["embedding_dim"],
            n_numerical_features=self.params["n_numerical_features"],
            output_dim=self.params["output_dim"],
        )


@dataclass(kw_only=True)
class FullyConnectedConfig(ModelConfig):
    cat_cardinalities: List[int]
    n_numeric_features: int
    output_dim: int