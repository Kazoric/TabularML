import torch
from torch import nn

from dataclasses import dataclass, field
from typing import List, Tuple, Callable

from core.model_base import TabularModel
from core.config import ModelConfig

# class FeatureTokenizer(nn.Module):
#     def __init__(self, num_features, cat_cardinalities, d_token):
#         super().__init__()

#         self.weight = nn.Parameter(torch.randn(num_features, d_token))
#         self.bias = nn.Parameter(torch.randn(num_features, d_token))

#         self.cat_embeddings = nn.ModuleList(
#             [
#                 nn.Embedding(cardinality, d_token)
#                 for cardinality in cat_cardinalities
#             ]
#         )

#     def forward(self, x_num, x_cat):
#         """
#         x_num : (B, N_num)
#         x_cat : (B, N_cat)
#         """
#         x_num = x_num.unsqueeze(-1)  # (B, F, 1)

#         num_tokens  = x_num * self.weight.unsqueeze(0)
#         num_tokens  = num_tokens  + self.bias.unsqueeze(0)

#         cat_tokens = []

#         for i, emb in enumerate(self.cat_embeddings):
#             token = emb(x_cat[:, i])
#             cat_tokens.append(token)

#         cat_tokens = torch.stack(cat_tokens, dim=1)

#         return torch.cat(
#             [num_tokens, cat_tokens],
#             dim=1
#         )


# class FTTransformer(nn.Module):
#     def __init__(
#         self,
#         num_features,
#         d_token=192,
#         n_heads=8,
#         n_layers=3,
#         dim_feedforward=384,
#         dropout=0.1,
#         output_dim=1,
#     ):
#         super().__init__()

#         self.tokenizer = FeatureTokenizer(
#             num_features=num_features,
#             cat_cardinalities=cat_cardinalities,
#             d_token=d_token,
#         )

#         self.cls_token = nn.Parameter(
#             torch.randn(1, 1, d_token)
#         )

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_token,
#             nhead=n_heads,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#             activation="gelu",
#         )

#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=n_layers,
#         )

#         self.norm = nn.LayerNorm(d_token)

#         self.head = nn.Sequential(
#             nn.Linear(d_token, d_token),
#             nn.ReLU(),
#             nn.Linear(d_token, output_dim),
#         )

#     def forward(self, x_num, x_cat):
#         # x : (B, num_features)

#         tokens = self.tokenizer(x_num, x_cat)

#         batch_size = x_num.shape[0]

#         cls = self.cls_token.expand(
#             batch_size, -1, -1
#         )

#         tokens = torch.cat([cls, tokens], dim=1)

#         encoded = self.transformer(tokens)

#         cls_embedding = encoded[:, 0]

#         cls_embedding = self.norm(cls_embedding)

#         output = self.head(cls_embedding)

#         return output
    

# class FTTransformerModel(TabularModel):
#     """Model wrapper for FTTransformer."""

#     def __init__(
#         self,
#         config,
#     ):
#         self.name = "FTTransformer"
        
#         embedding_dim = [
#             (n, min(50, (n + 1) // 2))
#             for n in config.cat_cardinalities
#         ]

#         n_numerical_features = config.n_numeric_features
#         output_dim = config.output_dim
        
#         # This dictionary is the "recipe" to recreate the model instance later.
#         self.params = {
#             "embedding_dim": embedding_dim,
#             "n_numerical_features": n_numerical_features,
#             "output_dim": output_dim,
#         }

#         # Initialize base class
#         super().__init__(model_config=config)

#     def build_model(self):
#         print(
#             f"Building FTTransformer model."
#         )
        
#         # Pass the stored params and the calculated dims to the architecture
#         return FTTransformer(
#             embedding_dim=self.params["embedding_dim"],
#             n_numerical_features=self.params["n_numerical_features"],
#             output_dim=self.params["output_dim"],
#         )


# @dataclass(kw_only=True)
# class FTTransformerConfig(ModelConfig):
#     cat_cardinalities: List[int]
#     n_numeric_features: int
#     output_dim: int


class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        n_numeric_features,
        d_token,
    ):
        super().__init__()

        self.n_numeric_features = n_numeric_features

        # Numériques
        self.num_weight = nn.Parameter(
            torch.randn(n_numeric_features, d_token)
        )

        self.num_bias = nn.Parameter(
            torch.zeros(n_numeric_features, d_token)
        )

        # Catégorielles
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_token)
            for cardinality in cat_cardinalities
        ])

    def forward(self, X_cat, X_num):

        tokens = []

        # Numériques
        if self.n_numeric_features > 0:
            num_tokens = (
                X_num.unsqueeze(-1)
                * self.num_weight.unsqueeze(0)
                + self.num_bias.unsqueeze(0)
            )

            tokens.append(num_tokens)

        # Catégorielles
        if len(self.embeddings) > 0:

            cat_tokens = [
                emb(X_cat[:, i])
                for i, emb in enumerate(self.embeddings)
            ]

            cat_tokens = torch.stack(cat_tokens, dim=1)

            tokens.append(cat_tokens)

        return torch.cat(tokens, dim=1)
    

class FTTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        n_numeric_features,
        output_dim,
        d_token=128,
        n_heads=8,
        n_layers=4,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(
            cat_cardinalities=cat_cardinalities,
            n_numeric_features=n_numeric_features,
            d_token=d_token,
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_token)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, output_dim),
        )

    def forward(self, X_cat, X_num):

        tokens = self.tokenizer(
            X_cat=X_cat,
            X_num=X_num,
        )

        batch_size = tokens.size(0)

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        tokens = torch.cat(
            [cls, tokens],
            dim=1,
        )

        encoded = self.transformer(tokens)

        cls_embedding = encoded[:, 0]

        return self.head(cls_embedding)
    
class FTTransformerModel(TabularModel):

    def __init__(self, config):

        self.name = "FTTransformer"

        self.params = {
            "cat_cardinalities": config.cat_cardinalities,
            "n_numeric_features": config.n_numeric_features,
            "output_dim": config.output_dim,
            "d_token": config.d_token,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "ff_dim": config.ff_dim,
            "dropout": config.dropout,
        }

        super().__init__(model_config=config)

    def build_model(self):

        print("Building FTTransformer model.")

        return FTTransformer(
            **self.params
        )

@dataclass(kw_only=True)
class FTTransformerConfig(ModelConfig):

    cat_cardinalities: List[int]

    n_numeric_features: int

    output_dim: int

    d_token: int = 128

    n_heads: int = 8

    n_layers: int = 4

    ff_dim: int = 256

    dropout: float = 0.1