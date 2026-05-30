"""Model I/O helpers for the simplified fraud stack."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch is optional for lightweight inference envs
    torch = None
    nn = None

logger = logging.getLogger("merchant.dl_models")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def model_path(filename: str) -> str:
    return os.path.join(MODEL_DIR, filename)


def load_pickle_artifact(filename: str) -> Any:
    path = model_path(filename)
    with open(path, "rb") as handle:
        artifact = pickle.load(handle)
    logger.info("Loaded %s", filename)
    return artifact


def save_pickle_artifact(filename: str, artifact: Any) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = model_path(filename)
    with open(path, "wb") as handle:
        pickle.dump(artifact, handle)
    logger.info("Saved %s", filename)
    return path


def torch_available() -> bool:
    return torch is not None and nn is not None


if nn is not None:
    class DNNFraudModel(nn.Module):
        """Dense tabular model with categorical embeddings."""

        def __init__(
            self,
            num_features: int,
            cat_cardinalities: list[int],
            hidden_dims: list[int] | None = None,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            hidden_dims = hidden_dims or [256, 128, 64]
            self.num_features = int(num_features)
            self.cat_cardinalities = [int(max(v, 2)) for v in cat_cardinalities]
            self.embedding_dims = [min(64, max(4, int(np.ceil(np.sqrt(v))))) for v in self.cat_cardinalities]
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(cardinality, emb_dim) for cardinality, emb_dim in zip(self.cat_cardinalities, self.embedding_dims)]
            )

            input_dim = self.num_features + sum(self.embedding_dims)
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for dim in hidden_dims:
                layers.extend([nn.Linear(prev_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)])
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x_num, x_cat):
            if len(self.cat_embeddings) > 0 and x_cat is not None and x_cat.shape[1] > 0:
                embedded = [emb(x_cat[:, idx]) for idx, emb in enumerate(self.cat_embeddings)]
                x = torch.cat([x_num] + embedded, dim=1)
            else:
                x = x_num
            return self.mlp(x).squeeze(1)


    class FTTransformerFraudModel(nn.Module):
        """Compact FT-Transformer style model for tabular fraud prediction."""

        def __init__(
            self,
            num_features: int,
            cat_cardinalities: list[int],
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.num_features = int(num_features)
            self.cat_cardinalities = [int(max(v, 2)) for v in cat_cardinalities]

            self.numeric_tokenizer = nn.Linear(1, d_model)
            self.cat_embeddings = nn.ModuleList([nn.Embedding(cardinality, d_model) for cardinality in self.cat_cardinalities])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

        def forward(self, x_num, x_cat):
            batch_size = x_num.shape[0]
            num_tokens = self.numeric_tokenizer(x_num.unsqueeze(-1)) if x_num.shape[1] > 0 else None
            tokens = []
            if num_tokens is not None:
                tokens.append(num_tokens)
            if len(self.cat_embeddings) > 0 and x_cat is not None and x_cat.shape[1] > 0:
                cat_tokens = [emb(x_cat[:, idx]).unsqueeze(1) for idx, emb in enumerate(self.cat_embeddings)]
                tokens.append(torch.cat(cat_tokens, dim=1))

            if not tokens:
                raise RuntimeError("Transformer requires at least one input token")

            x = torch.cat(tokens, dim=1)
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.encoder(x)
            cls_out = x[:, 0, :]
            return self.head(cls_out).squeeze(1)
else:
    class DNNFraudModel:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for DNNFraudModel")


    class FTTransformerFraudModel:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for FTTransformerFraudModel")


def save_torch_checkpoint(filename: str, payload: dict[str, Any]) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is not available; cannot save checkpoint")
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = model_path(filename)
    torch.save(payload, path)
    logger.info("Saved %s", filename)
    return path


def load_torch_checkpoint(filename: str, map_location: str = "cpu") -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is not available; cannot load checkpoint")
    path = model_path(filename)
    payload = torch.load(path, map_location=map_location)
    logger.info("Loaded %s", filename)
    return payload