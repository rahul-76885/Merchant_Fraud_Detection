"""ANN inference adapter (PyTorch)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dl_models import torch_available

if torch_available():
    import torch


class ANNModelAdapter:
    def __init__(self, model: Any, device: Any) -> None:
        self.model = model
        self.device = device

    def predict_proba(self, x_num: np.ndarray, x_cat: np.ndarray) -> float:
        if not torch_available():
            raise RuntimeError("PyTorch is unavailable for ANN inference")
        x_num_t = torch.from_numpy(x_num.astype(np.float32)).to(self.device)
        x_cat_t = torch.from_numpy(x_cat.astype(np.int64)).to(self.device)
        with torch.no_grad():
            probability = torch.sigmoid(self.model(x_num_t, x_cat_t)).detach().cpu().numpy()
        return float(np.asarray(probability).reshape(-1)[0])
