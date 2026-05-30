"""LightGBM inference adapter."""

from __future__ import annotations

from typing import Any

import numpy as np


class LightGBMModelAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_proba(self, features) -> float:
        probability = self.model.predict_proba(features)[:, 1]
        return float(np.asarray(probability).reshape(-1)[0])
