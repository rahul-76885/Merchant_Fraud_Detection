"""Model I/O helpers for the simplified fraud stack."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

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