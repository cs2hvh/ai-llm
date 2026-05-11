"""Determinism helpers — seed every RNG we know about."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (lazily) Keras/JAX backends.

    Call before constructing the model and before any data shuffling. Does
    NOT make CUDA fully deterministic — that requires backend-specific flags
    set at process start time and is generally not worth the throughput cost
    for large-scale training.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
