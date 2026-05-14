"""Pre-2 planning and migration utilities.

This package is intentionally separate from ``myllm`` while the PyTorch /
TorchTitan stack is being introduced. The existing ``myllm`` package remains
the pre-1 JAX/Keras reference implementation.
"""

from .config import DataMixConfig, DensePre2Config, load_data_mix_config, load_dense_config
from .guards import HETEROGENEOUS_TOPK_KD, reject_topk_kd_inputs

__all__ = [
    "DataMixConfig",
    "DensePre2Config",
    "HETEROGENEOUS_TOPK_KD",
    "load_data_mix_config",
    "load_dense_config",
    "reject_topk_kd_inputs",
]
