"""Tests for ShardingConfig (no JAX runtime needed)."""
from __future__ import annotations

import pytest

from myllm.training import ShardingConfig


class TestShardingConfig:
    def test_total_devices(self):
        c = ShardingConfig(data_parallel=8, model_parallel=1)
        assert c.total_devices == 8

    def test_2d_total_devices(self):
        c = ShardingConfig(data_parallel=4, model_parallel=8)
        assert c.total_devices == 32

    def test_default_model_parallel(self):
        c = ShardingConfig(data_parallel=8)
        assert c.model_parallel == 1

    def test_invalid_zero(self):
        with pytest.raises(ValueError):
            ShardingConfig(data_parallel=0, model_parallel=1)
        with pytest.raises(ValueError):
            ShardingConfig(data_parallel=8, model_parallel=0)

    def test_frozen(self):
        c = ShardingConfig(data_parallel=8)
        with pytest.raises(Exception):
            c.data_parallel = 16  # type: ignore[misc]
