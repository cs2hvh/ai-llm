"""Custom exception hierarchy.

All MyLLM exceptions derive from ``MyLLMError`` so callers can catch the
full surface in one place. Domain-specific errors derive from the
appropriate intermediate class.
"""
from __future__ import annotations


class MyLLMError(Exception):
    """Root exception for all MyLLM errors."""


class ConfigError(MyLLMError):
    """Raised when a configuration is missing, malformed, or inconsistent."""


class DataPipelineError(MyLLMError):
    """Raised by the data pipeline (loaders, filters, dedupe, mixing)."""


class FilterError(DataPipelineError):
    """Raised when a filter fails unexpectedly (not a normal reject decision)."""


class DedupeError(DataPipelineError):
    """Raised by the deduplication subsystem."""


class TrainingError(MyLLMError):
    """Raised by the training loop, optimizer, or checkpointing."""


class CheckpointError(TrainingError):
    """Raised by checkpoint save/load."""


class LossSpikeError(TrainingError):
    """Raised when the watchdog detects a non-recoverable loss divergence."""


class RunPodAPIError(MyLLMError):
    """Raised by the RunPod orchestration client."""


class RunPodLaunchError(RunPodAPIError):
    """Pod launch failed (capacity, quota, auth, ...)."""


class CostCeilingExceeded(RunPodAPIError):
    """A hard cost ceiling has been hit; orchestration must stop."""
