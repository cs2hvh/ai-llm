"""Thin retry-wrapped client around the RunPod Python SDK.

Why we wrap:
    - The SDK raises generic exceptions; we map them onto our hierarchy.
    - We add tenacity retries for transient errors (timeouts, 5xx).
    - We add structured logging at every API boundary.
    - We can swap the SDK for direct HTTP later if the SDK gets in the way.

Construction reads ``RUNPOD_API_KEY`` from env and validates it is set
before issuing any call.
"""
from __future__ import annotations

import os
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from myllm.utils import get_logger
from myllm.utils.exceptions import RunPodAPIError, RunPodLaunchError

log = get_logger(__name__)


class RunPodClient:
    """Wrapped RunPod SDK client. Retries + structured errors + logging."""

    def __init__(self, api_key: str | None = None) -> None:
        api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise RunPodAPIError("RUNPOD_API_KEY not set")
        try:
            import runpod
        except ImportError as e:
            raise ImportError(
                "runpod SDK not installed; install with `pip install runpod`"
            ) from e
        runpod.api_key = api_key
        self._sdk = runpod

    # ------------------------------------------------------------------ #
    # GPU types
    # ------------------------------------------------------------------ #
    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def list_gpu_types(self) -> list[dict[str, Any]]:
        try:
            return self._sdk.get_gpus()
        except Exception as e:
            raise RunPodAPIError(f"list_gpu_types failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Pod lifecycle
    # ------------------------------------------------------------------ #
    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        log.info("runpod_create_pod", name=payload.get("name"), gpu=payload.get("gpu_type_id"))
        try:
            pod = self._sdk.create_pod(**payload)
        except Exception as e:
            raise RunPodLaunchError(f"create_pod failed: {e}") from e
        if not isinstance(pod, dict) or "id" not in pod:
            raise RunPodLaunchError(f"create_pod returned malformed response: {pod!r}")
        log.info("runpod_pod_created", pod_id=pod["id"], name=payload.get("name"))
        return pod

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        try:
            return self._sdk.get_pod(pod_id)
        except Exception as e:
            raise RunPodAPIError(f"get_pod({pod_id}) failed: {e}") from e

    def stop_pod(self, pod_id: str) -> None:
        log.info("runpod_stop_pod", pod_id=pod_id)
        try:
            self._sdk.stop_pod(pod_id)
        except Exception as e:
            raise RunPodAPIError(f"stop_pod({pod_id}) failed: {e}") from e

    def terminate_pod(self, pod_id: str) -> None:
        log.info("runpod_terminate_pod", pod_id=pod_id)
        try:
            self._sdk.terminate_pod(pod_id)
        except Exception as e:
            raise RunPodAPIError(f"terminate_pod({pod_id}) failed: {e}") from e
