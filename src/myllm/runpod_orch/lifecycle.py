"""Pod-lifecycle orchestration: launch → wait-for-ready → use → terminate.

The high-level flow is a context manager so cleanup is guaranteed even on
exceptions:

    with PodLifecycle(client, ledger).launch(spec) as pod:
        ...                          # use pod['ssh_endpoint']
    # pod is terminated on exit; ledger updated.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from myllm.runpod_orch.client import RunPodClient
from myllm.runpod_orch.cost import CostLedger
from myllm.runpod_orch.spec import GPU_HOURLY_USD, PodSpec
from myllm.utils import get_logger
from myllm.utils.exceptions import RunPodAPIError, RunPodLaunchError

log = get_logger(__name__)


@dataclass
class PodLifecycle:
    """Orchestrate pod launch → wait → terminate, recording cost."""

    client: RunPodClient
    ledger: CostLedger
    poll_interval_seconds: float = 10.0
    ready_timeout_seconds: float = 1200.0  # 20 min

    @contextmanager
    def launch(self, spec: PodSpec) -> Iterator[dict[str, Any]]:
        """Launch a pod, wait for ``RUNNING``, yield, terminate on exit."""
        # Pre-flight: budget check.
        self.ledger.assert_below_ceiling()
        est_cost_per_hour = spec.estimated_hourly_cost_usd()
        if est_cost_per_hour > self.ledger.remaining_usd():
            raise RunPodLaunchError(
                f"insufficient budget: pod costs ${est_cost_per_hour:.2f}/hr, "
                f"remaining ${self.ledger.remaining_usd():.2f}"
            )

        pod = self.client.create_pod(spec.to_runpod_payload())
        pod_id = pod["id"]
        self.ledger.open_pod(
            pod_id=pod_id,
            hourly_rate_usd=GPU_HOURLY_USD[spec.gpu] * spec.gpu_count,
            pod_name=spec.name,
        )

        try:
            ready_pod = self._wait_for_ready(pod_id)
            log.info("pod_ready", pod_id=pod_id, name=spec.name)
            yield ready_pod
        finally:
            self._safe_terminate(pod_id)
            try:
                self.ledger.close_pod(pod_id, note=f"spec={spec.name}")
            except KeyError:
                # Already closed — ignore.
                pass

    def _wait_for_ready(self, pod_id: str) -> dict[str, Any]:
        """Wait for the pod to actually be allocated, not just queued.

        ``desiredStatus`` is what we asked RunPod for — it flips to RUNNING
        the instant the API accepts the create call, even before any host
        has been picked. The reliable signal that the pod has a live
        runtime is ``runtime`` being non-null in the GraphQL response (it
        starts as ``None`` while the scheduler is still working).
        """
        deadline = time.time() + self.ready_timeout_seconds
        last_status: str = ""
        while time.time() < deadline:
            pod = self.client.get_pod(pod_id)
            desired = (pod.get("desiredStatus") or "").upper()
            runtime = pod.get("runtime")
            if desired in {"FAILED", "ERROR", "TERMINATED", "EXITED"}:
                raise RunPodLaunchError(
                    f"pod {pod_id} entered terminal desiredStatus: {desired}"
                )
            if desired == "RUNNING" and runtime is not None:
                return pod
            new_status = f"{desired}|runtime={'set' if runtime else 'null'}"
            if new_status != last_status:
                log.info("pod_status", pod_id=pod_id, status=new_status)
                last_status = new_status
            time.sleep(self.poll_interval_seconds)
        raise RunPodLaunchError(
            f"pod {pod_id} did not become RUNNING (with runtime) within "
            f"{self.ready_timeout_seconds}s"
        )

    def _safe_terminate(self, pod_id: str) -> None:
        try:
            self.client.terminate_pod(pod_id)
        except RunPodAPIError as e:
            # Surface it but don't mask the original exception (if any).
            log.error("pod_terminate_failed", pod_id=pod_id, error=str(e))
