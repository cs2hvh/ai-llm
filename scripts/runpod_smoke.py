#!/usr/bin/env python3
"""Phase-0 RunPod orchestration smoke test.

Launches a tiny pod, waits for ``RUNNING``, prints pod metadata, terminates.
Validates the orchestration end-to-end (auth, capacity, lifecycle, cost
accounting). Does NOT exec into the pod or read its logs — that's a
follow-up; this is the minimum viable orchestration smoke.

Cost ceiling: $0.50 (A40 pod for ~5 min — A10 was retired from RunPod's
catalog, A40 is the current cheapest 48GB option). Hard-stops if the cost
ledger would exceed it.

Usage:
    set -a && source .env && set +a
    python scripts/runpod_smoke.py [--sku A40] [--ledger artifacts/smoke_ledger.jsonl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.runpod_orch import CostLedger, GPUSku, PodSpec  # noqa: E402
from myllm.runpod_orch.client import RunPodClient  # noqa: E402
from myllm.runpod_orch.lifecycle import PodLifecycle  # noqa: E402
from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sku",
        choices=[s.name for s in GPUSku],
        default="A40",
        help="GPU SKU. Default A40 (~$0.40/hr); cheapest available since A10 was retired.",
    )
    p.add_argument(
        "--ledger",
        default="artifacts/smoke_ledger.jsonl",
        help="Path to persistent cost ledger.",
    )
    p.add_argument(
        "--ceiling-usd",
        type=float,
        default=0.50,
        help="Hard cost ceiling for this run (default $0.50).",
    )
    p.add_argument(
        "--name",
        default="myllm-smoke",
        help="Pod name shown in the RunPod console.",
    )
    p.add_argument(
        "--image",
        default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        help="Image to launch. Public RunPod pytorch image is fine for a smoke.",
    )
    p.add_argument(
        "--cloud-type",
        choices=["SECURE", "COMMUNITY"],
        default="COMMUNITY",
        help="Secure Cloud has better interconnect; Community is cheaper.",
    )
    args = p.parse_args()
    configure_logging()

    sku = GPUSku[args.sku]
    spec = PodSpec(
        name=args.name,
        gpu=sku,
        gpu_count=1,
        image=args.image,
        container_disk_gb=20,
        volume_gb=0,
        cloud_type=args.cloud_type,
    )
    log.info(
        "smoke_plan",
        sku=sku.name,
        hourly_usd=spec.estimated_hourly_cost_usd(),
        ceiling_usd=args.ceiling_usd,
    )

    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(path=args.ledger, ceiling_usd=args.ceiling_usd)

    client = RunPodClient()
    lifecycle = PodLifecycle(
        client=client,
        ledger=ledger,
        poll_interval_seconds=10.0,
        ready_timeout_seconds=600.0,
    )

    try:
        with lifecycle.launch(spec) as pod:
            machine = pod.get("machine") or {}
            runtime = pod.get("runtime") or {}
            runtime_ports = runtime.get("ports") or []
            public_ip = next(
                (p.get("ip") for p in runtime_ports if p.get("isIpPublic")),
                None,
            )
            log.info(
                "pod_running",
                pod_id=pod.get("id"),
                name=pod.get("name"),
                gpu_display=machine.get("gpuDisplayName"),
                cost_per_hr_reported=pod.get("costPerHr"),
                public_ip=public_ip,
                runtime_ports=len(runtime_ports),
            )
            log.info("smoke_success_reached_running_state")
        log.info(
            "smoke_complete",
            spent_usd=round(ledger.total_usd, 6),
            ceiling_usd=args.ceiling_usd,
        )
    except Exception as e:
        log.error("smoke_failed", error_type=type(e).__name__, error=str(e))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
