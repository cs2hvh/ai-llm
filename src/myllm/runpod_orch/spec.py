"""Pod specification: GPU SKUs and PodSpec model.

We restrict GPU choice to a small allow-list so callers can't accidentally
launch ``RTX_3060`` for pretraining. New SKUs are added intentionally with
an updated cost.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GPUSku(str, Enum):
    """Allow-listed GPU SKUs.

    Values are the exact ``id`` strings RunPod's ``get_gpus()`` returns —
    verified against the live API on 2026-05-10. Update this enum if
    RunPod renames a SKU; ``runpod.get_gpus()`` is the source of truth.
    A10 was removed from RunPod's catalog; A40 is the cheapest 48GB SKU
    we keep around for orchestration smokes.
    """

    A40 = "NVIDIA A40"                       # 48GB, smoke + small data-prep
    A100_40GB = "NVIDIA A100-SXM4-40GB"      # 40GB SXM4
    A100_80GB = "NVIDIA A100-SXM4-80GB"      # 80GB SXM4, mid-tier training
    A100_PCIE_80GB = "NVIDIA A100 80GB PCIe" # 80GB PCIe
    H100_PCIE = "NVIDIA H100 PCIe"           # 80GB
    H100_SXM = "NVIDIA H100 80GB HBM3"       # 80GB SXM, primary 1B training SKU
    H100_NVL = "NVIDIA H100 NVL"             # 94GB NVL
    H200_SXM = "NVIDIA H200"                 # 141GB
    B200 = "NVIDIA B200"                     # 180GB, fastest at training
    L40S = "NVIDIA L40S"                     # 48GB, inference / quantize
    RTX_4090 = "NVIDIA GeForce RTX 4090"     # 24GB, dev / inference smoke
    RTX_A4000 = "NVIDIA RTX A4000"           # 16GB, cheapest dev


# Hourly USD rates — used ONLY by the cost ledger ceiling. Actual billing
# is on RunPod's side; these are intentionally rounded toward the higher of
# (secure, community) prices so ceilings never under-estimate.
#
# Last refreshed 2026-05-11 via live `runpod.get_gpu(...)` calls. Live deltas
# from prior 2026-05-10 entries:
#   - H200_SXM: 3.40 → 3.99 (secure cloud actual; was under-estimating)
#   - B200:     4.99 → 5.98 (community cloud actual; was under-estimating)
#   - H100_PCIE: 2.40 → 2.39 (close enough; kept higher)
#   - Others within 5% — no change needed.
GPU_HOURLY_USD: dict[GPUSku, float] = {
    GPUSku.A40: 0.40,
    GPUSku.A100_40GB: 1.10,
    GPUSku.A100_80GB: 1.49,        # live secure $1.49 / community $1.39
    GPUSku.A100_PCIE_80GB: 1.39,   # live secure $1.39 / community $1.19
    GPUSku.H100_PCIE: 2.39,        # live secure $2.39 / community $1.99
    GPUSku.H100_SXM: 2.99,         # live secure $2.99 / community $2.69
    GPUSku.H100_NVL: 2.60,
    GPUSku.H200_SXM: 3.99,         # live secure $3.99 / community $3.59
    GPUSku.B200: 5.98,             # live community $5.98 / secure $5.49 — pick higher
    GPUSku.L40S: 0.86,
    GPUSku.RTX_4090: 0.55,
    GPUSku.RTX_A4000: 0.20,
}


class PodSpec(BaseModel):
    """Declarative pod-launch request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64)
    gpu: GPUSku
    gpu_count: int = Field(ge=1, le=8)
    image: str = Field(default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    container_disk_gb: int = Field(default=100, ge=20, le=2000)
    volume_gb: int = Field(default=0, ge=0, le=10000)
    volume_mount_path: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    ports_tcp: tuple[int, ...] = (22,)
    bid_per_gpu_usd: float | None = None  # None = on-demand; set for spot
    cloud_type: str = Field(default="SECURE", pattern=r"^(SECURE|COMMUNITY)$")
    region_priority: tuple[str, ...] = ()  # empty = any region

    def estimated_hourly_cost_usd(self) -> float:
        return GPU_HOURLY_USD[self.gpu] * self.gpu_count

    def to_runpod_payload(self) -> dict[str, Any]:
        """Translate to the dict the runpod SDK expects.

        Kept here (not in client) so it's testable without the SDK installed.
        Note: runpod SDK 1.9 ``create_pod`` does not accept ``bid_per_gpu`` —
        spot/bid pricing requires the GraphQL API directly. ``bid_per_gpu_usd``
        is retained on the spec for record-keeping; we'll wire spot through a
        separate path when we actually need it.
        """
        return {
            "name": self.name,
            "image_name": self.image,
            "gpu_type_id": self.gpu.value,
            "gpu_count": self.gpu_count,
            "container_disk_in_gb": self.container_disk_gb,
            "volume_in_gb": self.volume_gb,
            "volume_mount_path": self.volume_mount_path,
            "ports": ",".join(f"{p}/tcp" for p in self.ports_tcp),
            "env": self.env,
            "cloud_type": self.cloud_type,
        }
