"""Cost ledger.

Tracks cumulative spend across pods and enforces hard ceilings. Persisted
as line-delimited JSON so a process restart sees the running total. The
ledger is the source of truth — RunPod's billing is authoritative for the
invoice but we keep our own to make the ceiling check fast and to detect
drift between expected and actual spend.

Each event is one of:
    pod_started        — pod went into RUNNING state (timer starts)
    pod_stopped        — pod stopped (timer ends, charge accrued)
    manual_charge      — manual adjustment (e.g., reconciling with invoice)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from myllm.utils import get_logger
from myllm.utils.exceptions import CostCeilingExceeded
from myllm.utils.io import atomic_write_text

log = get_logger(__name__)


@dataclass(frozen=True)
class CostEvent:
    timestamp: float
    kind: str
    pod_id: str
    pod_name: str
    hourly_rate_usd: float
    duration_seconds: float
    delta_usd: float
    note: str = ""


@dataclass
class CostLedger:
    """Persistent cost ledger with a hard ceiling."""

    path: str
    ceiling_usd: float
    pause_at_fraction: float = 0.90
    _events: list[CostEvent] = field(init=False, repr=False, default_factory=list)
    _open_pods: dict[str, tuple[float, float, str]] = field(
        init=False, repr=False, default_factory=dict
    )  # pod_id -> (start_ts, hourly_rate, name)

    def __post_init__(self) -> None:
        if self.ceiling_usd <= 0:
            raise ValueError("ceiling_usd must be positive")
        if not 0 < self.pause_at_fraction <= 1:
            raise ValueError("pause_at_fraction must be in (0, 1]")
        p = Path(self.path)
        if p.exists():
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    self._events.append(CostEvent(**data))
                except (json.JSONDecodeError, TypeError) as e:
                    log.warning("cost_ledger_skipped_bad_line", error=str(e))

    @property
    def total_usd(self) -> float:
        return sum(e.delta_usd for e in self._events)

    def remaining_usd(self) -> float:
        return self.ceiling_usd - self.total_usd

    def warn_if_near_ceiling(self) -> None:
        frac = self.total_usd / self.ceiling_usd
        if frac >= self.pause_at_fraction:
            log.warning(
                "cost_ledger_near_ceiling",
                spent_usd=self.total_usd,
                ceiling_usd=self.ceiling_usd,
                fraction=frac,
            )

    def assert_below_ceiling(self) -> None:
        if self.total_usd >= self.ceiling_usd:
            raise CostCeilingExceeded(
                f"spent ${self.total_usd:.2f} >= ceiling ${self.ceiling_usd:.2f}"
            )

    def open_pod(self, pod_id: str, hourly_rate_usd: float, pod_name: str) -> None:
        self.assert_below_ceiling()
        if pod_id in self._open_pods:
            log.warning("cost_ledger_double_open", pod_id=pod_id)
            return
        self._open_pods[pod_id] = (time.time(), hourly_rate_usd, pod_name)
        log.info(
            "cost_ledger_pod_opened",
            pod_id=pod_id,
            pod_name=pod_name,
            hourly_rate_usd=hourly_rate_usd,
        )

    def close_pod(self, pod_id: str, note: str = "") -> CostEvent:
        if pod_id not in self._open_pods:
            raise KeyError(f"pod_id not open in ledger: {pod_id}")
        start_ts, rate, name = self._open_pods.pop(pod_id)
        now = time.time()
        duration = now - start_ts
        delta = rate * (duration / 3600.0)
        event = CostEvent(
            timestamp=now,
            kind="pod_stopped",
            pod_id=pod_id,
            pod_name=name,
            hourly_rate_usd=rate,
            duration_seconds=duration,
            delta_usd=delta,
            note=note,
        )
        self._append(event)
        log.info(
            "cost_ledger_pod_closed",
            pod_id=pod_id,
            duration_seconds=duration,
            delta_usd=delta,
            cumulative_usd=self.total_usd,
        )
        self.warn_if_near_ceiling()
        return event

    def manual_charge(self, amount_usd: float, note: str) -> None:
        event = CostEvent(
            timestamp=time.time(),
            kind="manual_charge",
            pod_id="",
            pod_name="",
            hourly_rate_usd=0.0,
            duration_seconds=0.0,
            delta_usd=amount_usd,
            note=note,
        )
        self._append(event)

    def _append(self, event: CostEvent) -> None:
        self._events.append(event)
        # Append-only file — one JSON record per line.
        line = json.dumps(asdict(event), sort_keys=True)
        existing = ""
        p = Path(self.path)
        if p.exists():
            existing = p.read_text()
        atomic_write_text(self.path, existing + line + "\n")
