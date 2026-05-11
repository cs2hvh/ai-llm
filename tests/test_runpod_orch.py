"""Unit tests for the RunPod orchestration layer.

These are pure-Python tests: they validate spec construction, cost-ledger
arithmetic, and the lifecycle context manager via fakes. The real RunPod SDK
is mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from myllm.runpod_orch import CostLedger, GPUSku, PodSpec
from myllm.runpod_orch.lifecycle import PodLifecycle
from myllm.runpod_orch.spec import GPU_HOURLY_USD
from myllm.utils.exceptions import CostCeilingExceeded, RunPodLaunchError


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
class TestPodSpec:
    def test_basic_construction(self):
        spec = PodSpec(name="pilot-1", gpu=GPUSku.H100_SXM, gpu_count=8)
        assert spec.gpu_count == 8
        assert spec.estimated_hourly_cost_usd() == 8 * GPU_HOURLY_USD[GPUSku.H100_SXM]

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            PodSpec(name="x", gpu=GPUSku.A40, gpu_count=1, mystery_field=1)  # type: ignore

    def test_gpu_count_bounds(self):
        with pytest.raises(Exception):
            PodSpec(name="x", gpu=GPUSku.A40, gpu_count=0)
        with pytest.raises(Exception):
            PodSpec(name="x", gpu=GPUSku.A40, gpu_count=9)

    def test_runpod_payload_shape(self):
        spec = PodSpec(name="t", gpu=GPUSku.A100_80GB, gpu_count=2)
        payload = spec.to_runpod_payload()
        assert payload["gpu_type_id"] == GPUSku.A100_80GB.value
        assert payload["gpu_count"] == 2
        assert "ports" in payload


# --------------------------------------------------------------------------- #
# Cost ledger
# --------------------------------------------------------------------------- #
class TestCostLedger:
    def test_open_close_records_cost(self, tmp_path: Path, monkeypatch):
        ledger = CostLedger(path=str(tmp_path / "ledger.jsonl"), ceiling_usd=100.0)
        # Patch time.time() to control duration.
        import time as _time

        seq = iter([1000.0, 1000.0 + 3600.0])  # 1 hour
        monkeypatch.setattr(_time, "time", lambda: next(seq))
        ledger.open_pod("p1", hourly_rate_usd=2.0, pod_name="t")
        ev = ledger.close_pod("p1", note="done")
        assert ev.delta_usd == pytest.approx(2.0, rel=1e-6)
        assert ledger.total_usd == pytest.approx(2.0, rel=1e-6)

    def test_persists_across_instances(self, tmp_path: Path, monkeypatch):
        path = str(tmp_path / "ledger.jsonl")
        import time as _time

        seq = iter([1000.0, 1000.0 + 1800.0])  # 30 min
        monkeypatch.setattr(_time, "time", lambda: next(seq))
        ledger = CostLedger(path=path, ceiling_usd=50.0)
        ledger.open_pod("p1", 4.0, "t")
        ledger.close_pod("p1")

        ledger2 = CostLedger(path=path, ceiling_usd=50.0)
        assert ledger2.total_usd == pytest.approx(2.0, rel=1e-6)

    def test_ceiling_blocks_new_open(self, tmp_path: Path):
        path = str(tmp_path / "ledger.jsonl")
        ledger = CostLedger(path=path, ceiling_usd=10.0)
        ledger.manual_charge(15.0, "test")
        with pytest.raises(CostCeilingExceeded):
            ledger.open_pod("p1", 2.0, "t")

    def test_invalid_ceiling(self, tmp_path: Path):
        with pytest.raises(ValueError):
            CostLedger(path=str(tmp_path / "l.jsonl"), ceiling_usd=0)

    def test_ledger_is_jsonl(self, tmp_path: Path, monkeypatch):
        path = str(tmp_path / "ledger.jsonl")
        import time as _time

        seq = iter([1000.0, 1000.0 + 3600.0])
        monkeypatch.setattr(_time, "time", lambda: next(seq))
        ledger = CostLedger(path=path, ceiling_usd=100.0)
        ledger.open_pod("p1", 1.0, "t")
        ledger.close_pod("p1")
        # Each line is a valid JSON object.
        for line in Path(path).read_text().splitlines():
            assert json.loads(line)


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
class TestPodLifecycle:
    def _make_client(self, statuses: list[str]):
        """Each status string becomes one get_pod response.

        ``runtime`` is populated only for RUNNING (mirrors RunPod's GraphQL
        contract) so the lifecycle's actual-allocation check exercises both
        the "queued but desiredStatus=RUNNING" trap and the success path.
        """
        client = MagicMock()
        client.create_pod.return_value = {"id": "pod-1", "name": "t"}
        responses = []
        for s in statuses:
            resp: dict = {"desiredStatus": s, "runtime": None}
            if s == "RUNNING":
                resp["runtime"] = {"ports": [{"ip": "1.2.3.4", "type": "tcp"}]}
            responses.append(resp)
        client.get_pod.side_effect = responses
        return client

    def test_happy_path_terminates(self, tmp_path: Path, monkeypatch):
        client = self._make_client(["PENDING", "RUNNING"])
        ledger = CostLedger(path=str(tmp_path / "l.jsonl"), ceiling_usd=100.0)
        lc = PodLifecycle(client=client, ledger=ledger, poll_interval_seconds=0.0)
        # Avoid time.sleep delay.
        monkeypatch.setattr("time.sleep", lambda _: None)

        spec = PodSpec(name="t", gpu=GPUSku.A40, gpu_count=1)
        with lc.launch(spec) as pod:
            assert pod["desiredStatus"] == "RUNNING"
        client.terminate_pod.assert_called_once_with("pod-1")

    def test_terminal_status_raises(self, tmp_path: Path, monkeypatch):
        client = self._make_client(["FAILED"])
        ledger = CostLedger(path=str(tmp_path / "l.jsonl"), ceiling_usd=100.0)
        lc = PodLifecycle(client=client, ledger=ledger, poll_interval_seconds=0.0)
        monkeypatch.setattr("time.sleep", lambda _: None)

        spec = PodSpec(name="t", gpu=GPUSku.A40, gpu_count=1)
        with pytest.raises(RunPodLaunchError):
            with lc.launch(spec):
                pass
        client.terminate_pod.assert_called_once_with("pod-1")

    def test_budget_blocks_launch(self, tmp_path: Path):
        client = MagicMock()
        ledger = CostLedger(path=str(tmp_path / "l.jsonl"), ceiling_usd=0.50)
        lc = PodLifecycle(client=client, ledger=ledger)
        spec = PodSpec(name="t", gpu=GPUSku.H100_SXM, gpu_count=8)
        with pytest.raises(RunPodLaunchError):
            with lc.launch(spec):
                pass
        client.create_pod.assert_not_called()
