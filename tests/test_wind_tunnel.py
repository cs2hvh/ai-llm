"""Unit tests for ``scripts/wind_tunnel_sweep.py``.

These exist to keep the sweep launchable without surprises — a single
bug in argv generation or loss-parsing could waste a $30-50 cloud run.

Covers:
  - ``build_grid`` produces the expected 10-cell grid with stable ids
  - ``cell_command`` emits the right argv (override flags reach run_pretrain)
  - ``parse_final_loss`` extracts the last loss value from a structlog stream
  - ``select_best`` picks the lowest-loss done cell, ignores failed/pending
  - The default ``tokens_per_cell`` matches the documented economics
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# The sweep is a script, not a package — import it explicitly.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "wind_tunnel_sweep.py"
_spec = importlib.util.spec_from_file_location("wind_tunnel_sweep", _SCRIPT_PATH)
wt = importlib.util.module_from_spec(_spec)
sys.modules["wind_tunnel_sweep"] = wt
_spec.loader.exec_module(wt)


# --------------------------------------------------------------------------- #
# Grid construction
# --------------------------------------------------------------------------- #
class TestBuildGrid:
    def test_default_grid_has_ten_cells(self):
        cells = wt.build_grid()
        assert len(cells) == 10

    def test_grid_is_lr_x_init_product(self):
        cells = wt.build_grid(
            lr_grid=(1e-3, 2e-3, 4e-3),
            init_grid=(0.01, 0.02),
        )
        assert len(cells) == 6
        pairs = {(c.peak_lr, c.init_std) for c in cells}
        assert pairs == {
            (1e-3, 0.01), (1e-3, 0.02),
            (2e-3, 0.01), (2e-3, 0.02),
            (4e-3, 0.01), (4e-3, 0.02),
        }

    def test_cell_ids_are_unique(self):
        cells = wt.build_grid()
        ids = {c.cell_id for c in cells}
        assert len(ids) == len(cells)

    def test_cell_ids_filesystem_safe(self):
        """Cell ids become directory names; must not contain shell metachars."""
        cells = wt.build_grid()
        bad = set("/\\: \t")
        for c in cells:
            assert not any(ch in c.cell_id for ch in bad), c.cell_id

    def test_tokens_per_cell_threaded_through(self):
        cells = wt.build_grid(tokens_per_cell=42_000_000)
        assert all(c.tokens == 42_000_000 for c in cells)


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #
class TestCellCommand:
    def _make_cell(self) -> "wt.SweepCell":
        return wt.SweepCell(
            cell_id="lr1e-3_init2e-2",
            peak_lr=1.0e-3,
            init_std=0.02,
            tokens=200_000_000,
        )

    def test_argv_includes_override_flags(self):
        """The whole point of the sweep is per-cell HP overrides — verify
        they actually appear in the argv we'll subprocess.run()."""
        cmd = wt.cell_command(
            self._make_cell(),
            model_config="cfg.yaml",
            data_config="data.yaml",
            tokenizer_path="tok.json",
            checkpoint_root="ckpts",
            log_path="log.txt",
        )
        assert "--peak-lr-override" in cmd
        assert "--init-std-override" in cmd
        # Values should be parsable floats matching the cell.
        lr_idx = cmd.index("--peak-lr-override")
        init_idx = cmd.index("--init-std-override")
        assert float(cmd[lr_idx + 1]) == pytest.approx(1.0e-3)
        assert float(cmd[init_idx + 1]) == pytest.approx(0.02)

    def test_argv_disables_wandb(self):
        """Sweep cells must not log to W&B — they'd flood the project with
        10 disposable runs."""
        cmd = wt.cell_command(
            self._make_cell(), "c.yaml", "d.yaml", "t.json", "ckpts", "log.txt"
        )
        assert "--no-wandb" in cmd

    def test_argv_includes_run_name_with_cell_id(self):
        cmd = wt.cell_command(
            self._make_cell(), "c.yaml", "d.yaml", "t.json", "ckpts", "log.txt"
        )
        rn_idx = cmd.index("--run-name")
        assert "lr1e-3_init2e-2" in cmd[rn_idx + 1]

    def test_total_steps_derived_from_tokens(self):
        """tokens_per_cell / (micro_batch × seq_len) = total_steps."""
        cell = self._make_cell()  # tokens = 200M
        cmd = wt.cell_command(
            cell, "c.yaml", "d.yaml", "t.json", "ckpts", "log.txt",
            micro_batch_per_device=8,
            sequence_length=2048,
        )
        steps_idx = cmd.index("--total-steps")
        # 200M / (8 × 2048) = 200M / 16384 ≈ 12207
        assert int(cmd[steps_idx + 1]) == pytest.approx(12207, abs=1)


# --------------------------------------------------------------------------- #
# Loss-parsing — robustness matters because a misparsed log silently
# corrupts the sweep's optimum selection.
# --------------------------------------------------------------------------- #
class TestParseFinalLoss:
    def test_parses_last_loss_from_structlog(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text(
            '{"event": "step", "step": 100, "loss": 6.123, "lr_mult": 1.0}\n'
            '{"event": "step", "step": 200, "loss": 4.876, "lr_mult": 1.0}\n'
            '{"event": "step", "step": 300, "loss": 3.412, "lr_mult": 1.0}\n'
        )
        assert wt.parse_final_loss(log) == pytest.approx(3.412)

    def test_returns_none_for_missing_file(self, tmp_path):
        assert wt.parse_final_loss(tmp_path / "nope.log") is None

    def test_returns_none_for_log_without_loss(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text('{"event": "startup", "step": 0}\n')
        assert wt.parse_final_loss(log) is None

    def test_handles_scientific_notation(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text('{"loss": 1.5e-2}\n')
        assert wt.parse_final_loss(log) == pytest.approx(0.015)

    def test_ignores_malformed_loss_values(self, tmp_path):
        """If a line has 'loss: NaN-ish-text', we should skip it and
        return the last well-formed value."""
        log = tmp_path / "train.log"
        log.write_text(
            '{"loss": 2.0}\n'
            'random line without json\n'
            '{"loss": 1.5}\n'
        )
        assert wt.parse_final_loss(log) == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# Best-cell selection
# --------------------------------------------------------------------------- #
class TestSelectBest:
    def test_picks_lowest_loss(self):
        cells = [
            wt.SweepCell("a", 1e-3, 0.01, 1, final_loss=3.5, status="done"),
            wt.SweepCell("b", 2e-3, 0.01, 1, final_loss=2.8, status="done"),
            wt.SweepCell("c", 4e-3, 0.01, 1, final_loss=4.1, status="done"),
        ]
        best = wt.select_best(cells)
        assert best.cell_id == "b"

    def test_ignores_failed_cells(self):
        """A failed cell with a stale partial loss must NOT be selected."""
        cells = [
            wt.SweepCell("a", 1e-3, 0.01, 1, final_loss=1.0, status="failed"),
            wt.SweepCell("b", 2e-3, 0.01, 1, final_loss=2.5, status="done"),
        ]
        best = wt.select_best(cells)
        assert best.cell_id == "b"

    def test_returns_none_if_no_done_cells(self):
        cells = [
            wt.SweepCell("a", 1e-3, 0.01, 1, status="pending"),
            wt.SweepCell("b", 2e-3, 0.01, 1, status="running"),
        ]
        assert wt.select_best(cells) is None

    def test_returns_none_for_empty_list(self):
        assert wt.select_best([]) is None


# --------------------------------------------------------------------------- #
# Cost-economics guard — keeps the docs honest.
# --------------------------------------------------------------------------- #
class TestEconomicsGuards:
    def test_default_tokens_per_cell_is_200m(self):
        """Earlier version defaulted to 1B → ~$200 sweep. We dropped to 200M
        per μP literature. Regressing back would silently 5× the cost."""
        cells = wt.build_grid()
        assert all(c.tokens == 200_000_000 for c in cells)

    def test_grid_size_matches_documented_cost(self):
        """Doc claims ~10 cells × $3-5 = $30-50. If grid grows, update docs."""
        assert len(wt.build_grid()) == 10
