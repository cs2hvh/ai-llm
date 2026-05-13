"""Release scorecard — single-pass evaluation of a checkpoint across the
v1 benchmark set, producing both JSON (machine-readable) and Markdown
(drop-in for model_card_v1.md) outputs.

Run via ``scripts/build_release_scorecard.py``. The split here is the
usual lib-vs-CLI: this module is import-safe + testable with mocked
benchmark adapters; the script handles checkpoint loading + predict_fn
construction (which needs the JAX runtime).

Design:
  - A ``ScorecardBuilder`` accepts a list of ``Benchmark`` adapters +
    a ``PredictFn`` and produces a ``Scorecard`` aggregate.
  - The aggregate carries: per-benchmark accuracy + n_total + per-
    subgroup breakdown + metadata (model checkpoint path, eval
    timestamp, sample size, seed).
  - ``Scorecard.to_markdown()`` formats the same data as the
    "Evaluation" section of model_card_v1.md so the operator just
    pastes it in at release time.
  - Failures on individual benchmarks are recorded as errors but do NOT
    abort the rest. We'd rather get 9/10 benchmark scores than 0.

Why CLI-separate: building a predict_fn from a checkpoint requires JAX
runtime + the model architecture. Keeping that out of the lib lets us
unit-test the scorecard math without touching JAX.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from myllm.eval.runner import run_benchmark
from myllm.eval.types import Benchmark, EvalResult, PredictFn
from myllm.utils import get_logger

log = get_logger(__name__)


@dataclass
class BenchmarkScore:
    """One benchmark's row on the scorecard."""
    benchmark: str
    accuracy: float | None             # None if eval errored
    n_total: int
    per_subgroup: dict[str, float] = field(default_factory=dict)
    error: str | None = None           # set if eval failed; accuracy will be None


@dataclass
class Scorecard:
    """Full scorecard for one checkpoint."""
    model_checkpoint: str
    model_name: str
    eval_timestamp_utc: str
    sample_size_per_benchmark: int | None
    seed: int
    scores: list[BenchmarkScore] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_checkpoint": self.model_checkpoint,
            "model_name": self.model_name,
            "eval_timestamp_utc": self.eval_timestamp_utc,
            "sample_size_per_benchmark": self.sample_size_per_benchmark,
            "seed": self.seed,
            "scores": [
                {
                    "benchmark": s.benchmark,
                    "accuracy": s.accuracy,
                    "n_total": s.n_total,
                    "per_subgroup": s.per_subgroup,
                    "error": s.error,
                }
                for s in self.scores
            ],
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Render in a format suitable for pasting into
        docs/governance/model_card_v1.md's Evaluation section."""
        lines = [
            f"# Release Scorecard — {self.model_name}",
            "",
            f"**Checkpoint**: `{self.model_checkpoint}`",
            f"**Eval timestamp** (UTC): {self.eval_timestamp_utc}",
            f"**Sample size per benchmark**: "
            f"{self.sample_size_per_benchmark if self.sample_size_per_benchmark is not None else 'full'}",
            f"**Seed**: {self.seed}",
            "",
            "## Headline scores",
            "",
            "| Benchmark | Accuracy | N | Notes |",
            "|---|---|---|---|",
        ]
        for s in self.scores:
            if s.error:
                acc = "—"
                note = f"FAILED: {s.error[:80]}"
            else:
                acc = f"{s.accuracy * 100:.2f}%" if s.accuracy is not None else "—"
                note = ""
                if s.per_subgroup:
                    sub_parts = [f"{k}={v * 100:.1f}%" for k, v in sorted(s.per_subgroup.items())]
                    note = "  ".join(sub_parts)
            lines.append(f"| {s.benchmark} | {acc} | {s.n_total} | {note} |")
        if self.notes:
            lines.extend(["", "## Notes", "", self.notes])
        return "\n".join(lines) + "\n"


def build_scorecard(
    *,
    model_checkpoint: str,
    model_name: str,
    benchmarks: list[Benchmark],
    predict_fn: PredictFn,
    sample_size_per_benchmark: int | None = None,
    seed: int = 0,
    notes: str = "",
) -> Scorecard:
    """Run every benchmark via ``run_benchmark`` and aggregate into one
    scorecard. Failures on individual benchmarks are caught so the rest
    still run."""
    card = Scorecard(
        model_checkpoint=model_checkpoint,
        model_name=model_name,
        eval_timestamp_utc=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sample_size_per_benchmark=sample_size_per_benchmark,
        seed=seed,
        notes=notes,
    )

    for bench in benchmarks:
        log.info("scorecard_running_benchmark", benchmark=bench.name)
        try:
            result: EvalResult = run_benchmark(
                bench,
                predict_fn,
                sample_size=sample_size_per_benchmark,
                seed=seed,
            )
            per_sub = {k: r.accuracy for k, r in result.per_subgroup.items()}
            card.scores.append(
                BenchmarkScore(
                    benchmark=result.benchmark,
                    accuracy=result.accuracy,
                    n_total=result.n_total,
                    per_subgroup=per_sub,
                )
            )
            log.info(
                "scorecard_benchmark_done",
                benchmark=bench.name,
                accuracy=round(result.accuracy, 4),
                n=result.n_total,
            )
        except Exception as e:  # noqa: BLE001 — keep iterating
            log.warning(
                "scorecard_benchmark_failed",
                benchmark=bench.name,
                error=str(e),
            )
            card.scores.append(
                BenchmarkScore(
                    benchmark=getattr(bench, "name", "unknown"),
                    accuracy=None,
                    n_total=0,
                    error=f"{type(e).__name__}: {e}"[:300],
                )
            )

    return card


def write_scorecard(
    card: Scorecard,
    output_dir: str | Path,
    *,
    name_prefix: str = "scorecard",
) -> tuple[Path, Path]:
    """Write the scorecard to disk as JSON + Markdown. Returns the two paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name_prefix}.json"
    md_path = output_dir / f"{name_prefix}.md"
    json_path.write_text(card.to_json())
    md_path.write_text(card.to_markdown())
    log.info("scorecard_written", json=str(json_path), md=str(md_path))
    return json_path, md_path
