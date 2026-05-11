"""Generic eval runner — drives a Benchmark adapter against a PredictFn.

The runner is intentionally tiny: it iterates examples, calls predict,
scores each, and aggregates per-subgroup metrics. All benchmark-specific
logic (prompt formatting, answer extraction, scoring rule) lives in the
Benchmark adapter — see ``src/myllm/eval/benchmarks/``.

Test surface: feed a mock PredictFn that returns canned answers and
verify the runner computes the right accuracy. No real model required.
"""
from __future__ import annotations

from collections import defaultdict

from myllm.eval.types import Benchmark, EvalResult, PredictFn
from myllm.utils import get_logger

log = get_logger(__name__)


def run_benchmark(
    bench: Benchmark,
    predict: PredictFn,
    *,
    split: str = "test",
    sample_size: int | None = None,
    seed: int = 0,
    log_every: int = 100,
) -> EvalResult:
    """Run a benchmark end-to-end. Returns aggregate + per-subgroup metrics.

    Args:
        bench:        the benchmark adapter.
        predict:      callable ``prompt -> answer string``. Wired to the
                      model's generate path at eval time; a mock in tests.
        split:        usually ``"test"`` or ``"validation"``.
        sample_size:  if set, only evaluate the first N examples per
                      subgroup (after shuffling with ``seed``). None = full.
        seed:         shuffle seed for ``sample_size`` subsampling.
        log_every:    emit a progress log every N examples.

    Returns:
        ``EvalResult`` with overall accuracy + per-subgroup breakdown.
    """
    n_correct = 0
    n_total = 0
    per_group_correct: dict[str, int] = defaultdict(int)
    per_group_total: dict[str, int] = defaultdict(int)

    for i, ex in enumerate(bench.load_examples(split=split, sample_size=sample_size, seed=seed)):
        pred = predict(ex.prompt)
        ok = bench.score(pred, ex)
        group = bench.subgroup_key(ex)
        n_total += 1
        per_group_total[group] += 1
        if ok:
            n_correct += 1
            per_group_correct[group] += 1
        if (i + 1) % log_every == 0:
            log.info(
                "eval_progress",
                benchmark=bench.name,
                done=i + 1,
                running_accuracy=round(n_correct / max(1, n_total), 4),
            )

    overall_acc = n_correct / max(1, n_total)
    per_subgroup = {
        g: EvalResult(
            benchmark=bench.name,
            accuracy=per_group_correct[g] / max(1, per_group_total[g]),
            n_correct=per_group_correct[g],
            n_total=per_group_total[g],
        )
        for g in per_group_total
    }
    result = EvalResult(
        benchmark=bench.name,
        accuracy=overall_acc,
        n_correct=n_correct,
        n_total=n_total,
        per_subgroup=per_subgroup,
    )
    log.info(
        "eval_complete",
        benchmark=bench.name,
        accuracy=round(overall_acc, 4),
        n=n_total,
        groups=len(per_subgroup),
    )
    return result
