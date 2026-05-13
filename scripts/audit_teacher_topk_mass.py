#!/usr/bin/env python3
"""Teacher top-K mass audit (reviewer Q3, 2026-05-12 gate set, row "Tail audit").

The B teacher-distillation track caches the teacher's top-K logits per
token. ``K=8`` is the planned default. The risk: if the teacher's softmax
is heavy-tailed on certain domains (code, math, low-resource languages),
``K=8`` may capture <95% of probability mass and the student will be
trained against a truncated distribution that doesn't reflect the
teacher's true uncertainty.

This script measures that. For a sampled set of (batch, position)
pairs from a tokenized corpus, it computes

    p_k(b, t) = sum_{i in top-K logits} softmax(logits)_{b,t,i}

for every K in --ks (default ``4,8,16,32``) and aggregates the
distribution across positions. The summary tells you, for example,
"at K=8, p10 = 0.91" — i.e. 10% of positions have <91% top-K mass.

Reviewer trigger: **K=8 top mass < 0.95 often → move to K=16**.

Output format (JSON)::

    {
      "teacher": "deepseek-v4-pro-base",
      "n_positions": 65536,
      "by_k": {
        "4":  {"mean": ..., "p10": ..., "p50": ..., "p90": ..., "p99": ...,
               "frac_below_0.90": ..., "frac_below_0.95": ..., "frac_below_0.99": ...},
        "8":  {...},
        "16": {...},
        "32": {...}
      },
      "decision": {"recommended_k": 8, "rationale": "..."}
    }

CPU vs GPU: the synthetic-teacher path runs cheaply on CPU and validates
the math. The real-teacher path (vLLM offline inference) reuses
``_load_teacher`` from ``cache_teacher_logits.py`` and is GPU-only — a
32B-param teacher is impractical on CPU. Until GPU is available, run
this script with ``--synthetic-teacher`` to confirm wiring.

Typical real-teacher invocation::

    python scripts/audit_teacher_topk_mass.py \\
        --teacher-id deepseek-v4-pro-base \\
        --teacher-hf-model deepseek-ai/DeepSeek-V4-Pro-Base \\
        --tokenized-corpus /path/to/audit_slice.bin \\
        --tokenizer-path artifacts/tokenizer_v1.json \\
        --n-positions 65536 \\
        --ks 4,8,16,32 \\
        --output artifacts/teacher_audit/deepseek_v4_pro_base.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

# Reuse the teacher loader + corpus iterator from cache_teacher_logits so
# the audit and the real cache stay in lock-step on tokenization /
# batching semantics.
_CTL_PATH = _REPO / "scripts" / "cache_teacher_logits.py"
sys.path.insert(0, str(_REPO / "scripts"))
from cache_teacher_logits import (  # noqa: E402  (script-level import)
    _iter_tokenized_corpus,
    _load_teacher,
)

from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Core math
# --------------------------------------------------------------------------- #
def softmax_topk_mass(
    logits: np.ndarray, ks: Sequence[int]
) -> dict[int, np.ndarray]:
    """For each K in ``ks``, return the per-position top-K softmax mass.

    Args:
        logits: shape ``(N, V)`` — N (flattened) positions, V vocab.
        ks: ascending sequence of K values.

    Returns:
        ``{k: shape-(N,) float32 array}`` of per-position p_k values.

    Implementation notes:
      - ``np.partition`` is O(NV) and used once with k = max(ks); we then
        slice the partition for each requested K. Avoids one sort per K.
      - Subtracts per-row max for numerical stability before exp.
    """
    if not ks:
        raise ValueError("ks must be non-empty")
    if any(k <= 0 for k in ks):
        raise ValueError(f"all ks must be positive; got {list(ks)}")
    k_max = max(ks)
    n, v = logits.shape
    if k_max > v:
        raise ValueError(f"max(ks)={k_max} exceeds vocab size {v}")

    # Per-row stable softmax.
    row_max = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - row_max, dtype=np.float64)
    denom = exp.sum(axis=1, keepdims=True)  # (N, 1)

    # Partition once at -k_max so the last k_max columns hold the top-k_max
    # unsorted values.
    part = np.partition(exp, kth=v - k_max, axis=1)  # ascending
    top_k_max_unsorted = part[:, v - k_max :]
    # Sort just the top-k_max slice descending so we can slice top-K for each K.
    top_k_max_sorted = np.sort(top_k_max_unsorted, axis=1)[:, ::-1]

    out: dict[int, np.ndarray] = {}
    for k in ks:
        top_k_sum = top_k_max_sorted[:, :k].sum(axis=1, keepdims=True)
        mass = (top_k_sum / denom).astype(np.float32).reshape(-1)
        out[int(k)] = mass
    return out


def summarize_masses(
    masses: np.ndarray,
    thresholds: Sequence[float] = (0.90, 0.95, 0.99),
) -> dict[str, float]:
    """Quantile + threshold summary for a vector of per-position top-K masses."""
    if masses.size == 0:
        raise ValueError("masses array is empty")
    out: dict[str, float] = {
        "mean": float(masses.mean()),
        "p10": float(np.quantile(masses, 0.10)),
        "p50": float(np.quantile(masses, 0.50)),
        "p90": float(np.quantile(masses, 0.90)),
        "p99": float(np.quantile(masses, 0.99)),
    }
    for t in thresholds:
        # Fraction of positions whose top-K mass is strictly less than t.
        key = f"frac_below_{t:.2f}"
        out[key] = float((masses < t).mean())
    return out


def recommend_k(
    by_k: dict[int, dict[str, float]],
    *,
    target_mass: float = 0.95,
    max_frac_below: float = 0.10,
    candidate_ks: Sequence[int] = (4, 8, 16, 32),
) -> dict[str, Any]:
    """Pick the smallest K such that at most ``max_frac_below`` of positions
    fall below ``target_mass``. Reviewer rule: K=8 default, escalate to 16
    if "K=8 top mass <0.95 often" (often = >10% of positions by default).
    """
    frac_key = f"frac_below_{target_mass:.2f}"
    sorted_ks = sorted(int(k) for k in by_k if int(k) in candidate_ks)
    if not sorted_ks:
        return {
            "recommended_k": None,
            "rationale": f"no candidate K in {candidate_ks} present in by_k",
        }
    chosen: int | None = None
    for k in sorted_ks:
        stats = by_k.get(k) or by_k.get(str(k))
        if stats is None:
            continue
        frac = stats.get(frac_key)
        if frac is None:
            continue
        if frac <= max_frac_below:
            chosen = k
            break
    if chosen is None:
        chosen = sorted_ks[-1]
        rationale = (
            f"no K ≤ {sorted_ks[-1]} satisfies frac_below_{target_mass} "
            f"≤ {max_frac_below}; using largest candidate K={chosen} but "
            f"consider K > {chosen} or full distillation"
        )
    else:
        rationale = (
            f"smallest K with frac_below_{target_mass} ≤ {max_frac_below} "
            f"is K={chosen}"
        )
    return {
        "recommended_k": int(chosen),
        "rationale": rationale,
        "target_mass": float(target_mass),
        "max_frac_below": float(max_frac_below),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_audit(
    *,
    teacher_id: str,
    teacher_hf_model: str,
    corpus_path: Path,
    n_positions: int,
    ks: Sequence[int],
    batch_size: int,
    sequence_length: int,
    synthetic: bool,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """End-to-end audit. Returns the JSON-ready summary dict."""
    forward = _load_teacher(teacher_hf_model, synthetic=synthetic)
    rng = np.random.default_rng(rng_seed)
    sampled_masses: dict[int, list[np.ndarray]] = {int(k): [] for k in ks}
    collected = 0

    # Iterate corpus in batches, run forward, sample positions, accumulate
    # mass vectors. Stop once we have n_positions samples per K.
    for _start, _end, batch in _iter_tokenized_corpus(
        corpus_path=Path(corpus_path),
        start_token=0,
        end_token=Path(corpus_path).stat().st_size // 4,  # uint32 -> bytes / 4
        batch_size=batch_size,
        sequence_length=sequence_length,
    ):
        logits = forward(batch)  # (B, S, V)
        B, S, V = logits.shape
        # Flatten positions; sample without replacement up to remaining quota.
        flat_logits = logits.reshape(B * S, V)
        remaining = n_positions - collected
        if remaining <= 0:
            break
        # Use the whole batch's positions; if we'd exceed the quota, take a
        # random subset.
        if flat_logits.shape[0] > remaining:
            idx = rng.choice(flat_logits.shape[0], size=remaining, replace=False)
            flat_logits = flat_logits[idx]
        masses_for_batch = softmax_topk_mass(flat_logits, ks)
        for k, vec in masses_for_batch.items():
            sampled_masses[k].append(vec)
        collected += flat_logits.shape[0]
        if collected >= n_positions:
            break

    if collected == 0:
        raise RuntimeError(
            f"corpus iterator yielded no positions for {corpus_path}"
        )
    by_k = {}
    for k, parts in sampled_masses.items():
        merged = np.concatenate(parts)
        by_k[int(k)] = summarize_masses(merged)
    decision = recommend_k(by_k)
    return {
        "teacher": teacher_id,
        "teacher_hf_model": teacher_hf_model,
        "synthetic": synthetic,
        "n_positions": int(collected),
        "ks": [int(k) for k in ks],
        "by_k": {str(k): v for k, v in by_k.items()},
        "decision": decision,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_ks(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Top-K mass audit on teacher softmax distributions."
    )
    p.add_argument("--teacher-id", required=True)
    p.add_argument("--teacher-hf-model", required=True)
    p.add_argument("--tokenized-corpus", required=True, type=Path)
    p.add_argument("--tokenizer-path", type=Path, default=None,
                   help="Informational; not directly used by the audit but "
                        "logged in the output for provenance.")
    p.add_argument("--n-positions", type=int, default=65536)
    p.add_argument("--ks", type=_parse_ks, default=[4, 8, 16, 32])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--sequence-length", type=int, default=2048)
    p.add_argument("--synthetic-teacher", action="store_true",
                   help="Use the synthetic teacher from cache_teacher_logits.py "
                        "instead of loading the real HF model. CPU-friendly.")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    configure_logging()
    log.info(
        "audit_start",
        teacher=args.teacher_id,
        synthetic=args.synthetic_teacher,
        n_positions=args.n_positions,
        ks=args.ks,
    )
    summary = run_audit(
        teacher_id=args.teacher_id,
        teacher_hf_model=args.teacher_hf_model,
        corpus_path=args.tokenized_corpus,
        n_positions=args.n_positions,
        ks=args.ks,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        synthetic=args.synthetic_teacher,
        rng_seed=args.rng_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    log.info(
        "audit_done",
        path=str(args.output),
        recommended_k=summary["decision"]["recommended_k"],
        rationale=summary["decision"]["rationale"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
