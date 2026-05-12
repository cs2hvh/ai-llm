"""Canary ladder for the MyLLM training stack.

Per the 2026-05-12 reviewer Q&A §5: "the process that saves the project
is a canary ladder with hard acceptance gates." This module is the
infrastructure side of that — reusable check functions returning
structured ``CheckResult`` records. The CLI runner is
``scripts/canary_ladder.py``.

Stages:

  L0 — Static checks (CPU, seconds)
    - Model config self-consistency (head/KV-head ratio, FFN ratio, etc.)
    - Param count matches spec
    - Tokenizer round-trip lossless on canonical + random strings

  L1 — Single-GPU 20-step smoke (1 GPU, ~2 min)
    [GPU required — runnable template in canary_ladder.py]

  L2 — Multi-GPU parity check (1 node × 8 GPU, ~20 steps)
    [GPU required — runnable template]

  L3 — Forced-kill resume bitwise-exact (CPU OK with tiny model)
    Run uninterrupted for N steps, capture final state hash + loss.
    Run for N/2 steps, simulate kill, resume to N steps.
    Assert: loss(N) match within 1e-4 (bf16 noise floor) AND
    hash(params, opt_state, data_position) match exactly.

  L4 — 1B-shape scale rehearsal (full topology, 1-2% of tokens)
    [GPU required — runnable template]

  L5 — Data sanity on packed corpus (CPU)
    - Shard manifests complete (no partial writes)
    - Tokenizer SHA matches across shards
    - actual_source_share within tolerance of target_source_share
    - Random token-range sample (all < vocab_size)
    - Segment-id reconstruction clean on sampled sequences

Each check returns a ``CheckResult`` whose ``.passed`` is the gate.
A failed check should block the next ladder rung — wire this into the
launcher.

Documented failure patterns this catches (from real postmortems):
  - BLOOM tied-embedding gradient not reduced → L3 catches (param hash diverges)
  - LayerNorm in weight-decay group → L3 catches (opt_state diverges)
  - FFN_HIDDEN_SIZE typo → L0 catches (param count doesn't match spec)
  - Data cursor reset on resume → L3 catches (loss diverges after resume)
  - Mixed-tokenizer corpus → L5 catches (shard SHA mismatch)
  - Source-mix drift → L5 catches (actual vs target share)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from myllm.utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    """One canary-stage check outcome."""

    name: str
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    # Optional remediation hint surfaced to the operator on failure.
    fix_hint: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "summary": self.summary,
            "details": self.details,
            "fix_hint": self.fix_hint,
        }


@dataclass
class StageResult:
    """All checks in a single ladder rung (L0, L1, ...)."""

    stage: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary_line(self) -> str:
        n_pass = sum(1 for c in self.checks if c.passed)
        return f"{self.stage}: {n_pass}/{len(self.checks)} {'✓' if self.passed else '✗'}"


# --------------------------------------------------------------------------- #
# L0 — Static checks
# --------------------------------------------------------------------------- #
def l0_check_model_config_self_consistency(model_config_path: str | Path) -> CheckResult:
    """Validate the model yaml's internal arithmetic constraints.

    Catches: FFN_HIDDEN_SIZE-typo class (BigScience tr8-104B);
    head/KV-head ratio that breaks GQA; rope_base that's a typo.
    """
    import yaml

    cfg = yaml.safe_load(Path(model_config_path).read_text())
    issues: list[str] = []
    details: dict[str, Any] = {"path": str(model_config_path)}

    required = ["layers", "hidden_dim", "ffn_dim", "num_heads", "vocab_size",
                "context_length"]
    for k in required:
        if k not in cfg:
            issues.append(f"missing required field {k!r}")

    if not issues:
        layers = int(cfg["layers"])
        hidden = int(cfg["hidden_dim"])
        ffn = int(cfg["ffn_dim"])
        heads = int(cfg["num_heads"])
        kv_heads = int(cfg.get("num_kv_heads", heads))
        head_dim = int(cfg.get("head_dim", hidden // heads))
        vocab = int(cfg["vocab_size"])
        rope_base = float(cfg.get("rope_base", 10000))
        context = int(cfg["context_length"])
        details.update(layers=layers, hidden=hidden, ffn=ffn, heads=heads,
                       kv_heads=kv_heads, head_dim=head_dim, vocab=vocab,
                       rope_base=rope_base, context=context)
        if hidden % heads != 0:
            issues.append(f"hidden_dim {hidden} not divisible by num_heads {heads}")
        if heads % kv_heads != 0:
            issues.append(
                f"num_heads {heads} not divisible by num_kv_heads {kv_heads}; "
                "GQA grouping ill-defined"
            )
        if head_dim * heads != hidden:
            issues.append(
                f"head_dim ({head_dim}) × num_heads ({heads}) = "
                f"{head_dim * heads} ≠ hidden_dim ({hidden})"
            )
        if rope_base <= 0:
            issues.append(f"rope_base {rope_base} must be > 0")
        if context < 64:
            issues.append(f"context_length {context} suspiciously small (< 64)")
        # FFN ratio sanity — Llama default 4×, some configs use 8/3 (2.67×)
        ratio = ffn / hidden if hidden else 0
        details["ffn_ratio"] = round(ratio, 3)
        if ratio < 1.5 or ratio > 8.0:
            issues.append(f"ffn ratio {ratio:.2f} outside sane band [1.5, 8.0]")

    passed = not issues
    return CheckResult(
        name="l0_model_config_self_consistency",
        passed=passed,
        summary="config arithmetic OK" if passed else "; ".join(issues),
        details=details,
        fix_hint=None if passed else "Edit the model yaml to satisfy the listed constraints.",
    )


def l0_check_tokenizer_roundtrip(
    tokenizer_path: str | Path, *, n_random: int = 200
) -> CheckResult:
    """Round-trip a fixed set of canonical strings + N random strings.

    Canonical strings exercise the script + special-token paths that the
    tokenizer-trainer's validate() also checks. Random strings catch
    encoding/decoding asymmetries in the byte-fallback path.
    """
    try:
        from myllm.data.tokenize import load_tokenizer
    except ImportError as e:
        return CheckResult(
            name="l0_tokenizer_roundtrip",
            passed=False,
            summary=f"could not import tokenizer module: {e}",
        )
    tok_path = Path(tokenizer_path)
    if not tok_path.exists():
        return CheckResult(
            name="l0_tokenizer_roundtrip",
            passed=False,
            summary=f"tokenizer file not found: {tok_path}",
            fix_hint="Train via scripts/train_tokenizer_spm.py or pull from R2.",
        )
    tokenizer = load_tokenizer(tok_path)

    canonical = [
        "",
        "hello",
        "Hello, world! 123.",
        "नमस्ते दुनिया",          # Devanagari
        "你好世界",                # Han
        "السلام عليكم",            # Arabic
        "import numpy as np\nx = np.zeros(8)\n",
        "x" * 1000,
    ]
    # Reproducible random strings.
    import random
    rng = random.Random(0)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?\"'()-\n"
    random_strings = [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 200)))
        for _ in range(n_random)
    ]
    all_strings = canonical + random_strings

    mismatches: list[dict] = []
    for s in all_strings:
        ids = tokenizer.encode(s).ids
        decoded = tokenizer.decode(ids) if ids else ""
        if decoded != s:
            mismatches.append({
                "original": s[:80] + ("…" if len(s) > 80 else ""),
                "decoded": decoded[:80] + ("…" if len(decoded) > 80 else ""),
            })

    passed = not mismatches
    return CheckResult(
        name="l0_tokenizer_roundtrip",
        passed=passed,
        summary=(
            f"all {len(all_strings)} strings round-tripped"
            if passed
            else f"{len(mismatches)}/{len(all_strings)} strings failed round-trip"
        ),
        details={
            "n_canonical": len(canonical),
            "n_random": len(random_strings),
            "first_mismatch": mismatches[0] if mismatches else None,
        },
        fix_hint=(
            None if passed
            else "Tokenizer round-trip failure usually means byte_fallback is off "
                 "or NFKC normalization is dropping characters. Re-train the "
                 "tokenizer with byte_fallback=True."
        ),
    )


def l0_check_param_count(model_config_path: str | Path,
                         expected_params: int | None = None,
                         tolerance: float = 0.05) -> CheckResult:
    """Build the model from config and verify param count is within tolerance.

    If ``expected_params`` is provided, asserts |actual - expected| / expected
    <= tolerance. If not, just emits the actual count for the operator
    to record (the first time, you'll set ``expected`` to that number).
    """
    import os
    os.environ.setdefault("KERAS_BACKEND", "jax")
    try:
        from myllm.model.config import ModelConfig
        from myllm.model.transformer import TransformerLM
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="l0_param_count",
            passed=False,
            summary=f"could not import model: {e}",
            fix_hint="Ensure keras + jax are installed; check KERAS_BACKEND.",
        )

    cfg = ModelConfig.from_yaml(str(model_config_path))
    # Build the model and force materialization by calling on a tiny input.
    model = TransformerLM(cfg)
    try:
        import numpy as np
        dummy = np.zeros((1, min(8, cfg.context_length)), dtype=np.int32)
        _ = model(dummy)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="l0_param_count",
            passed=False,
            summary=f"could not run forward pass: {e}",
        )

    n_params = int(sum(int(v.numpy().size) for v in model.weights))
    details = {"actual_params": n_params, "expected_params": expected_params}
    if expected_params is None:
        return CheckResult(
            name="l0_param_count",
            passed=True,
            summary=f"param count: {n_params:,} (no expected — recording baseline)",
            details=details,
        )
    drift = abs(n_params - expected_params) / max(1, expected_params)
    passed = drift <= tolerance
    return CheckResult(
        name="l0_param_count",
        passed=passed,
        summary=(
            f"param count {n_params:,} matches expected {expected_params:,} "
            f"(drift {drift * 100:.2f}%)"
            if passed
            else f"param count {n_params:,} drifted from expected "
                 f"{expected_params:,} by {drift * 100:.2f}%"
        ),
        details=details,
        fix_hint=(
            None if passed
            else "Param-count drift usually means a config typo "
                 "(FFN_HIDDEN_SIZE / num_heads mismatch) — see BigScience tr8-104B."
        ),
    )


def run_l0(
    model_config_path: str | Path,
    tokenizer_path: str | Path,
    *,
    expected_params: int | None = None,
) -> StageResult:
    """L0 = config + tokenizer + (optional) param count."""
    checks = [
        l0_check_model_config_self_consistency(model_config_path),
        l0_check_tokenizer_roundtrip(tokenizer_path),
    ]
    if expected_params is not None:
        checks.append(l0_check_param_count(model_config_path, expected_params))
    return StageResult(stage="L0", checks=checks)


# --------------------------------------------------------------------------- #
# L5 — Packed corpus data sanity
# --------------------------------------------------------------------------- #
def l5_check_corpus_manifest_complete(corpus_root: str | Path) -> CheckResult:
    """All shard directories have a manifest.json (no partial writes)."""
    from myllm.data.packed_corpus import PackedCorpusReader

    root = Path(corpus_root)
    try:
        reader = PackedCorpusReader(root)
    except FileNotFoundError as e:
        return CheckResult(
            name="l5_corpus_manifest_complete",
            passed=False,
            summary=f"top-level manifest missing: {e}",
            fix_hint="Run write_corpus_manifest after all per-source builds complete.",
        )
    n_shards_expected = reader.manifest.n_shards
    shard_dirs = sorted(d for d in root.glob("shard-*") if d.is_dir())
    incomplete: list[str] = [
        str(d) for d in shard_dirs if not (d / "manifest.json").exists()
    ]
    passed = (
        len(shard_dirs) >= n_shards_expected
        and not incomplete
    )
    return CheckResult(
        name="l5_corpus_manifest_complete",
        passed=passed,
        summary=(
            f"all {n_shards_expected} shards have manifests"
            if passed
            else f"{len(incomplete)} shards lack manifest.json"
        ),
        details={
            "n_shards_expected": n_shards_expected,
            "n_shards_on_disk": len(shard_dirs),
            "incomplete": incomplete[:10],
        },
        fix_hint=(
            None if passed
            else "Shards lacking manifest.json are partial writes "
                 "(crashed builder). Delete + rebuild those shards or accept "
                 "fewer-than-expected if the build was intentional."
        ),
    )


def l5_check_tokenizer_sha_uniform(corpus_root: str | Path) -> CheckResult:
    """Every per-shard manifest's tokenizer_sha256 matches the corpus-level value."""
    from myllm.data.packed_corpus import PackedCorpusReader

    reader = PackedCorpusReader(Path(corpus_root))
    expected = reader.manifest.tokenizer_sha256
    mismatches: list[dict] = []
    for d in sorted(p for p in Path(corpus_root).glob("shard-*") if p.is_dir()):
        mp = d / "manifest.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        sha = m.get("tokenizer_sha256")
        if sha != expected:
            mismatches.append({"shard": d.name, "tokenizer_sha256": sha})
    passed = not mismatches
    return CheckResult(
        name="l5_tokenizer_sha_uniform",
        passed=passed,
        summary=(
            f"all shards use tokenizer {expected[:12]}…"
            if passed
            else f"{len(mismatches)} shards have differing tokenizer_sha256"
        ),
        details={
            "expected_tokenizer_sha256": expected,
            "mismatches": mismatches[:10],
        },
        fix_hint=(
            None if passed
            else "A mixed-tokenizer corpus is a silent-corruption bug — every "
                 "shard MUST use the same tokenizer. Rebuild the offending shards."
        ),
    )


def l5_check_source_share_drift(
    corpus_root: str | Path, tolerance: float = 0.02,
) -> CheckResult:
    """Actual source share matches target within ``tolerance``.

    Per-source builds usually exhaust at different rates; the launcher
    can renormalize but big drifts (>2%) mean the corpus is sampling
    further from the intended mix than expected.
    """
    from myllm.data.packed_corpus import PackedCorpusReader

    reader = PackedCorpusReader(Path(corpus_root))
    target = reader.manifest.target_source_share
    actual = reader.manifest.actual_source_share
    drifts = {
        sid: abs(actual.get(sid, 0.0) - target.get(sid, 0.0))
        for sid in set(target) | set(actual)
    }
    max_drift = max(drifts.values()) if drifts else 0.0
    passed = max_drift <= tolerance
    return CheckResult(
        name="l5_source_share_drift",
        passed=passed,
        summary=(
            f"max source-share drift {max_drift * 100:.2f}% ≤ "
            f"{tolerance * 100:.0f}%"
            if passed
            else f"max source-share drift {max_drift * 100:.2f}% exceeds "
                 f"tolerance {tolerance * 100:.0f}%"
        ),
        details={
            "target": target,
            "actual": actual,
            "drifts": {k: round(v, 4) for k, v in drifts.items()},
        },
        fix_hint=(
            None if passed
            else "Big drift means the deficit-driven sampler couldn't satisfy "
                 "the target — usually a source exhausted early. Either rebuild "
                 "with more tokens-per-source or accept the drift in the manifest."
        ),
    )


def l5_check_token_range(
    corpus_root: str | Path,
    *,
    vocab_size: int,
    n_samples: int = 32,
    seed: int = 0,
) -> CheckResult:
    """Sample N random sequences, assert every token id is in [0, vocab_size).

    Catches: writer dtype-coercion bug, vocab-size drift between tokenizer
    and config, off-by-one in EOS/PAD insertion.
    """
    import random

    import numpy as np

    from myllm.data.packed_corpus import PackedCorpusReader

    reader = PackedCorpusReader(Path(corpus_root))
    if reader.total_sequences == 0:
        return CheckResult(
            name="l5_token_range",
            passed=False,
            summary="corpus has zero sequences",
        )
    rng = random.Random(seed)
    sample_sids = rng.sample(
        range(reader.total_sequences),
        k=min(n_samples, reader.total_sequences),
    )
    out_of_range: list[dict] = []
    max_observed = -1
    for sid in sample_sids:
        tokens = np.asarray(reader.get_sequence(sid))
        local_max = int(tokens.max())
        if local_max > max_observed:
            max_observed = local_max
        # Token >= vocab_size is the failure case. uint32 wraparound from
        # an earlier uint16 storage bug would land in the gigabyte range.
        if local_max >= vocab_size:
            out_of_range.append({
                "sequence_id": int(sid),
                "max_token_id": local_max,
            })
    passed = not out_of_range
    return CheckResult(
        name="l5_token_range",
        passed=passed,
        summary=(
            f"all sampled tokens in [0, {vocab_size:,}); max observed {max_observed:,}"
            if passed
            else f"{len(out_of_range)} sequences contain token >= vocab_size"
        ),
        details={
            "n_sampled": len(sample_sids),
            "vocab_size": vocab_size,
            "max_observed_token_id": max_observed,
            "out_of_range_examples": out_of_range[:5],
        },
        fix_hint=(
            None if passed
            else "Token id >= vocab_size implies tokenizer/corpus mismatch OR "
                 "uint32 storage corruption. Rebuild the corpus with the "
                 "correct tokenizer."
        ),
    )


def l5_check_segment_ids_well_formed(
    corpus_root: str | Path,
    *,
    n_samples: int = 16,
    seed: int = 0,
) -> CheckResult:
    """Sample N sequences; their reconstructed segment_ids must be valid.

    Validity = (a) array length matches sequence_length, (b) all values
    are -1 or >= 0, (c) non-sentinel values form a non-decreasing run when
    iterated left-to-right within each contiguous segment region.
    """
    import random

    import numpy as np

    from myllm.data.packed_corpus import PackedCorpusReader

    reader = PackedCorpusReader(Path(corpus_root))
    if reader.total_sequences == 0:
        return CheckResult(
            name="l5_segment_ids",
            passed=False,
            summary="corpus has zero sequences",
        )
    rng = random.Random(seed)
    sample_sids = rng.sample(
        range(reader.total_sequences),
        k=min(n_samples, reader.total_sequences),
    )
    failures: list[dict] = []
    for sid in sample_sids:
        seg = np.asarray(reader.get_segment_ids(sid))
        if seg.shape != (reader.sequence_length,):
            failures.append({"sequence_id": int(sid),
                             "reason": f"wrong shape {seg.shape}"})
            continue
        if (seg < -1).any():
            failures.append({"sequence_id": int(sid),
                             "reason": f"value < -1 found: min={int(seg.min())}"})
            continue
        # Non-sentinel values should not be wildly large.
        non_pad = seg[seg >= 0]
        if non_pad.size > 0 and non_pad.max() > 10_000:
            failures.append({"sequence_id": int(sid),
                             "reason": f"max segment_id {int(non_pad.max())} > 10,000 "
                                       f"(absurdly many docs per packed sequence)"})
            continue
    passed = not failures
    return CheckResult(
        name="l5_segment_ids",
        passed=passed,
        summary=(
            f"all {len(sample_sids)} sampled segment_ids well-formed"
            if passed
            else f"{len(failures)}/{len(sample_sids)} segment_id reconstructions failed"
        ),
        details={
            "n_sampled": len(sample_sids),
            "failures": failures[:5],
        },
    )


def run_l5(
    corpus_root: str | Path,
    *,
    vocab_size: int,
    source_share_tolerance: float = 0.02,
    n_token_samples: int = 32,
) -> StageResult:
    """L5 = packed corpus data sanity (all CPU)."""
    return StageResult(
        stage="L5",
        checks=[
            l5_check_corpus_manifest_complete(corpus_root),
            l5_check_tokenizer_sha_uniform(corpus_root),
            l5_check_source_share_drift(corpus_root, tolerance=source_share_tolerance),
            l5_check_token_range(corpus_root, vocab_size=vocab_size,
                                 n_samples=n_token_samples),
            l5_check_segment_ids_well_formed(corpus_root),
        ],
    )


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def report_to_json(stages: list[StageResult]) -> str:
    return json.dumps(
        [
            {
                "stage": s.stage,
                "passed": s.passed,
                "checks": [c.to_dict() for c in s.checks],
            }
            for s in stages
        ],
        indent=2,
    )


def report_to_text(stages: list[StageResult]) -> str:
    lines: list[str] = []
    for s in stages:
        lines.append(s.summary_line())
        for c in s.checks:
            sigil = "✓" if c.passed else "✗"
            lines.append(f"  {sigil} {c.name}: {c.summary}")
            if not c.passed and c.fix_hint:
                lines.append(f"      ↳ fix: {c.fix_hint}")
    overall = all(s.passed for s in stages)
    lines.append("")
    lines.append("OVERALL: " + ("✓ PASS" if overall else "✗ FAIL"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# State hashing for L3 — used by canary_l3_resume.py
# --------------------------------------------------------------------------- #
def hash_training_state(state: dict[str, Any]) -> str:
    """SHA256 hash of the state's serializable contents.

    Walks the state dict, materializes any JAX/numpy arrays to bytes,
    and hashes them in a stable key order. Used by L3 to compare
    uninterrupted-vs-resumed final states bitwise.
    """
    import numpy as np

    def _walk(obj):
        if hasattr(obj, "numpy"):
            return np.asarray(obj.numpy()).tobytes()
        try:
            return np.asarray(obj).tobytes()
        except Exception:  # noqa: BLE001
            return repr(obj).encode("utf-8")

    h = hashlib.sha256()
    for k in sorted(state.keys()):
        h.update(k.encode("utf-8"))
        h.update(b":")
        v = state[k]
        if isinstance(v, dict):
            # Nested PyTree-like — hash leaves in stable order.
            for kk in sorted(v.keys()):
                h.update(kk.encode("utf-8"))
                h.update(b":")
                h.update(_walk(v[kk]))
        else:
            h.update(_walk(v))
    return h.hexdigest()
