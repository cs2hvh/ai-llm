#!/usr/bin/env python3
"""End-to-end pretraining launcher (Phase 2 pilot, Phase 4 base).

Wires:
    1. data: HF stream → filter chain → tokenize → pack → batch
       (live path; for the B2 packed-corpus path see docs/plan_v3_after_review3.md §4)
    2. model: Keras 3 + JAX backend (KERAS_BACKEND=jax)
    3. optimizer: Optax AdamW with WSD schedule (Warmup-Stable-Decay)
       — see resolve_wsd_schedule_params() for the resolver
    4. JAX mesh: data-parallel sharding across visible GPUs
       (full FSDP weight partitioning is overkill at 1B per 2026-05-12 review;
       see docs/plan_v3_after_review3.md §2.4 for the size-vs-strategy threshold)
    5. W&B: experiment tracking
    6. checkpoint manager: Orbax + R2 mirror
    7. training loop: with watchdog + resume

Usage on a RunPod pod:
    python scripts/run_pretrain.py \\
        --model-config configs/pilot_250m.yaml \\
        --data-config configs/data/pretrain_mix.yaml \\
        --tokenizer-key tokenizer/myllm-bpe-128k-v1.json \\
        --run-name pilot-250m-001 \\
        --total-steps 10000

The script is structured so each section can be unit-tested or replaced
without touching the others.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# IMPORTANT: set Keras backend BEFORE importing keras anywhere.
os.environ.setdefault("KERAS_BACKEND", "jax")

import yaml  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.data.decontamination import (  # noqa: E402
    DecontaminationConfig,
    DecontaminationIndex,
    DecontaminationReport,
    extract_prompts_from_benchmark,
)
from myllm.data.filters import (  # noqa: E402
    DecontaminationFilter,
    FilterChain,
    LengthFilter,
    PIIRedactor,
    RepetitionFilter,
    SymbolRatioFilter,
)
from myllm.data.loader import HFStreamLoader  # noqa: E402
from myllm.data.mixture import MixtureSampler, SourceWeight  # noqa: E402
from myllm.data.pack import SequencePacker  # noqa: E402
from myllm.data.special_tokens import (  # noqa: E402
    SpecialTokens,
    verify_tokenizer_has_required,
)
from myllm.data.tokenize import (  # noqa: E402
    load_tokenizer,
    make_input_label_pairs,
    tokenize_documents,
)
from myllm.data.types import Document  # noqa: E402
from myllm.model import ModelConfig  # noqa: E402
from myllm.training.checkpoint import CheckpointConfig, find_resume_step  # noqa: E402
from myllm.training.loop import LoopConfig, run as train_loop  # noqa: E402
from myllm.training.quarantine import QuarantineWriter  # noqa: E402
from myllm.training.mesh import (  # noqa: E402
    ShardingConfig,
    build_mesh_and_shardings,
    shard_batch,
    shard_state,
)
from myllm.training.optimizer import OptimizerConfig, build_optimizer, label_model_variables  # noqa: E402
from myllm.training.train_step import make_train_step  # noqa: E402
from myllm.training.watchdog import LossSpikeWatchdog  # noqa: E402
from myllm.utils import configure_logging, get_logger  # noqa: E402
from myllm.utils.determinism import set_global_seed  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Tokenizer fetch — moved to myllm.utils.storage 2026-05-16 so library code
# (myllm.infer.predict, etc.) can use it without the sys.path hack scripts
# require. This module re-exports for backwards compatibility.
# --------------------------------------------------------------------------- #
from myllm.utils.storage import ensure_tokenizer_local  # noqa: E402, F401


# --------------------------------------------------------------------------- #
# Data pipeline
# --------------------------------------------------------------------------- #
def _instantiate_benchmark_for_decontam(bench_id: str):
    """Look up a benchmark adapter by id. Lazy-imports to avoid eval deps at
    every run_pretrain import."""
    if bench_id == "mmlu-prox":
        from myllm.eval.benchmarks import MMLUProXBenchmark
        return MMLUProXBenchmark()
    if bench_id == "belebele":
        from myllm.eval.benchmarks import BelebeleBenchmark
        return BelebeleBenchmark()
    if bench_id == "milu":
        from myllm.eval.benchmarks import MILUBenchmark
        return MILUBenchmark()
    raise ValueError(
        f"unknown benchmark id for decontamination adapter path: {bench_id!r}. "
        f"Known adapters: mmlu-prox, belebele, milu. "
        f"For prompt-only loaders (gsm8k, math, etc.) see "
        f"src/myllm/data/prompt_loaders.py."
    )


def build_decontamination_index(decon_cfg: dict | None) -> DecontaminationIndex | None:
    """Build an index from a yaml ``decontamination`` block, or return None.

    Two modes:
        - ``index_path`` (fast path): load a previously serialized index.
        - ``benchmarks`` (slow path): instantiate adapters, pull prompts
          from HF, build the index live at run-start. Slower (minutes) but
          fine for one-off pilot runs.

    Returns None when ``enabled`` is false / missing.
    """
    if not decon_cfg or not decon_cfg.get("enabled", False):
        return None

    if "index_path" in decon_cfg:
        path = decon_cfg["index_path"]
        log.info("decontamination_loading_prebuilt_index", path=path)
        idx = DecontaminationIndex.load_json(path)
        log.info(
            "decontamination_index_loaded",
            n_benchmarks=len(idx.signatures),
            total_ngrams=sum(len(s.ngrams) for s in idx.signatures.values()),
        )
        return idx

    benchmarks_cfg = decon_cfg.get("benchmarks") or []
    if not benchmarks_cfg:
        raise ValueError(
            "decontamination.enabled=true requires either 'index_path' or a "
            "non-empty 'benchmarks' list"
        )

    cfg = DecontaminationConfig(
        ngram_size=int(decon_cfg.get("ngram_size", 13)),
        hash_seed=int(decon_cfg.get("hash_seed", 0xDECAF)),
    )
    idx = DecontaminationIndex(cfg)
    # Two registry paths:
    #   - Heavyweight Benchmark adapters (mmlu-prox, belebele, milu) used by
    #     the eval harness — reuse their load_examples to keep prompt
    #     formatting consistent with eval-time scoring.
    #   - Prompt-only loaders (mmlu-pro, humaneval-plus, mbpp-plus, gsm8k,
    #     math, mgsm, bbh, ifeval) added 2026-05-12 to cover the extended
    #     v1 gate set without building full Benchmark adapters yet.
    from myllm.data.prompt_loaders import PROMPT_LOADERS, load_prompts
    _ADAPTER_IDS = {"mmlu-prox", "belebele", "milu"}
    for entry in benchmarks_cfg:
        bench_id = entry["id"]
        sample_size = entry.get("sample_size")
        log.info(
            "decontamination_indexing_benchmark",
            id=bench_id,
            sample_size=sample_size,
        )
        if bench_id in _ADAPTER_IDS:
            bench = _instantiate_benchmark_for_decontam(bench_id)
            prompts = extract_prompts_from_benchmark(bench, sample_size=sample_size)
        elif bench_id in PROMPT_LOADERS:
            prompts = load_prompts(bench_id, sample_size=sample_size)
        else:
            raise ValueError(
                f"unknown benchmark id for decontamination: {bench_id!r}. "
                f"Known adapters: {sorted(_ADAPTER_IDS)}; "
                f"prompt loaders: {sorted(PROMPT_LOADERS)}"
            )
        idx.add_benchmark(bench_id, prompts)
    log.info(
        "decontamination_index_built",
        n_benchmarks=len(idx.signatures),
        total_ngrams=sum(len(s.ngrams) for s in idx.signatures.values()),
    )
    return idx


def build_filter_chain(
    filter_cfg: dict,
    *,
    decon_index: DecontaminationIndex | None = None,
    decon_report: DecontaminationReport | None = None,
) -> FilterChain:
    """Compose the per-doc filter chain.

    Ordering (cheap → expensive, reject-fast):
        1. LengthFilter       — rejects too-short / too-long docs
        2. RepetitionFilter   — rejects spammy n-gram repetition
        3. SymbolRatioFilter  — rejects symbol-heavy docs
        4. DecontaminationFilter (optional) — rejects benchmark overlap
        5. PIIRedactor        — mutates surviving docs in place

    Decontamination is placed before PII so we don't redact docs we'll
    drop anyway. PII still runs on docs that match nothing.
    """
    filters: list = []
    if "length" in filter_cfg:
        filters.append(LengthFilter(**filter_cfg["length"]))
    if "repetition" in filter_cfg:
        filters.append(RepetitionFilter(**filter_cfg["repetition"]))
    if "symbol_ratio" in filter_cfg:
        filters.append(SymbolRatioFilter(**filter_cfg["symbol_ratio"]))
    if decon_index is not None:
        filters.append(
            DecontaminationFilter(index=decon_index, report=decon_report)
        )
    if "pii" in filter_cfg:
        filters.append(PIIRedactor(**filter_cfg["pii"]))
    if not filters:
        raise ValueError("filter chain is empty; specify at least one filter")
    return FilterChain(tuple(filters))


def filtered_documents(loader: HFStreamLoader, chain: FilterChain):
    """Apply the filter chain; yield Documents that pass."""
    for doc in loader:
        out, decision = chain.apply(doc)
        if decision.keep:
            yield out


def build_data_pipeline(
    data_config: dict,
    tokenizer,
    eos_token_id: int,
    *,
    decon_index: DecontaminationIndex | None = None,
    decon_report: DecontaminationReport | None = None,
):
    """Compose: streams → filters → tokenize → pack → (input, label) pairs.

    When ``decon_index`` is provided, the filter chain includes a
    ``DecontaminationFilter`` that drops benchmark-overlapping docs and
    accumulates stats into ``decon_report``.
    """
    chain = build_filter_chain(
        data_config["filters"],
        decon_index=decon_index,
        decon_report=decon_report,
    )

    sources = {}
    weights = []
    for entry in data_config["sources"]:
        loader = HFStreamLoader(
            dataset=entry["dataset"],
            category=entry["category"],
            text_field=entry.get("text_field", "text"),
            config_name=entry.get("config_name"),
            split=entry.get("split", "train"),
            trust_remote_code=bool(entry.get("trust_remote_code", False)),
        )
        key = f"{entry['dataset']}::{entry.get('config_name', '_')}"
        sources[key] = filtered_documents(loader, chain)
        weights.append(SourceWeight(key, float(entry["share"])))

    sampler = MixtureSampler(
        sources=sources,
        weights=weights,
        seed=int(data_config.get("data_seed", 0)),
        on_exhaust="cycle",  # pretrain is long; recycle exhausted streams
    )

    def doc_stream():
        for _, doc in sampler:
            yield doc

    token_streams = tokenize_documents(doc_stream(), tokenizer)
    packer = SequencePacker(
        sequence_length=int(data_config["batch"]["sequence_length"]),
        eos_token_id=eos_token_id,
        drop_last=True,
    )
    packed = packer.pack(token_streams)
    return make_input_label_pairs(packed)


def batch_pairs(
    pairs,
    micro_batch: int,
    sequence_length_minus_one: int,
):
    """Group (input, label, segment_ids, loss_mask) tuples into micro-batches.

    Yields ``{"input_ids", "labels", "segment_ids", "loss_mask"}`` dicts of
    numpy int32 arrays with shape ``[micro_batch, sequence_length_minus_one]``.

    Accepts either:
      - 4-tuples ``(input, label, segment_ids, loss_mask)`` from the
        post-2026-05-12 make_input_label_pairs (P0-2 fix).
      - 2-tuples ``(input, label)`` from older callers — segment_ids
        defaults to all-zeros (single segment per pack) and loss_mask
        defaults to all-ones (no boundary masking).
    """
    import numpy as np

    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    seg_ids: list[list[int]] = []
    masks: list[list[int]] = []
    for item in pairs:
        if len(item) == 4:
            inp, lab, sid, msk = item
        else:
            inp, lab = item
            sid = [0] * len(inp)
            msk = [1] * len(inp)
        inputs.append(inp)
        labels.append(lab)
        seg_ids.append(sid)
        masks.append(msk)
        if len(inputs) == micro_batch:
            yield {
                "input_ids": np.asarray(inputs, dtype=np.int32),
                "labels": np.asarray(labels, dtype=np.int32),
                "segment_ids": np.asarray(seg_ids, dtype=np.int32),
                "loss_mask": np.asarray(masks, dtype=np.int32),
            }
            inputs.clear()
            labels.clear()
            seg_ids.clear()
            masks.clear()


# --------------------------------------------------------------------------- #
# Model + optimizer init
# --------------------------------------------------------------------------- #
def resolve_micro_batch(
    *,
    cli_override: int | None,
    model_yaml: dict,
    data_yaml: dict,
    default: int = 8,
) -> int:
    """Resolve ``micro_batch_per_device`` with documented priority.

    Priority (highest to lowest):
        1. CLI override (``--micro-batch-override`` flag)
        2. model yaml's ``batch.micro_batch_per_device`` (e.g. wind_tunnel_b
           sets 4 because the 300M model doesn't fit at 8)
        3. data yaml's ``batch.micro_batch_per_device`` (shared default)
        4. hardcoded fallback

    Pure function; testable without JAX/Keras. Added 2026-05-12 after the
    Phase B re-audit caught Proxy B's micro_batch=4 being ignored because
    run_pretrain.py only read from data_yaml.
    """
    if cli_override is not None:
        log.info("micro_batch_source", source="cli_override", value=int(cli_override))
        return int(cli_override)
    model_batch = (model_yaml or {}).get("batch", {}).get("micro_batch_per_device")
    if model_batch is not None:
        log.info("micro_batch_source", source="model_yaml", value=int(model_batch))
        return int(model_batch)
    data_batch = (data_yaml or {}).get("batch", {}).get("micro_batch_per_device")
    if data_batch is not None:
        log.info("micro_batch_source", source="data_yaml", value=int(data_batch))
        return int(data_batch)
    log.warning("micro_batch_source", source="hardcoded_fallback", value=default)
    return int(default)


# resolve_wsd_schedule_params moved to myllm.training.state_init 2026-05-16.
# Re-exported here so any external caller of `from scripts.run_pretrain
# import resolve_wsd_schedule_params` keeps working.
from myllm.training.state_init import resolve_wsd_schedule_params  # noqa: E402, F401


# init_model_and_optimizer + initial_train_state moved to
# myllm.training.state_init 2026-05-16. Re-exported here for backwards
# compatibility with any external caller of `from scripts.run_pretrain
# import init_model_and_optimizer, initial_train_state`.
from myllm.training.state_init import (  # noqa: E402, F401
    init_model_and_optimizer,
    initial_train_state,
)


# --------------------------------------------------------------------------- #
# W&B
# --------------------------------------------------------------------------- #
def init_wandb(run_name: str, config_dump: dict, disabled: bool):
    if disabled:
        log.info("wandb_disabled")
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb_not_installed_falling_back_to_logs")
        return None
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "myllm"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_name,
        config=config_dump,
    )

    def log_metrics(step: int, metrics: dict) -> None:
        wandb.log(metrics, step=step)

    return log_metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", required=True)
    p.add_argument("--data-config", required=True)
    p.add_argument("--tokenizer-path", default="artifacts/tokenizer.json")
    p.add_argument(
        "--tokenizer-key",
        default=None,
        help="R2 key to download the tokenizer from if not local.",
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument(
        "--checkpoint-root",
        default="artifacts/checkpoints",
        help="Local directory for Orbax checkpoints. Mirrored to R2 on rollover.",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--no-watchdog",
        action="store_true",
        help="Disable the loss-spike watchdog. Useful for wind-tunnel HP sweeps "
             "where cells must run to completion so we get final loss numbers "
             "even on too-high LR cells.",
    )
    p.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use random tokens instead of HF data — for end-to-end loop smoke.",
    )
    p.add_argument(
        "--packed-corpus-root",
        default=None,
        help=(
            "If set, read training data from a pre-built packed corpus at this "
            "path (built by scripts/build_packed_corpus.py + "
            "scripts/compose_mixed_corpus.py) instead of streaming from HF. "
            "Recommended for production runs — gives token-exact resume and "
            "avoids HF stream + filter + tokenize overhead at training time."
        ),
    )
    p.add_argument(
        "--no-shard",
        action="store_true",
        help="Skip JAX mesh sharding (single-device debug runs only).",
    )
    p.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP/ZeRO-3 sharded state. Params, gradients, and "
             "optimizer state are split across all visible devices along "
             "the data axis instead of replicated. Cuts per-device memory "
             "~5x at 1B/seq=8192 vs DP-replicated state. Requires N>=2 "
             "devices; on N=1, FSDP collapses to replicated behavior. "
             "2026-05-13 (FSDP plan Commit D). See "
             "docs/review/QUERIES_FOR_REVIEWER_2026-05-12-evening.md and "
             "myllm.training.mesh.make_param_shardings for the design.",
    )
    p.add_argument(
        "--checkpoint-r2-prefix",
        default=None,
        help="If set, each checkpoint mirrors to s3://$S3_BUCKET/<prefix>/step-NNNN/",
    )
    # Wind-tunnel sweep overrides — let scripts/wind_tunnel_sweep.py
    # parametrise individual cells without writing one yaml per cell.
    p.add_argument(
        "--peak-lr-override",
        type=float,
        default=None,
        help="Override OptimizerConfig.peak_lr (used by wind-tunnel sweep).",
    )
    # R0 decay-phase distillation activation.
    p.add_argument(
        "--decay-phase-config",
        default=None,
        help="Path to decay_phase_distillation.yaml. When set, the loop "
             "activates teacher-guided distillation in the WSD decay phase. "
             "Without it, runs are pure CE+z (stable-only)."
    )
    p.add_argument(
        "--distillation-cache-root",
        default="artifacts/distillation_cache",
        help="Local dir holding {teacher_id}_manifest.json + shards. "
             "Required when --decay-phase-config is set."
    )
    p.add_argument(
        "--allow-cross-tokenizer-distill",
        action="store_true",
        help="Disable the fail-closed tokenizer-SHA gate on the distillation "
             "path (2026-05-16 P0). The current top-K logit KD assumes "
             "teacher and student share a tokenizer vocabulary so that "
             "teacher_topk_indices are meaningful in the student logit space "
             "(scripts/cache_teacher_logits.py:163 clamps + "
             "src/myllm/training/loss.py:279 gathers student logits at "
             "teacher indices). For cross-tokenizer teachers (DeepSeek/Olmo) "
             "this is invalid. Default: launcher refuses to start. Use this "
             "flag ONLY if you've pivoted to a tokenizer-agnostic distillation "
             "method (e.g., hidden-state distill) and know what you're doing."
    )
    p.add_argument(
        "--init-std-override",
        type=float,
        default=None,
        help="Override ModelConfig.init_std (used by wind-tunnel sweep).",
    )
    p.add_argument(
        "--micro-batch-override",
        type=int,
        default=None,
        help="Override micro_batch_per_device. Highest priority in the "
             "resolver: CLI > model yaml > data yaml > default 8.",
    )
    p.add_argument(
        "--use-chunked-ce",
        action="store_true",
        help="Use the chunked tied-LM-head CE loss path (avoids "
             "[B, S, V] logit materialisation). Required for production "
             "1B+ training at large vocab to avoid OOM; recommended in "
             "general. 2026-05-12 reviewer P0-4: was previously exposed "
             "only in benchmark_throughput.py — production silently used "
             "the full-logit path.",
    )
    p.add_argument(
        "--chunked-ce-num-chunks",
        type=int,
        default=8,
        help="Number of vocab chunks for --use-chunked-ce. Must divide "
             "model.vocab_size. Production V=131072 / 8 = 16384.",
    )
    p.add_argument(
        "--use-muon",
        action="store_true",
        help="Override config to use the Muon hybrid optimizer (Muon on "
             "hidden weight matrices via optax.contrib.muon; AdamW on "
             "embeddings + LM head + RMSNorm gains). Also activates by "
             "setting `optimizer.type: muon_hybrid` in the model YAML. "
             "Round D10 (2026-05-18). No public JAX-Muon training at >=1B "
             "exists as of 2026-05-18 — treat the first run as a smoke.",
    )
    # Eval-during-training (MVP: validation loss + perplexity on a held-out
    # batch slice). Benchmark scoring (MMLU/HumanEval) comes in a follow-up
    # once the pilot has shown the wiring is sound.
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Run eval (validation loss + perplexity) every N steps. "
             "None disables eval. Recommended for Stage 1 pilot: 5000.",
    )
    p.add_argument(
        "--eval-n-batches",
        type=int,
        default=8,
        help="How many batches to hold out at training start for the eval "
             "set. Same shape as train batches so JIT doesn't recompile. "
             "Default 8 = small enough that eval is <5%% of train cost, "
             "large enough for a stable mean estimate.",
    )
    p.add_argument(
        "--per-source-val-loss",
        action="store_true",
        help="In addition to aggregate val_loss/val_ppl, report per-source "
             "val loss bucketed by the source_id of each LABEL token (e.g., "
             "fineweb_edu, github_code_clean, mc4_zh). Requires the data "
             "path to be the packed-corpus reader (synthetic-data and "
             "tokenize-on-the-fly paths don't carry source provenance). "
             "Phase 1.2 (P0-1, 2026-05-15).",
    )
    p.add_argument(
        "--reset-data-position-on-resume",
        action="store_true",
        help="On checkpoint resume, force the data_position cursor back to "
             "0 (start of corpus). Use for Stage 1.5 decay-only passes "
             "after corpus exhaustion: load the final checkpoint's "
             "weights but re-iterate the corpus from the start. The model "
             "step counter still increments normally; only the data "
             "iterator is rewound. Without this, an exhausted-corpus "
             "resume immediately ends (iter has 0 remaining batches).",
    )
    p.add_argument(
        "--corpus-epochs",
        type=int,
        default=1,
        help="How many times the corpus iterator cycles through the packed "
             "corpus before stopping. Default 1 = single-pass (legacy "
             "behavior; matches the 2026-05-13 pilot). For Stage 2 (1B "
             "rehearsal at 10-30B tokens vs the 5B-token pilot corpus), "
             "set to 6+ so the run reaches --total-steps before the "
             "data iterator exhausts. Set to 0 for unlimited (iterator "
             "wraps forever; --total-steps is the only stop signal). "
             "Sequence IDs and data_position remain monotonically "
             "increasing across epochs, so resume is bitwise-exact.",
    )
    p.add_argument(
        "--production",
        action="store_true",
        help="Enable fail-closed safety guards for production training runs. "
             "Currently triggers: (a) strict resume safety — refuse to "
             "resume from a checkpoint that's missing the data_position "
             "field (would otherwise silently re-feed already-trained data "
             "to the model; P0-3 reviewer fix). More guards can be added "
             "here as Stage 2/3 prep items land. Smoke tests and dev runs "
             "should leave this off for the more permissive default "
             "behavior.",
    )
    args = p.parse_args()

    configure_logging()
    set_global_seed(args.seed)

    # Configs
    model_cfg = ModelConfig.from_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)
    # The model yaml may carry its own `batch:` block (Proxy B does, with
    # micro_batch_per_device=4 because the 300M model doesn't fit at 8).
    # We load the raw yaml separately because ModelConfig doesn't expose
    # the batch block as a field.
    model_yaml_raw = load_yaml(args.model_config)

    # Apply wind-tunnel sweep overrides if provided.
    if args.init_std_override is not None:
        model_cfg = model_cfg.model_copy(update={"init_std": args.init_std_override})
        log.info("init_std_overridden", value=args.init_std_override)

    # 2026-05-12 re-audit P0 fix: resolve micro_batch_per_device with priority
    #   1. --micro-batch-override CLI flag
    #   2. model yaml's batch.micro_batch_per_device (e.g. Proxy B sets 4)
    #   3. data yaml's batch.micro_batch_per_device (the shared default)
    #   4. hardcoded fallback (8)
    # Previously only #3 was read, so model yaml's request was silently
    # ignored — Proxy B's micro_batch=4 was being run as 8.
    micro_batch = resolve_micro_batch(
        cli_override=args.micro_batch_override,
        model_yaml=model_yaml_raw,
        data_yaml=data_cfg,
    )

    # P0-3 fix (2026-05-12 audit): make model_cfg.context_length authoritative
    # for ALL sequence-length math. Previously data_cfg.batch.sequence_length
    # and model_cfg.context_length could drift independently, causing silent
    # corruption (RoPE tables too short for the packed sequence length, or
    # total_steps math assuming a different seq_len than the runtime uses).
    #
    # Invariant: the packer produces sequences of length `context_length + 1`,
    # because after the next-token shift in `make_input_label_pairs` the
    # model sees exactly `context_length` tokens (which matches RoPE).
    # So:  packed_seq_len = model_cfg.context_length + 1
    #
    # If data_cfg.batch.sequence_length is set explicitly, validate it
    # matches. If unset, derive from model_cfg.context_length.
    expected_packed_seq_len = int(model_cfg.context_length) + 1
    yaml_packed_seq_len = data_cfg.get("batch", {}).get("sequence_length")
    if yaml_packed_seq_len is not None:
        if int(yaml_packed_seq_len) != expected_packed_seq_len:
            raise ValueError(
                f"sequence-length invariant violation:\n"
                f"  data_cfg.batch.sequence_length = {yaml_packed_seq_len}\n"
                f"  model_cfg.context_length       = {model_cfg.context_length}\n"
                f"  expected (context_length + 1)  = {expected_packed_seq_len}\n"
                f"These must match. Either edit data_cfg.batch.sequence_length "
                f"to {expected_packed_seq_len}, or drop the field and let it "
                f"derive from model_cfg.context_length."
            )
    packed_seq_len = expected_packed_seq_len  # length the packer emits
    model_input_len = int(model_cfg.context_length)  # length the model sees
    log.info(
        "sequence_length_resolved",
        packed_seq_len=packed_seq_len,
        model_input_len=model_input_len,
        source="model_cfg.context_length (authoritative)",
    )

    # Decon report is only populated when the real data path is active +
    # decontamination is enabled in the data yaml; synthetic path leaves it
    # None and the end-of-run emit logic is a no-op.
    decon_report: DecontaminationReport | None = None

    # Packed-corpus reader handle for Phase 1.2 per-source eval. Set only
    # when the packed-corpus path is active (synthetic + on-the-fly paths
    # don't carry source provenance, so per-source eval can't be built).
    packed_corpus_reader: Any = None

    if args.synthetic_data:
        # Smoke path: bypass tokenizer + HF entirely. Use the model's
        # context_length directly — synthetic batches have shape
        # [micro_batch, context_length] (no shift, no packer).
        from myllm.data.synthetic import make_synthetic_data_iter

        eos_id = 1
        pad_id = 0
        # Resume cursor (L3 canary fix, 2026-05-12): peek the latest
        # checkpoint step BEFORE building the iter, so the iter yields
        # batch[resume_step] first — matching what the uninterrupted run
        # saw at that step. Without this, a resumed run sees batch[0..]
        # again starting at step=resume_step, and the model's weights
        # diverge from the uninterrupted reference even though
        # data_position matches by coincidence (it's a token counter,
        # not a content fingerprint).
        synth_start_step = find_resume_step(args.checkpoint_root) or 0
        batch_iter = make_synthetic_data_iter(
            micro_batch=micro_batch,
            sequence_length=model_input_len,
            vocab_size=model_cfg.vocab_size,
            n_steps=args.total_steps + 1,
            seed=args.seed,
            start_step=synth_start_step,
        )
        log.info(
            "data_pipeline_synthetic",
            micro_batch=micro_batch,
            seq_len=model_input_len,
            start_step=synth_start_step,
        )
    elif args.packed_corpus_root is not None:
        # Packed-corpus path: random-access reader on a pre-built corpus.
        # Token-exact resume via peek of the checkpoint manifest's
        # ``data_position`` field. No HF stream / filter / tokenize at
        # training time — all of that ran offline during B2 corpus build.
        from myllm.data.packed_corpus import (
            PackedCorpusReader,
            iter_packed_pairs,
            peek_data_position_from_checkpoint,
            sequence_id_from_data_position,
        )
        # Tokenizer is still loaded — we need EOS/PAD ids for batch padding
        # and the tokenizer SHA256 cross-check.
        tok_path = ensure_tokenizer_local(args.tokenizer_path, args.tokenizer_key)
        tokenizer = load_tokenizer(tok_path)
        verify_tokenizer_has_required(tokenizer)
        eos_id = tokenizer.token_to_id(SpecialTokens.EOS)
        pad_id = tokenizer.token_to_id(SpecialTokens.PAD)

        reader = PackedCorpusReader(args.packed_corpus_root)
        packed_corpus_reader = reader  # expose for Phase 1.2 per-source eval
        if reader.sequence_length != packed_seq_len:
            raise ValueError(
                f"packed corpus sequence_length {reader.sequence_length} != "
                f"expected {packed_seq_len} (model.context_length + 1). "
                f"Re-build the corpus with the matching sequence length."
            )
        # Resume cursor: peek manifest, convert to sequence_id.
        #
        # 2026-05-12 P0 fix from packed-corpus L3 canary: the conversion
        # MUST divide by ``model_input_len`` (= context_length), not
        # ``packed_seq_len`` (= context_length + 1). The loop increments
        # ``data_position`` by ``B * input_ids.shape[1]`` per consumed
        # batch (see loop.py:_advance_data_position), and ``input_ids``
        # has shape ``[B, model_input_len]`` — NOT packed_seq_len.
        # The previous code divided by packed_seq_len and produced an
        # off-by-one (e.g., data_position=128 / 33 = 3 instead of the
        # correct 128 / 32 = 4) so every resume re-consumed the last
        # already-trained-on sequence, silently corrupting training.
        # P0-3 fix: in production mode, fail-closed if a real checkpoint
        # exists but is missing the data_position field (would silently
        # re-feed already-trained data to the model otherwise).
        resumed_data_position = peek_data_position_from_checkpoint(
            args.checkpoint_root,
            strict=args.production,
        )
        if args.reset_data_position_on_resume:
            # 2026-05-14: Stage 1.5 decay-only pass after corpus exhaustion.
            # Load weights from the final checkpoint but re-iterate the
            # corpus from the start. Step counter (state["step"]) is
            # untouched — the WSD schedule's "where am I in training"
            # logic stays consistent with the original run.
            log.warning(
                "data_position_reset_on_resume",
                original=resumed_data_position,
                msg="--reset-data-position-on-resume set; re-iterating "
                    "corpus from sequence 0 with restored weights",
            )
            resumed_data_position = 0
        start_sid = sequence_id_from_data_position(
            resumed_data_position, model_input_len,
        )
        log.info(
            "data_pipeline_packed_corpus",
            root=str(args.packed_corpus_root),
            total_sequences=reader.total_sequences,
            total_tokens=reader.total_tokens,
            tokenizer_sha256=reader.manifest.tokenizer_sha256,
            resumed_data_position=resumed_data_position,
            start_sequence_id=start_sid,
        )
        pair_iter = iter_packed_pairs(
            reader,
            start_sequence_id=start_sid,
            epochs=args.corpus_epochs if args.corpus_epochs > 0 else None,
        )
        batch_iter = batch_pairs(pair_iter, micro_batch, model_input_len)
    else:
        # Real path: tokenizer + HF stream + filters + tokenize + pack.
        tok_path = ensure_tokenizer_local(args.tokenizer_path, args.tokenizer_key)
        tokenizer = load_tokenizer(tok_path)
        verify_tokenizer_has_required(tokenizer)  # raises if anything missing
        eos_id = tokenizer.token_to_id(SpecialTokens.EOS)
        pad_id = tokenizer.token_to_id(SpecialTokens.PAD)
        # Inject the resolved length into data_cfg so the packer reads the
        # authoritative value (overriding any drift from yaml).
        data_cfg.setdefault("batch", {})["sequence_length"] = packed_seq_len

        # R6 decontamination: build (or load) the index, then thread the
        # filter into the chain. The report accumulates per-benchmark
        # match stats and is emitted as an OLMo-2-style CSV at end-of-run.
        decon_cfg = data_cfg.get("decontamination")
        decon_index = build_decontamination_index(decon_cfg)
        if decon_index is not None:
            decon_report = DecontaminationReport()

        pair_iter = build_data_pipeline(
            data_cfg,
            tokenizer,
            eos_token_id=eos_id,
            decon_index=decon_index,
            decon_report=decon_report,
        )
        batch_iter = batch_pairs(pair_iter, micro_batch, model_input_len)

    # Model + optimizer
    # Resolve peak_lr in priority order (2026-05-11 audit caught a bug where
    # the model config's lr_schedule.peak_lr was silently ignored):
    #   1. --peak-lr-override CLI flag (wind-tunnel sweep cells)
    #   2. model_cfg's yaml's lr_schedule.peak_lr section
    #   3. fall back to 2e-4 (legacy default; intentionally pessimistic)
    yaml_lr_schedule = load_yaml(args.model_config).get("lr_schedule", {})
    yaml_peak_lr = yaml_lr_schedule.get("peak_lr")
    if args.peak_lr_override is not None:
        peak_lr_value = args.peak_lr_override
        log.info("peak_lr_source", source="cli_override", value=peak_lr_value)
    elif yaml_peak_lr is not None:
        peak_lr_value = float(yaml_peak_lr)
        log.info("peak_lr_source", source="model_yaml", value=peak_lr_value)
    else:
        peak_lr_value = 2.0e-4
        log.warning("peak_lr_source", source="hardcoded_fallback", value=peak_lr_value)
    # Read the YAML's `optimizer` block for type + Muon hyperparams. The
    # legacy fields (beta1/beta2/wd/eps) are still picked up from the
    # OptimizerConfig dataclass defaults; only the Muon-specific knobs
    # surface here for now. See `configs/base_1b.yaml` `optimizer:` for
    # the schema.
    yaml_optimizer_block = load_yaml(args.model_config).get("optimizer", {}) or {}
    yaml_optimizer_type = str(yaml_optimizer_block.get("type", "adamw")).lower()
    cli_use_muon = getattr(args, "use_muon", False)
    use_muon = cli_use_muon or yaml_optimizer_type in ("muon", "muon_hybrid")
    opt_cfg = OptimizerConfig(
        peak_lr=float(peak_lr_value),
        use_muon=use_muon,
        muon_beta=float(yaml_optimizer_block.get("muon_beta", 0.95)),
        muon_ns_steps=int(yaml_optimizer_block.get("muon_ns_steps", 5)),
        muonclip_threshold=yaml_optimizer_block.get("muonclip_threshold"),
    )
    if use_muon:
        log.info(
            "optimizer_use_muon",
            source=("cli_override" if cli_use_muon else "yaml"),
            muon_beta=opt_cfg.muon_beta,
            muon_ns_steps=opt_cfg.muon_ns_steps,
            muonclip_threshold=opt_cfg.muonclip_threshold,
        )
    model, optimizer = init_model_and_optimizer(
        model_cfg, opt_cfg, total_steps=args.total_steps,
        lr_schedule_cfg=yaml_lr_schedule,
    )
    state = initial_train_state(model, optimizer)

    # JAX mesh + sharding
    #
    # Three modes (in order of preference for production training):
    #   1. --fsdp + N>=2 devices: full FSDP/ZeRO-3 sharded state. Params,
    #      grads, and opt-state are split across the data axis. Per-device
    #      memory ~5x lower than DP-replicated at 1B/seq=8192. The
    #      train_step compiles with in_shardings + donate_argnums + grad
    #      with_sharding_constraint to force reduce-scatter.
    #   2. No --fsdp (default), N>=1 devices: DP-replicated state (the
    #      pre-FSDP behavior). Batch is sharded along axis 0; params are
    #      replicated on every device. Works on any device count.
    #   3. --no-shard: skip sharding entirely (single-device debug only).
    #
    # state_shardings + batch_sharding are passed to make_train_step
    # later. They're None for the DP-replicated and --no-shard modes.
    state_shardings = None
    batch_sharding_for_train_step = None

    if not args.no_shard:
        import jax

        n_devices = len(jax.devices())
        sharding_cfg = ShardingConfig(data_parallel=n_devices, model_parallel=1)
        mesh, data_sharding, replicate_sharding = build_mesh_and_shardings(sharding_cfg)

        if args.fsdp and n_devices > 1:
            # ----- FSDP path (Commit D) -----
            from jax.sharding import NamedSharding
            from jax.sharding import PartitionSpec as P

            from myllm.training.mesh import make_param_shardings
            from myllm.training.optimizer import (
                make_optimizer_state_sharding,
            )

            log.info(
                "sharding_init",
                mode="fsdp",
                data_parallel=n_devices,
                model_parallel=1,
            )

            # Build per-leaf shardings from the materialised (CPU-init'd)
            # trainables. The model was already constructed on host; here
            # we just compute "what sharding SHOULD each leaf have" based
            # on its shape (longest-divisible-axis rule, see mesh.py).
            trainable_raw = state["trainable_variables"]
            param_shardings = make_param_shardings(trainable_raw, mesh)

            # Move trainables onto the sharded layout (real placement
            # happens here).
            trainable_sharded = jax.tree.map(
                lambda x, s: jax.device_put(x, s),
                trainable_raw, param_shardings,
            )
            # Non-trainables (RoPE tables) are small and read-only; replicate.
            non_trainable_replicated = jax.tree.map(
                lambda x: jax.device_put(x, replicate_sharding),
                state["non_trainable_variables"],
            )

            # Init opt-state UNDER jit with out_shardings = sharded layout.
            # This is the key trick: we never materialise unsharded opt
            # state on host. eval_shape derives the structure; out_shardings
            # tells XLA where to put each leaf.
            opt_state_shardings = make_optimizer_state_sharding(
                optimizer, trainable_sharded, mesh,
            )
            opt_init_jit = jax.jit(
                optimizer.init, out_shardings=opt_state_shardings
            )
            opt_state_sharded = opt_init_jit(trainable_sharded)

            # Scalars are replicated. device_put their initial values too
            # so the state dict is sharded-coherent.
            #
            # NOTE: ``data_position`` is intentionally NOT placed on device
            # and NOT included in state_shardings below. The training loop
            # pops it from state before each train_step_fn call (int32-
            # overflow fix from commit 9f442f7) and restores it as a Python
            # int afterward. Including it in state_shardings would cause a
            # pytree-structure mismatch under --fsdp (in_shardings has 6
            # keys; state arriving at JIT has 5). We carry data_position
            # OUTSIDE the JIT'd state pytree throughout the loop. Bugfix
            # 2026-05-16.
            step_repl = jax.device_put(state["step"], replicate_sharding)
            lrmult_repl = jax.device_put(
                state["lr_recovery_multiplier"], replicate_sharding
            )
            # Preserve data_position value as a plain Python int so the
            # loop can use it on first iteration; never touches the device.
            data_position_value = int(state.get("data_position", 0))

            state = {
                "trainable_variables": trainable_sharded,
                "non_trainable_variables": non_trainable_replicated,
                "opt_state": opt_state_sharded,
                "step": step_repl,
                "lr_recovery_multiplier": lrmult_repl,
                "data_position": data_position_value,
            }

            # Build the parallel sharding pytree for make_train_step's
            # in_shardings contract. MUST NOT include data_position — the
            # loop pops it before calling train_step_fn, so the JIT'd
            # function never sees it.
            state_shardings = {
                "trainable_variables": param_shardings,
                "non_trainable_variables": jax.tree.map(
                    lambda _: replicate_sharding,
                    state["non_trainable_variables"],
                ),
                "opt_state": opt_state_shardings,
                "step": replicate_sharding,
                "lr_recovery_multiplier": replicate_sharding,
            }
            batch_sharding_for_train_step = data_sharding

        else:
            # ----- DP-replicated path (pre-FSDP, still the default) -----
            if args.fsdp and n_devices <= 1:
                log.warning(
                    "fsdp_falling_back_to_dp_replicated",
                    reason="--fsdp requested but only 1 device visible",
                )
            log.info(
                "sharding_init",
                mode="dp_replicated",
                data_parallel=n_devices,
                model_parallel=1,
                devices=n_devices,
            )
            state = shard_state(state, replicate_sharding)
            # state_shardings stays None — train_step uses the pre-FSDP
            # path (no donate, no grad constraint, no out_sharding contract).

        # Both sharded modes shard the batch along the data axis.
        def _shard_each(it):
            for b in it:
                yield shard_batch(b, data_sharding)

        batch_iter = _shard_each(batch_iter)
    else:
        log.warning("sharding_disabled", reason="--no-shard flag set")

    # R0 decay-phase distillation: optional. When enabled, the loop
    # injects teacher top-K logits into batches once we pass the
    # activation step; the train_step's mixed-loss kicks in.
    decay_phase_activation = None
    distill_alpha = 1.0  # default: stable phase = pure CE
    distill_temperature = 1.0
    teacher_weights: tuple[float, ...] | None = None
    if args.decay_phase_config is not None:
        from pathlib import Path as _Path

        from myllm.data.teacher_cache import MultiTeacherCacheReader, TeacherCacheReader
        from myllm.training.decay_phase import (
            DecayPhaseActivation,
            SequentialCorpusPositions,
        )

        decay_cfg = load_yaml(args.decay_phase_config)
        distill_alpha = float(decay_cfg.get("alpha", 0.3))
        distill_temperature = float(decay_cfg.get("temperature", 1.0))
        teachers_spec = decay_cfg.get("teachers", [])
        if not teachers_spec:
            raise ValueError(
                f"decay_phase_config {args.decay_phase_config} has no 'teachers'; "
                f"cannot activate distillation"
            )
        cache_root = _Path(args.distillation_cache_root)
        readers = []
        weights = []

        # B1 fail-closed gate (2026-05-16 P0 from re-audit). The current
        # top-K logit distillation gathers student logits at teacher
        # cache's `teacher_topk_indices` (loss.py:279) under the implicit
        # assumption that student and teacher tokenizers share a vocab
        # indexing. For DeepSeek-V4-Pro / Olmo-3-32B teachers this is
        # false; the resulting "KL" would be computed across mismatched
        # logit positions and is meaningless. Refuse to launch unless
        # student tokenizer SHA matches every teacher cache's recorded
        # SHA, OR the operator explicitly opts out via
        # --allow-cross-tokenizer-distill (after pivoting to a
        # tokenizer-agnostic distill method).
        if packed_corpus_reader is None:
            raise RuntimeError(
                "Distillation (--decay-phase-config) requires "
                "--packed-corpus-root so the student tokenizer SHA can be "
                "verified against each teacher cache manifest. "
                "Synthetic / on-the-fly tokenisation can't provide the "
                "tokenizer_sha256 stamp the gate needs."
            )
        student_tokenizer_sha = (
            packed_corpus_reader.manifest.tokenizer_sha256
        )

        for t in teachers_spec:
            teacher_id = t["id"]
            manifest_path = cache_root / f"{teacher_id}_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"teacher cache manifest not found: {manifest_path}. "
                    f"Run scripts/cache_teacher_logits.py to generate it first."
                )
            # Tokenizer-SHA compatibility check (B1 fail-closed gate).
            try:
                with open(manifest_path) as f:
                    teacher_manifest = json.load(f)
                teacher_tokenizer_sha = teacher_manifest.get("tokenizer_sha256")
            except (OSError, ValueError) as e:
                raise RuntimeError(
                    f"could not parse teacher manifest {manifest_path}: {e}"
                ) from e
            if teacher_tokenizer_sha is None:
                raise RuntimeError(
                    f"teacher manifest {manifest_path} is missing "
                    f"`tokenizer_sha256`. Re-run cache_teacher_logits.py "
                    f"with the current cache format."
                )
            if teacher_tokenizer_sha != student_tokenizer_sha:
                if args.allow_cross_tokenizer_distill:
                    log.warning(
                        "distillation_tokenizer_mismatch_allowed",
                        teacher_id=teacher_id,
                        teacher_sha=teacher_tokenizer_sha[:16],
                        student_sha=student_tokenizer_sha[:16],
                        msg="--allow-cross-tokenizer-distill set; proceeding. "
                            "The top-K logit KD path assumes shared vocab "
                            "indexing — make sure you're using a "
                            "tokenizer-agnostic distill method or you'll "
                            "compute meaningless gradients.",
                    )
                else:
                    raise RuntimeError(
                        f"Distillation tokenizer mismatch: teacher "
                        f"{teacher_id!r} cache was built with tokenizer "
                        f"sha256={teacher_tokenizer_sha[:16]}..., but the "
                        f"packed corpus (student) uses "
                        f"sha256={student_tokenizer_sha[:16]}.... The "
                        f"current top-K logit KD path "
                        f"(loss.py:kl_div_topk_loss) gathers student "
                        f"logits at teacher_topk_indices, which is only "
                        f"meaningful when both vocabularies share an "
                        f"indexing. Either (a) rebuild the teacher cache "
                        f"with a teacher that shares the student's "
                        f"tokenizer, or (b) pivot to a tokenizer-agnostic "
                        f"distill method (hidden-state distill, synthetic-"
                        f"target generation) and pass "
                        f"--allow-cross-tokenizer-distill to acknowledge."
                    )
            readers.append(TeacherCacheReader(teacher_id, manifest_path, cache_root))
            weights.append(float(t.get("weight", 1.0)))
        teacher_weights = tuple(weights)
        multi_reader = MultiTeacherCacheReader(readers)
        decay_phase_activation = DecayPhaseActivation.from_yaml(
            args.decay_phase_config,
            total_steps=args.total_steps,
            reader=multi_reader,
            position_fn=SequentialCorpusPositions(),
        )
        log.info(
            "decay_phase_configured",
            n_teachers=len(readers),
            activation_step=decay_phase_activation.activation_step,
            total_steps=args.total_steps,
            alpha=distill_alpha,
            temperature=distill_temperature,
            teacher_weights=teacher_weights,
        )

    # Train step
    train_step_fn = make_train_step(
        model=model,
        optimizer=optimizer,
        z_loss_coef=model_cfg.z_loss_coef,
        ignore_index=pad_id,
        distill_alpha=distill_alpha,
        distill_temperature=distill_temperature,
        teacher_weights=teacher_weights,
        use_chunked_ce=args.use_chunked_ce,
        chunked_ce_num_chunks=args.chunked_ce_num_chunks,
        final_logit_softcap=model_cfg.final_logit_softcap,
        # FSDP contract (2026-05-13 Commit D). When the sharding block
        # above set --fsdp mode, these are populated; the train_step's
        # JIT will declare in_shardings + donate_argnums=(0,) and
        # constrain grads via with_sharding_constraint to force
        # reduce-scatter. In DP-replicated / --no-shard modes these
        # stay None and the train_step compiles pre-FSDP style.
        state_shardings=state_shardings,
        batch_sharding=batch_sharding_for_train_step,
    )
    if args.use_chunked_ce:
        log.info(
            "chunked_ce_enabled",
            num_chunks=args.chunked_ce_num_chunks,
            vocab_size=model_cfg.vocab_size,
            chunk_size=model_cfg.vocab_size // args.chunked_ce_num_chunks,
        )

    # FSDP Commit F (2026-05-13): MYLLM_DEBUG_HLO=1 lowers train_step,
    # compiles, and inspects collective ops in the HLO. Catches the
    # "silent grad replication" bug class: FSDP setup looks correct
    # (sharded params + opt state) but XLA falls back to all-reduce on
    # grads. Training is mathematically correct but ~2-4x slower than
    # it should be. Bug is invisible without HLO inspection because
    # loss, memory, and checkpoints all look normal.
    if os.environ.get("MYLLM_DEBUG_HLO") == "1":
        import jax
        # Also turn on JAX's donation logger so we see if donate_argnums
        # got silently disabled (e.g. due to a sharding mismatch between
        # input and output).
        try:
            jax.config.update("jax_log_donation", True)
        except Exception:
            pass  # older JAX versions

        from myllm.training.mesh import inspect_train_step_collectives

        # Take one batch from the iterator for the inspection. We then
        # push it back in front of the iterator so the actual training
        # loop sees it first.
        import itertools
        peek_batch = next(iter(batch_iter))
        batch_iter = itertools.chain([peek_batch], batch_iter)

        try:
            counts, hlo = inspect_train_step_collectives(
                train_step_fn, state, peek_batch,
            )
        except Exception as e:  # noqa: BLE001
            log.error("debug_hlo_lowering_failed", error=str(e))
            counts, hlo = {}, ""

        # Detect the JAX platform — assertion semantics differ between
        # CPU (where XLA lowers reduce-scatter into all-reduce + slice
        # by default, so the string "reduce-scatter" rarely appears in
        # HLO even on a correctly-FSDP'd program) and GPU/TPU (where
        # NCCL/RCCL provides native reduce-scatter and the string IS
        # the right signal).
        platform = jax.devices()[0].platform if jax.devices() else "cpu"
        log.info("debug_hlo_collective_counts", platform=platform, **counts)

        # FSDP-active expectation on GPU: reduce-scatter must appear.
        # On CPU, the same logical program lowers without that op name
        # — log and continue, don't hard-fail (would block every
        # canary on the simulated-CPU smoke path).
        if (
            state_shardings is not None
            and counts
            and platform in ("gpu", "tpu")
            and counts.get("reduce_scatter", 0) == 0
        ):
            log.error(
                "fsdp_compiled_to_all_reduce",
                platform=platform,
                counts=counts,
                hint=(
                    "FSDP path is active (state_shardings provided) "
                    "but compiled HLO has zero reduce-scatter ops on "
                    f"platform={platform}. XLA is emitting DDP-shaped "
                    "collectives — FSDP memory savings but no bandwidth "
                    "savings. Most likely cause: with_sharding_constraint "
                    "on grads missing or applied to the wrong leaf in "
                    "train_step.py:_train_step_body. Set "
                    "MYLLM_DEBUG_HLO_DUMP=/tmp/hlo.txt to write the "
                    "full HLO for inspection."
                ),
            )
            return 5
        elif (
            state_shardings is not None
            and counts
            and platform == "cpu"
            and counts.get("reduce_scatter", 0) == 0
        ):
            # CPU: log a soft note. CPU XLA backend often lowers
            # reduce-scatter into all-reduce + slice, so this is
            # expected. Real check happens on GPU.
            log.info(
                "debug_hlo_cpu_no_reduce_scatter_expected",
                hint=(
                    "Zero reduce-scatter ops on CPU is expected — XLA's "
                    "CPU backend lowers FSDP semantics into all-reduce "
                    "+ slice. Re-run on GPU to verify the actual "
                    "reduce-scatter optimization fires."
                ),
            )

        # Optional: dump full HLO if asked.
        dump_path = os.environ.get("MYLLM_DEBUG_HLO_DUMP")
        if dump_path:
            Path(dump_path).write_text(hlo)
            log.info("debug_hlo_dumped", path=dump_path, bytes=len(hlo))

    # W&B
    config_dump = {
        "model": model_cfg.model_dump(),
        "data": data_cfg,
        "args": vars(args),
    }
    on_metrics = init_wandb(args.run_name, config_dump, disabled=args.no_wandb)

    # Watchdog
    watchdog = None if args.no_watchdog else LossSpikeWatchdog()
    if args.no_watchdog:
        log.warning("watchdog_disabled", reason="--no-watchdog flag set")

    # Loop config — eval_every threads through to the periodic eval hook
    # built below. None disables eval entirely.
    loop_cfg = LoopConfig(
        total_steps=args.total_steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        reset_data_position_on_resume=args.reset_data_position_on_resume,
    )
    ckpt_cfg = CheckpointConfig(
        root=args.checkpoint_root,
        keep_last_n=3,
        keep_every_n=5000,
        r2_prefix=args.checkpoint_r2_prefix,
    )

    log.info(
        "training_start",
        run_name=args.run_name,
        total_steps=args.total_steps,
        seq_len=model_input_len,
        micro_batch=micro_batch,
    )

    # B6 (2026-05-12 audit): wire a QuarantineWriter so any nan_skipped
    # incident records the offending batch's provenance for post-mortem.
    quarantine_path = Path(args.checkpoint_root) / "quarantine.jsonl"
    quarantine = QuarantineWriter(path=quarantine_path)
    log.info("quarantine_writer_attached", path=str(quarantine_path))

    # Eval-during-training (MVP: validation loss + perplexity on a small
    # held-out slice of the data iterator, taken before training starts).
    # When --eval-every is not set, eval is disabled and the loop runs
    # exactly as before. Batches are taken off the top of the iterator so
    # the same data isn't seen during both eval and training.
    #
    # 2026-05-15 Phase 1.5: switched to a forward-only ``make_eval_step``
    # under FSDP. train_step's ``donate_argnums=(0,)`` is what previously
    # made eval-during-FSDP unsafe (calling train_step on the live state
    # would clobber its buffers). The new ``eval_step`` runs forward + CE
    # only, no grads, no opt update, no donation — safe to call between
    # FSDP train steps.
    eval_fn = None
    if args.eval_every is not None:
        log.info(
            "eval_hook_enabling",
            eval_every=args.eval_every,
            n_held_out_batches=args.eval_n_batches,
            fsdp=bool(args.fsdp),
            per_source=bool(args.per_source_val_loss),
        )
        # Phase 1.2 (P0-1): per-source val loss requires the packed-corpus
        # path (synthetic + on-the-fly tokenisation don't carry source
        # provenance). Refuse politely if --per-source-val-loss was asked
        # but the data path can't provide it.
        if args.per_source_val_loss and packed_corpus_reader is None:
            log.warning(
                "per_source_val_loss_skipped",
                reason="--per-source-val-loss requires --packed-corpus-root; "
                       "synthetic / on-the-fly data has no DocSpan source-ids",
            )

        if args.per_source_val_loss and packed_corpus_reader is not None:
            # Build held-out batches annotated with per-token source-ids.
            # Held-out sequences are taken off the TOP of the corpus; the
            # training iterator skips them via the data_position cursor
            # advance — but since reader is random-access we can read them
            # directly without consuming the training stream.
            from myllm.training.eval_step import make_eval_step
            from myllm.training.eval_hook import (
                build_per_source_held_out,
                make_per_source_validation_loss_eval_from_eval_step,
            )
            held_out, src_arrays, src_vocab = build_per_source_held_out(
                packed_corpus_reader,
                n_sequences=args.eval_n_batches * micro_batch,
                micro_batch_size=micro_batch,
            )
            if held_out:
                eval_step_fn = make_eval_step(
                    model=model,
                    z_loss_coef=model_cfg.z_loss_coef,
                    ignore_index=pad_id,
                    use_chunked_ce=args.use_chunked_ce,
                    chunked_ce_num_chunks=args.chunked_ce_num_chunks,
                    final_logit_softcap=model_cfg.final_logit_softcap,
                    return_per_token_nll=True,
                    state_shardings=state_shardings,
                    batch_sharding=batch_sharding_for_train_step,
                )
                eval_fn = make_per_source_validation_loss_eval_from_eval_step(
                    eval_step_fn, held_out, src_arrays, src_vocab, label="val",
                )
                log.info(
                    "eval_hook_attached",
                    n_batches=len(held_out),
                    path="per-source (eval_step + DocSpan source-ids)",
                    sources=sorted(src_vocab.keys()),
                )
            else:
                log.warning(
                    "eval_hook_skipped",
                    reason="per-source held-out builder returned 0 batches",
                )
        else:
            # Legacy aggregate-only path. Same as 1.5 introduced.
            from myllm.training.eval_hook import take_held_out_batches
            held_out, batch_iter = take_held_out_batches(
                batch_iter, args.eval_n_batches,
            )
            if held_out:
                if args.fsdp:
                    from myllm.training.eval_step import make_eval_step
                    from myllm.training.eval_hook import (
                        make_validation_loss_eval_from_eval_step,
                    )
                    eval_step_fn = make_eval_step(
                        model=model,
                        z_loss_coef=model_cfg.z_loss_coef,
                        ignore_index=pad_id,
                        use_chunked_ce=args.use_chunked_ce,
                        chunked_ce_num_chunks=args.chunked_ce_num_chunks,
                        final_logit_softcap=model_cfg.final_logit_softcap,
                        state_shardings=state_shardings,
                        batch_sharding=batch_sharding_for_train_step,
                    )
                    eval_fn = make_validation_loss_eval_from_eval_step(
                        eval_step_fn, held_out, label="val",
                    )
                    log.info(
                        "eval_hook_attached",
                        n_batches=len(held_out),
                        path="eval_step (FSDP-safe forward-only)",
                    )
                else:
                    from myllm.training.eval_hook import make_validation_loss_eval
                    eval_fn = make_validation_loss_eval(
                        train_step_fn, held_out, label="val",
                    )
                    log.info(
                        "eval_hook_attached",
                        n_batches=len(held_out),
                        path="train_step (legacy, DP-replicated)",
                    )
            else:
                log.warning(
                    "eval_hook_skipped",
                    reason="data iterator produced 0 held-out batches",
                )

    final_state = train_loop(
        train_step_fn=train_step_fn,
        initial_state=state,
        data_iter=batch_iter,
        loop_config=loop_cfg,
        checkpoint_config=ckpt_cfg,
        watchdog=watchdog,
        on_metrics=on_metrics,
        decay_phase=decay_phase_activation,
        quarantine=quarantine,
        eval_fn=eval_fn,
    )

    # R6: emit the per-gate contamination CSV. Only populated when
    # decontamination was enabled + the real (non-synthetic) data path
    # ran. Write next to the checkpoints so post-mortem tooling finds it.
    if decon_report is not None:
        decon_csv_path = (
            data_cfg.get("decontamination", {}).get("report_csv_path")
            or str(Path(args.checkpoint_root) / "decontamination_report.csv")
        )
        out = Path(decon_csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(decon_report.to_csv())
        log.info(
            "decontamination_report_emitted",
            path=str(out),
            n_corpus_docs_scanned=decon_report.n_corpus_docs_scanned,
            n_corpus_docs_with_any_match=decon_report.n_corpus_docs_with_any_match,
        )

    log.info("training_complete", final_step=int(final_state["step"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
