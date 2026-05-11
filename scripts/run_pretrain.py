#!/usr/bin/env python3
"""End-to-end pretraining launcher (Phase 2 pilot, Phase 4 base).

Wires:
    1. data: HF stream → filter chain → tokenize → pack → batch
    2. model: Keras 3 + JAX backend (KERAS_BACKEND=jax)
    3. optimizer: Optax AdamW with cosine-warmup schedule
    4. JAX mesh: data-parallel sharding across visible GPUs (FSDP later)
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
from myllm.training.checkpoint import CheckpointConfig  # noqa: E402
from myllm.training.loop import LoopConfig, run as train_loop  # noqa: E402
from myllm.training.mesh import (  # noqa: E402
    ShardingConfig,
    build_mesh_and_shardings,
    shard_batch,
    shard_state,
)
from myllm.training.optimizer import OptimizerConfig, build_optimizer, label_model_variables  # noqa: E402
from myllm.training.schedule import cosine_with_warmup  # noqa: E402
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
# Tokenizer fetch
# --------------------------------------------------------------------------- #
def ensure_tokenizer_local(local_path: str, remote_key: str | None) -> str:
    """If local file is absent and a remote key is given, download from R2."""
    p = Path(local_path)
    if p.exists():
        log.info("tokenizer_already_local", path=str(p))
        return str(p)
    if remote_key is None:
        raise FileNotFoundError(
            f"tokenizer not at {local_path} and --tokenizer-key not provided"
        )
    from myllm.utils.storage import download_file

    download_file(remote_key, p)
    log.info("tokenizer_downloaded", path=str(p), remote_key=remote_key)
    return str(p)


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
        f"unknown benchmark id for decontamination: {bench_id!r}. "
        f"Known: mmlu-prox, belebele, milu."
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
    for entry in benchmarks_cfg:
        bench_id = entry["id"]
        sample_size = entry.get("sample_size")
        log.info(
            "decontamination_indexing_benchmark",
            id=bench_id,
            sample_size=sample_size,
        )
        bench = _instantiate_benchmark_for_decontam(bench_id)
        prompts = extract_prompts_from_benchmark(bench, sample_size=sample_size)
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
def resolve_wsd_schedule_params(
    peak_lr: float,
    total_steps: int,
    *,
    lr_schedule_cfg: dict | None = None,
) -> dict:
    """Resolve WSD schedule parameters from yaml + sensible defaults.

    Returns dict with keys: ``warmup_steps``, ``decay_steps``, ``stable_steps``,
    ``end_lr``. Pure function — testable without JAX/Keras.

    P0 audit (2026-05-12): this used to be hardcoded inline in
    ``init_model_and_optimizer``, silently overriding yaml lr_schedule.
    Factored out so it can be unit-tested.

    Defaults when fields are absent:
        warmup_steps = min(2000, total_steps // 10)
        decay_fraction = 0.15
        end_lr_ratio = 0.1
    """
    schedule_cfg = lr_schedule_cfg or {}
    default_warmup = max(1, min(2000, total_steps // 10))
    warmup_steps = int(schedule_cfg.get("warmup_steps", default_warmup))
    decay_fraction = float(schedule_cfg.get("decay_fraction", 0.15))
    end_lr_ratio = float(schedule_cfg.get("end_lr_ratio", 0.1))
    decay_steps = max(1, int(total_steps * decay_fraction))
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)
    end_lr = peak_lr * end_lr_ratio
    return {
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "stable_steps": stable_steps,
        "end_lr": end_lr,
    }


def init_model_and_optimizer(
    model_cfg: ModelConfig,
    opt_cfg: OptimizerConfig,
    total_steps: int,
    *,
    lr_schedule_cfg: dict | None = None,
):
    """Construct model, build it (allocate weights), wire optimizer.

    ``lr_schedule_cfg`` is the optional yaml `lr_schedule` block. When set,
    its fields override the hardcoded WSD defaults. P0 audit (2026-05-12)
    flagged that this used to be ignored — pilot's configured warmup/decay
    were silently overridden by hardcoded values.
    """
    from myllm.model.transformer import build_model

    log.info("building_model", name=model_cfg.name, params_estimate=model_cfg.param_count_estimate())
    model = build_model(model_cfg)

    # Warmup-Stable-Decay schedule (playbook recommendation): linear warmup,
    # then constant peak_lr through the stable phase, then linear decay over
    # the last fraction. Doesn't commit to total_steps upfront — any stable-phase
    # checkpoint can be cooled in 10-15% of remaining compute.
    import optax  # noqa: F811

    sched = resolve_wsd_schedule_params(
        opt_cfg.peak_lr, total_steps, lr_schedule_cfg=lr_schedule_cfg
    )
    warmup_steps = sched["warmup_steps"]
    decay_steps = sched["decay_steps"]
    stable_steps = sched["stable_steps"]
    end_lr = sched["end_lr"]
    log.info(
        "lr_schedule_resolved",
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        peak_lr=opt_cfg.peak_lr,
        end_lr=end_lr,
        source="yaml lr_schedule" if lr_schedule_cfg else "hardcoded defaults",
    )

    lr_fn = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=opt_cfg.peak_lr,
                transition_steps=warmup_steps,
            ),
            optax.constant_schedule(value=opt_cfg.peak_lr),
            optax.linear_schedule(
                init_value=opt_cfg.peak_lr,
                end_value=end_lr,
                transition_steps=decay_steps,
            ),
        ],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )

    # muP per-parameter LR scaling. When model_cfg.mup is None, width_mult
    # collapses to 1.0 and build_optimizer returns the legacy single-AdamW
    # chain (no behavior change). When muP is set, hidden weights get LR
    # scaled by 1/width_mult.
    width_mult = model_cfg.mup_width_multiplier()
    param_labels = label_model_variables(model) if model_cfg.mup is not None else None
    if model_cfg.mup is not None:
        log.info(
            "mup_optimizer_active",
            width_mult=width_mult,
            base_width=model_cfg.mup.base_width,
            hidden_dim=model_cfg.hidden_dim,
            n_embedding=param_labels.count("embedding"),
            n_norm=param_labels.count("norm"),
            n_hidden=param_labels.count("hidden"),
        )

    optimizer = build_optimizer(
        opt_cfg, lr_fn,
        param_labels=param_labels,
        mup_width_mult=width_mult,
    )
    return model, optimizer


def initial_train_state(model, optimizer):
    import jax.numpy as jnp

    trainable = [v.value for v in model.trainable_variables]
    non_trainable = [v.value for v in model.non_trainable_variables]
    opt_state = optimizer.init(trainable)
    return {
        "trainable_variables": trainable,
        "non_trainable_variables": non_trainable,
        "opt_state": opt_state,
        "step": 0,
        "lr_recovery_multiplier": jnp.float32(1.0),
    }


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
        "--no-shard",
        action="store_true",
        help="Skip JAX mesh sharding (single-device debug runs only).",
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
        "--init-std-override",
        type=float,
        default=None,
        help="Override ModelConfig.init_std (used by wind-tunnel sweep).",
    )
    args = p.parse_args()

    configure_logging()
    set_global_seed(args.seed)

    # Configs
    model_cfg = ModelConfig.from_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)

    # Apply wind-tunnel sweep overrides if provided.
    if args.init_std_override is not None:
        model_cfg = model_cfg.model_copy(update={"init_std": args.init_std_override})
        log.info("init_std_overridden", value=args.init_std_override)

    micro_batch = int(data_cfg["batch"]["micro_batch_per_device"])

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

    if args.synthetic_data:
        # Smoke path: bypass tokenizer + HF entirely. Use the model's
        # context_length directly — synthetic batches have shape
        # [micro_batch, context_length] (no shift, no packer).
        from myllm.data.synthetic import make_synthetic_data_iter

        eos_id = 1
        pad_id = 0
        batch_iter = make_synthetic_data_iter(
            micro_batch=micro_batch,
            sequence_length=model_input_len,
            vocab_size=model_cfg.vocab_size,
            n_steps=args.total_steps + 1,
            seed=args.seed,
        )
        log.info(
            "data_pipeline_synthetic",
            micro_batch=micro_batch,
            seq_len=model_input_len,
        )
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
    opt_cfg = OptimizerConfig(
        peak_lr=float(peak_lr_value),
    )
    model, optimizer = init_model_and_optimizer(
        model_cfg, opt_cfg, total_steps=args.total_steps,
        lr_schedule_cfg=yaml_lr_schedule,
    )
    state = initial_train_state(model, optimizer)

    # JAX mesh + sharding (data-parallel; FSDP lands later)
    if not args.no_shard:
        import jax

        n_devices = len(jax.devices())
        sharding_cfg = ShardingConfig(data_parallel=n_devices, model_parallel=1)
        log.info(
            "sharding_init", data_parallel=n_devices, model_parallel=1, devices=n_devices
        )
        _, data_sharding, replicate_sharding = build_mesh_and_shardings(sharding_cfg)
        state = shard_state(state, replicate_sharding)

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
        for t in teachers_spec:
            teacher_id = t["id"]
            manifest_path = cache_root / f"{teacher_id}_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"teacher cache manifest not found: {manifest_path}. "
                    f"Run scripts/cache_teacher_logits.py to generate it first."
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
    )

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

    # Loop config
    loop_cfg = LoopConfig(
        total_steps=args.total_steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
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

    final_state = train_loop(
        train_step_fn=train_step_fn,
        initial_state=state,
        data_iter=batch_iter,
        loop_config=loop_cfg,
        checkpoint_config=ckpt_cfg,
        watchdog=watchdog,
        on_metrics=on_metrics,
        decay_phase=decay_phase_activation,
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
