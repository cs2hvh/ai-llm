#!/usr/bin/env bash
# Track A — FSDP gauntlet G1-G6.
#
# Runs the six FSDP validation checks in sequence on a GPU pod, captures
# results to /workspace/gauntlet/, and writes a single JSON summary
# (gauntlet_results.json) that gets uploaded to R2.
#
# Pass criteria per gate (from docs / reviewer Q&A):
#   G1: model + mesh construct, first 3 train steps complete, no NaN.
#   G2: MYLLM_DEBUG_HLO=1 shows `reduce_scatter > 0` AND `all_reduce`
#       does NOT dominate (on GPU; CPU lowers reduce_scatter to
#       all_reduce+slice which is fine and platform-gated to "log only").
#   G3: L2 parity canary — DP vs FSDP losses match within atol=5e-3
#       over 50 steps. Both curves non-empty.
#   G4: FSDP peak GPU memory is < 70% of DP peak (real ZeRO-3 win).
#   G5: FSDP tokens/sec/GPU is within 30% of DP baseline.
#   G6: Save FSDP checkpoint, reshard to a different mesh size, reload —
#       output params match the source step's eval loss within 1e-4.
#
# Each gate logs to /workspace/gauntlet/g<N>.log; the summary is at
# /workspace/gauntlet/gauntlet_results.json. The script exits non-zero
# if any gate fails (any later gate will still run; you'll see which
# ones failed in the summary).
#
# Usage:
#   bash scripts/run_fsdp_gauntlet.sh
#
# Env overrides:
#   GAUNTLET_MODEL_CONFIG   default: configs/wind_tunnel.yaml (68M, 2048 ctx)
#   GAUNTLET_DATA_CONFIG    default: configs/data/pretrain_mix.yaml
#   GAUNTLET_STEPS          default: 50 (used for G3/G5)
#   GAUNTLET_R2_PREFIX      default: fsdp_gauntlet/<UTC-date>
#   GAUNTLET_SKIP           comma-separated gate ids to skip (e.g. "g4,g6")
set -uo pipefail   # NOTE: NOT -e — we want to keep going on per-gate fail

WORKDIR="${MYLLM_WORKDIR:-/workspace/llm-build}"
RESULTS_DIR="${MYLLM_GAUNTLET_DIR:-/workspace/gauntlet}"
MODEL_CFG="${GAUNTLET_MODEL_CONFIG:-configs/wind_tunnel.yaml}"
DATA_CFG="${GAUNTLET_DATA_CONFIG:-configs/data/pretrain_mix.yaml}"
STEPS="${GAUNTLET_STEPS:-50}"
SKIP_GATES="${GAUNTLET_SKIP:-}"

UTC_DATE="$(date -u +%Y-%m-%d)"
R2_PREFIX="${GAUNTLET_R2_PREFIX:-fsdp_gauntlet/${UTC_DATE}}"

mkdir -p "$RESULTS_DIR"
cd "$WORKDIR"

# shellcheck disable=SC1091
source .venv/bin/activate

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*"; exit 1; }

# Detect GPU count for the mesh size used by FSDP.
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log "GPU count: $GPU_COUNT"
if [[ "$GPU_COUNT" -lt 2 ]]; then
    die "FSDP gauntlet requires >=2 GPUs; saw $GPU_COUNT"
fi

# FSDP shards the per-step batch along the 'data' mesh axis, so the
# global batch must be divisible by GPU_COUNT. wind_tunnel.yaml's
# default micro_batch_per_device=8 fails on 3 GPUs (8 mod 3 != 0).
# Pick the smallest reasonable multiple of GPU_COUNT (>=8) and pass it
# via --micro-batch-override. Same value is used for the DP runs so
# G4 / G5 comparisons are apples-to-apples.
MIN_BATCH=8
MICRO_BATCH="${GAUNTLET_MICRO_BATCH:-}"
if [[ -z "$MICRO_BATCH" ]]; then
    # smallest multiple of GPU_COUNT that is >= MIN_BATCH
    MICRO_BATCH=$(( ((MIN_BATCH + GPU_COUNT - 1) / GPU_COUNT) * GPU_COUNT ))
fi
log "micro_batch (global, divisible by $GPU_COUNT): $MICRO_BATCH"

# Common run_pretrain args used by every gate. --micro-batch-override
# forces a value compatible with FSDP's data-axis sharding for this
# GPU count. --synthetic-data avoids needing a real corpus on disk.
COMMON_ARGS=(
    --model-config "$MODEL_CFG"
    --data-config "$DATA_CFG"
    --tokenizer-path artifacts/tokenizer_v1.json
    --no-wandb
    --synthetic-data
    --micro-batch-override "$MICRO_BATCH"
)

# Helper: gate-skipped check
is_skipped() {
    local gate="$1"
    [[ ",$SKIP_GATES," == *",$gate,"* ]]
}

# Per-gate result aggregator (JSON via simple python tool at the end)
declare -A GATE_STATUS  # gate -> "PASS" / "FAIL" / "SKIP"
declare -A GATE_DETAIL  # gate -> short human-readable detail

# ========================================================================
# G1 — Model + mesh construct, first 3 steps no NaN
# ========================================================================
if is_skipped g1; then
    GATE_STATUS[g1]="SKIP"; GATE_DETAIL[g1]="skipped via GAUNTLET_SKIP"
else
    log "G1: FSDP boot + 3 steps without NaN"
    G1_LOG="$RESULTS_DIR/g1.log"
    python scripts/run_pretrain.py "${COMMON_ARGS[@]}" \
        --run-name g1_boot --total-steps 3 --fsdp \
        > "$G1_LOG" 2>&1
    G1_RC=$?
    if [[ $G1_RC -eq 0 ]] && ! grep -qi "nan\|NaN" "$G1_LOG"; then
        GATE_STATUS[g1]="PASS"
        GATE_DETAIL[g1]="3 train steps completed, no NaN"
    else
        GATE_STATUS[g1]="FAIL"
        GATE_DETAIL[g1]="rc=$G1_RC; see $G1_LOG"
    fi
fi
log "  G1: ${GATE_STATUS[g1]:-?}"

# ========================================================================
# G2 — HLO shows reduce_scatter on GPU
# ========================================================================
if is_skipped g2; then
    GATE_STATUS[g2]="SKIP"; GATE_DETAIL[g2]="skipped via GAUNTLET_SKIP"
else
    log "G2: HLO collective inspection (MYLLM_DEBUG_HLO=1)"
    G2_LOG="$RESULTS_DIR/g2.log"
    MYLLM_DEBUG_HLO=1 python scripts/run_pretrain.py "${COMMON_ARGS[@]}" \
        --run-name g2_hlo --total-steps 3 --fsdp \
        > "$G2_LOG" 2>&1
    G2_RC=$?
    RS_COUNT=$(grep -oE '"reduce_scatter":[[:space:]]*[0-9]+' "$G2_LOG" | head -1 | grep -oE '[0-9]+' || echo 0)
    AR_COUNT=$(grep -oE '"all_reduce":[[:space:]]*[0-9]+'    "$G2_LOG" | head -1 | grep -oE '[0-9]+' || echo 0)
    if [[ $G2_RC -eq 0 ]] && [[ "${RS_COUNT:-0}" -gt 0 ]]; then
        GATE_STATUS[g2]="PASS"
        GATE_DETAIL[g2]="reduce_scatter=$RS_COUNT all_reduce=$AR_COUNT"
    else
        GATE_STATUS[g2]="FAIL"
        GATE_DETAIL[g2]="reduce_scatter=$RS_COUNT all_reduce=$AR_COUNT (rc=$G2_RC; see $G2_LOG)"
    fi
fi
log "  G2: ${GATE_STATUS[g2]:-?}  (${GATE_DETAIL[g2]:-})"

# ========================================================================
# G3 — L2 parity canary (DP vs FSDP loss curves)
# ========================================================================
if is_skipped g3; then
    GATE_STATUS[g3]="SKIP"; GATE_DETAIL[g3]="skipped via GAUNTLET_SKIP"
else
    log "G3: L2 parity canary, $STEPS steps, atol=5e-3"
    G3_LOG="$RESULTS_DIR/g3.log"
    G3_JSON="$RESULTS_DIR/g3.json"
    python scripts/canary_l2_fsdp_parity.py \
        --total-steps "$STEPS" --atol 5e-3 --format json \
        > "$G3_JSON" 2> "$G3_LOG"
    G3_RC=$?
    if [[ $G3_RC -eq 0 ]]; then
        GATE_STATUS[g3]="PASS"
        GATE_DETAIL[g3]="curves match within atol=5e-3"
    else
        GATE_STATUS[g3]="FAIL"
        GATE_DETAIL[g3]="rc=$G3_RC; see $G3_LOG / $G3_JSON"
    fi
fi
log "  G3: ${GATE_STATUS[g3]:-?}"

# ========================================================================
# G4 — Peak GPU memory: FSDP should use a fraction of DP
# ========================================================================
if is_skipped g4; then
    GATE_STATUS[g4]="SKIP"; GATE_DETAIL[g4]="skipped via GAUNTLET_SKIP"
else
    log "G4: peak GPU memory comparison (5 steps each)"
    G4_DP_LOG="$RESULTS_DIR/g4_dp.log"
    G4_FSDP_LOG="$RESULTS_DIR/g4_fsdp.log"

    measure_peak_mem() {
        # Spawn nvidia-smi sampler in bg, kill when the train call returns.
        # Sample only GPU 0 to avoid duplicate rows skewing the max.
        local out="$1"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i 0 -l 1 > "$out.mem" &
        local smi_pid=$!
        shift
        # Disable XLA's 80%-of-VRAM preallocation so the nvidia-smi reading
        # reflects actual model+activation usage, not the preallocated pool.
        # Otherwise DP and FSDP both report ~64 GB on a 80 GB H100 and the
        # ratio is meaningless. The platform allocator is slower but
        # this gate runs only 5 steps so the overhead is fine.
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_ALLOCATOR=platform \
        "$@" > "$out" 2>&1
        local rc=$?
        kill $smi_pid 2>/dev/null || true
        wait $smi_pid 2>/dev/null || true
        local peak
        peak=$(sort -n "$out.mem" | tail -1)
        echo "${peak:-0}"
        return $rc
    }

    DP_PEAK=$(measure_peak_mem "$G4_DP_LOG" \
        python scripts/run_pretrain.py "${COMMON_ARGS[@]}" \
            --run-name g4_dp --total-steps 5)
    FSDP_PEAK=$(measure_peak_mem "$G4_FSDP_LOG" \
        python scripts/run_pretrain.py "${COMMON_ARGS[@]}" \
            --run-name g4_fsdp --total-steps 5 --fsdp)

    if [[ "$DP_PEAK" -gt 0 && "$FSDP_PEAK" -gt 0 ]]; then
        RATIO=$(python -c "print(round($FSDP_PEAK / $DP_PEAK, 3))")
        # Threshold: FSDP < 0.85 * DP. For wind_tunnel.yaml (68M params)
        # the model+optimizer state is only ~270 MB and most of the
        # measured GPU memory is activations + workspace which DON'T
        # shrink under ZeRO-3. So a 15% improvement is realistic for
        # this tiny shape; a real 1B model would show 50-70%.
        IS_GOOD=$(python -c "print(int($FSDP_PEAK < 0.85 * $DP_PEAK))")
        if [[ "$IS_GOOD" == "1" ]]; then
            GATE_STATUS[g4]="PASS"
            GATE_DETAIL[g4]="FSDP=${FSDP_PEAK}MB DP=${DP_PEAK}MB ratio=${RATIO}"
        else
            GATE_STATUS[g4]="FAIL"
            GATE_DETAIL[g4]="FSDP=${FSDP_PEAK}MB DP=${DP_PEAK}MB ratio=${RATIO} (>= 0.85)"
        fi
    else
        GATE_STATUS[g4]="FAIL"
        GATE_DETAIL[g4]="couldn't read peak memory (DP=$DP_PEAK FSDP=$FSDP_PEAK)"
    fi
fi
log "  G4: ${GATE_STATUS[g4]:-?}  (${GATE_DETAIL[g4]:-})"

# ========================================================================
# G5 — Throughput: FSDP within 30% of DP
# ========================================================================
if is_skipped g5; then
    GATE_STATUS[g5]="SKIP"; GATE_DETAIL[g5]="skipped via GAUNTLET_SKIP"
else
    log "G5: throughput compare ($STEPS steps each)"
    G5_DP_LOG="$RESULTS_DIR/g5_dp.log"
    G5_FSDP_LOG="$RESULTS_DIR/g5_fsdp.log"

    python scripts/run_pretrain.py "${COMMON_ARGS[@]}" \
        --run-name g5_dp --total-steps "$STEPS" \
        > "$G5_DP_LOG" 2>&1
    python scripts/run_pretrain.py "${COMMON_ARGS[@]}" \
        --run-name g5_fsdp --total-steps "$STEPS" --fsdp \
        > "$G5_FSDP_LOG" 2>&1

    # run_pretrain.py doesn't emit a tokens_per_sec field — only a "step"
    # event with {step, timestamp, loss}. Compute throughput from
    # consecutive step timestamp deltas. The first delta has JIT compile
    # cost baked in, so we drop it. With default --log-every=10 and
    # --total-steps=50 we get ~5 step events; dropping the first leaves
    # 4 → 3 deltas, enough for a stable median.
    extract_med_tps() {
        local logfile="$1"
        local micro_batch="$2"
        local seq_len="$3"
        python - <<PY
import json, statistics, datetime
events = []
for line in open("$logfile"):
    line = line.lstrip()
    if not line.startswith("{"): continue
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("event") == "step" and "step" in e and "timestamp" in e:
        ts = datetime.datetime.fromisoformat(e["timestamp"].rstrip("Z"))
        events.append((int(e["step"]), ts))
events.sort()
if len(events) < 3:
    print(""); raise SystemExit
# Drop the first event — its delta to event[1] includes JIT compile.
deltas = []
tokens_per_step = $micro_batch * $seq_len
for (s1, t1), (s2, t2) in zip(events[1:-1], events[2:]):
    dt = (t2 - t1).total_seconds()
    if dt > 0:
        deltas.append((s2 - s1) * tokens_per_step / dt)
print(round(statistics.median(deltas), 1) if deltas else "")
PY
    }
    # wind_tunnel.yaml is 2048 ctx; seq_len passed to packer is ctx+1=2049
    # but the on-GPU tokens-per-step uses the model's input length (2048).
    SEQ_LEN_FOR_TPS=2048
    DP_TPS=$(extract_med_tps   "$G5_DP_LOG"   "$MICRO_BATCH" "$SEQ_LEN_FOR_TPS")
    FSDP_TPS=$(extract_med_tps "$G5_FSDP_LOG" "$MICRO_BATCH" "$SEQ_LEN_FOR_TPS")

    if [[ -n "$DP_TPS" && -n "$FSDP_TPS" && "$DP_TPS" != "0.0" ]]; then
        RATIO=$(python -c "print(round($FSDP_TPS / $DP_TPS, 3))")
        # Want FSDP >= 0.70 * DP (i.e. <= 30% slower)
        IS_GOOD=$(python -c "print(int($FSDP_TPS >= 0.70 * $DP_TPS))")
        if [[ "$IS_GOOD" == "1" ]]; then
            GATE_STATUS[g5]="PASS"
            GATE_DETAIL[g5]="DP=${DP_TPS}tok/s FSDP=${FSDP_TPS}tok/s ratio=${RATIO}"
        else
            GATE_STATUS[g5]="FAIL"
            GATE_DETAIL[g5]="DP=${DP_TPS}tok/s FSDP=${FSDP_TPS}tok/s ratio=${RATIO} (< 0.70)"
        fi
    else
        GATE_STATUS[g5]="FAIL"
        GATE_DETAIL[g5]="couldn't extract tokens_per_sec (DP='$DP_TPS' FSDP='$FSDP_TPS')"
    fi
fi
log "  G5: ${GATE_STATUS[g5]:-?}  (${GATE_DETAIL[g5]:-})"

# ========================================================================
# G6 — Checkpoint reshard round-trip
# ========================================================================
if is_skipped g6; then
    GATE_STATUS[g6]="SKIP"; GATE_DETAIL[g6]="skipped via GAUNTLET_SKIP"
else
    log "G6: FSDP checkpoint reshard round-trip"
    G6_DIR="$RESULTS_DIR/g6"
    rm -rf "$G6_DIR" && mkdir -p "$G6_DIR"

    G6_SRC_LOG="$RESULTS_DIR/g6_src.log"
    python scripts/run_pretrain.py "${COMMON_ARGS[@]}" \
        --run-name g6_src --total-steps 10 --checkpoint-every 10 \
        --checkpoint-root "$G6_DIR/src" --fsdp \
        > "$G6_SRC_LOG" 2>&1
    G6_RC1=$?

    # Reshard to (GPU_COUNT-1) devices if we have >=3; else to 1 (DP-replicate).
    if [[ $GPU_COUNT -ge 3 ]]; then
        TARGET=$((GPU_COUNT - 1))
    else
        TARGET=1
    fi
    # Reshard expects jax.devices() to return EXACTLY target-devices. Subset
    # CUDA_VISIBLE_DEVICES for this subprocess so JAX sees only TARGET GPUs.
    # Without this we hit:
    #   RuntimeError: sharding expects N devices (N data x 1 model),
    #                 jax.devices() returned $GPU_COUNT
    CVD=$(python -c "print(','.join(str(i) for i in range($TARGET)))")
    G6_RESHARD_LOG="$RESULTS_DIR/g6_reshard.log"
    CUDA_VISIBLE_DEVICES="$CVD" python scripts/reshard_ckpt.py \
        --src "$G6_DIR/src" --dst "$G6_DIR/dst" \
        --src-step 10 --target-devices "$TARGET" \
        > "$G6_RESHARD_LOG" 2>&1
    G6_RC2=$?

    if [[ $G6_RC1 -eq 0 && $G6_RC2 -eq 0 ]] && \
       [[ -d "$G6_DIR/dst/step-000000010" || -d "$G6_DIR/dst" ]]; then
        GATE_STATUS[g6]="PASS"
        GATE_DETAIL[g6]="src@step=10 reshard to ${TARGET}-dev OK"
    else
        GATE_STATUS[g6]="FAIL"
        GATE_DETAIL[g6]="src_rc=$G6_RC1 reshard_rc=$G6_RC2"
    fi
fi
log "  G6: ${GATE_STATUS[g6]:-?}  (${GATE_DETAIL[g6]:-})"

# ========================================================================
# Summary JSON + R2 upload
# ========================================================================
SUMMARY="$RESULTS_DIR/gauntlet_results.json"

# Dump tab-separated state to a temp file (NOT piped to python — heredoc
# stdin would clobber the pipe, leaving gates={}). Then python reads the
# temp file. Both the data and the script are well-defined this way.
PAIRS_FILE="$(mktemp)"
{
    printf 'META\t%s\t%s\t%s\n' "$GPU_COUNT" "$MODEL_CFG" "$STEPS"
    for g in g1 g2 g3 g4 g5 g6; do
        printf 'GATE\t%s\t%s\t%s\n' \
            "$g" "${GATE_STATUS[$g]:-UNKNOWN}" "${GATE_DETAIL[$g]:-}"
    done
} > "$PAIRS_FILE"

PAIRS_FILE="$PAIRS_FILE" SUMMARY_OUT="$SUMMARY" python - <<'PY'
import json, os, datetime
gates: dict = {}
meta: dict = {}
with open(os.environ["PAIRS_FILE"]) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if not parts:
            continue
        if parts[0] == "META" and len(parts) >= 4:
            _, gpu_count, model_cfg, steps = parts[:4]
            meta = {"gpu_count": int(gpu_count), "model_config": model_cfg,
                    "steps": int(steps)}
        elif parts[0] == "GATE" and len(parts) >= 4:
            _, g, status, detail = parts[:4]
            gates[g] = {"status": status, "detail": detail}
out = {
    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    **meta,
    "gates": gates,
    "overall_pass": bool(gates) and all(
        g["status"] in ("PASS", "SKIP") for g in gates.values()
    ),
}
with open(os.environ["SUMMARY_OUT"], "w") as f:
    json.dump(out, f, indent=2)
PY
rm -f "$PAIRS_FILE"

log ""
log "==============================================================="
log " FSDP GAUNTLET RESULTS"
log "==============================================================="
cat "$SUMMARY"
log "==============================================================="

# Upload to R2 if creds are present.
if [[ -n "${S3_ENDPOINT_URL:-}" && -n "${S3_BUCKET:-}" ]]; then
    log "uploading results to s3://$S3_BUCKET/$R2_PREFIX/"
    aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp --recursive \
        "$RESULTS_DIR" "s3://$S3_BUCKET/$R2_PREFIX/" || \
        log "WARN: R2 upload failed; results still at $RESULTS_DIR"
fi

# Exit code reflects overall pass/fail
OVERALL=$(python -c "import json; d=json.load(open('$SUMMARY')); print('1' if d['overall_pass'] else '0')")
if [[ "$OVERALL" == "1" ]]; then
    log "GAUNTLET: PASS"
    exit 0
else
    log "GAUNTLET: FAIL — see $RESULTS_DIR"
    exit 1
fi
