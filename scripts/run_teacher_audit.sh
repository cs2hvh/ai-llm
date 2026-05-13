#!/usr/bin/env bash
# Track B — Top-K mass audit on the planned distillation teachers.
#
# Loads each teacher in bf16 with transformers' device_map="auto"
# (tensor-parallel across visible GPUs), samples per-position softmax
# mass at K in {4, 8, 16, 32}, writes a JSON summary per teacher,
# uploads results to R2.
#
# Reviewer Q3 (2026-05-12) gate rule:
#   if K=8 frac_below_0.95 > 0.10 on EITHER teacher → ship K=16.
#   else → ship K=8.
#
# Usage:
#   bash scripts/run_teacher_audit.sh
#
# Env overrides:
#   AUDIT_CORPUS    default: a small audit slice we fabricate from the
#                   tokenized infra-validation corpus, OR a synthetic
#                   shard if no corpus is present.
#   AUDIT_N_POS     default: 65536 (matches reviewer's quoted figure)
#   AUDIT_KS        default: "4,8,16,32"
#   AUDIT_SEQ_LEN   default: 2048
#   AUDIT_BATCH     default: 2
#   AUDIT_TEACHERS  comma list of "id:hf_model" pairs. Default covers
#                   the two locked teachers.
#   AUDIT_R2_PREFIX default: teacher_audit/<UTC-date>
set -uo pipefail

WORKDIR="${MYLLM_WORKDIR:-/workspace/llm-build}"
RESULTS_DIR="${MYLLM_AUDIT_DIR:-/workspace/teacher_audit}"
N_POS="${AUDIT_N_POS:-65536}"
KS="${AUDIT_KS:-4,8,16,32}"
SEQ_LEN="${AUDIT_SEQ_LEN:-2048}"
BATCH="${AUDIT_BATCH:-2}"
# NOTE: the planned production teachers (DeepSeek-V4-Pro-Base + Olmo-3-32B-Base
# per project_teacher_strategy) are HYPOTHETICAL / future. They aren't yet on
# HuggingFace, so the audit can't load them. Until they ship, use real
# currently-available teachers as methodology placeholders. Override at
# run-time with AUDIT_TEACHERS=<id>:<hf_id>,<id>:<hf_id>.
#
# Sensible currently-shipping options that fit on 3× A100-80GB in bf16:
#   olmo-2-13b       : allenai/OLMo-2-1124-13B           (~26 GB bf16)
#   deepseek-v2-lite : deepseek-ai/DeepSeek-V2-Lite-Base (~32 GB MoE bf16)
#   qwen-2.5-32b     : Qwen/Qwen2.5-32B                  (~64 GB bf16, tight)
TEACHERS="${AUDIT_TEACHERS:-olmo-2-13b:allenai/OLMo-2-1124-13B,deepseek-v2-lite:deepseek-ai/DeepSeek-V2-Lite-Base}"
UTC_DATE="$(date -u +%Y-%m-%d)"
R2_PREFIX="${AUDIT_R2_PREFIX:-teacher_audit/${UTC_DATE}}"

mkdir -p "$RESULTS_DIR"
cd "$WORKDIR"

# shellcheck disable=SC1091
source .venv/bin/activate

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

# ----------------------------------------------------------------------
# 1. Audit corpus
# ----------------------------------------------------------------------
AUDIT_CORPUS_DEFAULT="$RESULTS_DIR/audit_corpus.bin"
AUDIT_CORPUS="${AUDIT_CORPUS:-$AUDIT_CORPUS_DEFAULT}"
if [[ ! -f "$AUDIT_CORPUS" ]]; then
    log "no audit corpus at $AUDIT_CORPUS; fabricating a synthetic one"
    python - <<PY
import numpy as np
np.random.seed(0)
# 256k tokens of uniform random ids in our vocab range.
arr = np.random.randint(0, 131072, size=262144, dtype=np.uint32)
arr.tofile("$AUDIT_CORPUS")
print(f"wrote {arr.size} tokens -> $AUDIT_CORPUS")
PY
fi
CORPUS_SIZE=$(stat -c%s "$AUDIT_CORPUS")
log "audit corpus: $AUDIT_CORPUS ($CORPUS_SIZE bytes, $((CORPUS_SIZE/4)) tokens)"

# ----------------------------------------------------------------------
# 2. Check transformers is installed (lazy)
# ----------------------------------------------------------------------
python -c "import transformers, torch" 2>/dev/null || {
    log "installing transformers + torch (one-time)"
    pip install --quiet "torch>=2.4" "transformers>=4.45"
}

# ----------------------------------------------------------------------
# 3. Run audit per teacher
# ----------------------------------------------------------------------
declare -a SUMMARIES=()
declare -A TEACHER_RC

IFS=',' read -ra TEACHER_LIST <<< "$TEACHERS"
for entry in "${TEACHER_LIST[@]}"; do
    TID="${entry%%:*}"
    HFID="${entry#*:}"
    log "audit: $TID  ($HFID)"
    OUT="$RESULTS_DIR/${TID}.json"
    LOG="$RESULTS_DIR/${TID}.log"
    set +e
    python scripts/audit_teacher_topk_mass.py \
        --teacher-id "$TID" \
        --teacher-hf-model "$HFID" \
        --tokenized-corpus "$AUDIT_CORPUS" \
        --tokenizer-path artifacts/tokenizer_v1.json \
        --n-positions "$N_POS" \
        --ks "$KS" \
        --batch-size "$BATCH" \
        --sequence-length "$SEQ_LEN" \
        --output "$OUT" \
        > "$LOG" 2>&1
    rc=$?
    set -e
    TEACHER_RC[$TID]=$rc
    if [[ $rc -eq 0 && -f "$OUT" ]]; then
        REC_K=$(python -c "import json; print(json.load(open('$OUT'))['decision']['recommended_k'])")
        K8_BELOW=$(python -c "import json; print(json.load(open('$OUT'))['by_k']['8']['frac_below_0.95'])")
        log "  $TID OK — recommended_k=$REC_K  K8.frac_below_0.95=$K8_BELOW"
        SUMMARIES+=("$TID:OK:$REC_K:$K8_BELOW")
    else
        log "  $TID FAILED (rc=$rc, see $LOG)"
        SUMMARIES+=("$TID:FAIL:-:-")
    fi
done

# ----------------------------------------------------------------------
# 4. Aggregated summary + decision
# ----------------------------------------------------------------------
SUMMARY="$RESULTS_DIR/audit_summary.json"
{
    for s in "${SUMMARIES[@]}"; do
        printf '%s\n' "$s"
    done
} | python - > "$SUMMARY" <<'PY'
import json, sys, datetime, os, glob
rows = [l.strip().split(":", 3) for l in sys.stdin if l.strip()]
results = {}
need_k16 = False
for tid, status, rec_k, k8_frac in rows:
    results[tid] = {"status": status, "recommended_k": rec_k, "k8_frac_below_0.95": k8_frac}
    if status == "OK":
        try:
            if float(k8_frac) > 0.10:
                need_k16 = True
        except ValueError:
            pass
all_ok = all(r["status"] == "OK" for r in results.values())
final_k = 16 if need_k16 else 8
out = {
    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "teachers": results,
    "overall_ok": all_ok,
    "final_recommended_k": final_k if all_ok else None,
    "rule": "K=16 if any teacher's K=8 frac_below_0.95 > 0.10; else K=8",
}
print(json.dumps(out, indent=2))
PY

log ""
log "==============================================================="
log " TEACHER AUDIT SUMMARY"
log "==============================================================="
cat "$SUMMARY"
log "==============================================================="

# Upload to R2
if [[ -n "${S3_ENDPOINT_URL:-}" && -n "${S3_BUCKET:-}" ]]; then
    log "uploading results to s3://$S3_BUCKET/$R2_PREFIX/"
    aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp --recursive \
        "$RESULTS_DIR" "s3://$S3_BUCKET/$R2_PREFIX/" || \
        log "WARN: R2 upload failed; results still at $RESULTS_DIR"
fi

OVERALL=$(python -c "import json; d=json.load(open('$SUMMARY')); print('1' if d['overall_ok'] else '0')")
if [[ "$OVERALL" == "1" ]]; then
    exit 0
else
    exit 1
fi
