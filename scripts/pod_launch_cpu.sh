#!/usr/bin/env bash
# CPU pod bootstrap + B2 corpus build — analogous to pod_launch.sh (which is
# the GPU/H200 sweep launcher). This one runs the offline packed-corpus
# build on a CPU-only pod: pulls llm-build code, installs deps, builds each
# per-source corpus in turn (with R2 streaming so disk doesn't fill), then
# runs the cross-source compose pass.
#
# Usage on the pod after SSH-ing in:
#
#   1. Export these secrets (paste your values):
#        export AWS_ACCESS_KEY_ID=...
#        export AWS_SECRET_ACCESS_KEY=...
#        export S3_BUCKET=llm-data
#        export S3_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
#        export HF_TOKEN=...
#
#   2. (Optional) override defaults:
#        export MYLLM_CORPUS_NAME=corpus_v1            # R2 path prefix
#        export MYLLM_SAMPLE_LIMIT_PER_SOURCE=         # cap docs/source for smoke
#        export MYLLM_SEQUENCE_LENGTH=8192             # match model context+1
#        export MYLLM_SEQUENCES_PER_SHARD=65536        # ~536M tokens/shard at 8192
#        export MYLLM_DELETE_LOCAL=true                # stream-to-R2 mode
#        export MYLLM_HF_REVISION=                     # pin HF dataset revisions
#        export MYLLM_SOURCES="HuggingFaceFW/fineweb-edu pg19 ..."  # subset to build
#
#   3. curl + bash this script:
#        curl -fsSL https://<r2-public-or-presigned>/pod_launch_cpu.sh | bash
#      OR if already pulled:
#        bash pod_launch_cpu.sh
#
# What it does (in order):
#   - Installs awscli (if missing), python deps from requirements.txt
#   - Pulls llm-build code tarball from R2 → /workspace/llm-build
#   - Pulls production tokenizer from R2 → artifacts/tokenizer_v1.json
#   - Pulls (or builds) the decontamination index → artifacts/
#   - For each source in configs/data/pretrain_mix.yaml:
#       runs scripts/build_packed_corpus.py with --r2-prefix +
#       optional --delete-local-after-upload
#   - Runs scripts/compose_mixed_corpus.py to produce the training-time
#     mixed corpus, also mirroring to R2
#   - Prints a JSON summary at the end
set -euo pipefail

CODE_KEY="${MYLLM_CODE_KEY:-code/llm-build-latest.tar.gz}"
TOKENIZER_KEY="${MYLLM_TOKENIZER_KEY:-tokenizer/myllm-spm-unigram-131k-v2.json}"
WORKDIR="${MYLLM_WORKDIR:-/workspace/llm-build}"

CORPUS_NAME="${MYLLM_CORPUS_NAME:-corpus_v1}"
SEQUENCE_LENGTH="${MYLLM_SEQUENCE_LENGTH:-8192}"
SEQUENCES_PER_SHARD="${MYLLM_SEQUENCES_PER_SHARD:-65536}"
SAMPLE_LIMIT="${MYLLM_SAMPLE_LIMIT_PER_SOURCE:-}"   # blank = no limit
DELETE_LOCAL="${MYLLM_DELETE_LOCAL:-true}"          # default ON to keep disk bounded
HF_REVISION="${MYLLM_HF_REVISION:-}"                # blank = HF default (main)
SOURCES_OVERRIDE="${MYLLM_SOURCES:-}"               # blank = all from pretrain_mix.yaml

# R2 path structure:
#   <S3_BUCKET>/<CORPUS_NAME>/sources/<source-id>/shard-NNNNNN/...
#   <S3_BUCKET>/<CORPUS_NAME>/train/shard-NNNNNN/...
SOURCES_R2_PREFIX="${CORPUS_NAME}/sources"
TRAIN_R2_PREFIX="${CORPUS_NAME}/train"

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*"; exit 1; }

# ----------------------------------------------------------------------
# Secrets sanity check
# ----------------------------------------------------------------------
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY S3_BUCKET S3_ENDPOINT_URL HF_TOKEN; do
    if [[ -z "${!v:-}" ]]; then
        die "$v is not exported. Set all five before running:
        AWS_ACCESS_KEY_ID  AWS_SECRET_ACCESS_KEY  S3_BUCKET  S3_ENDPOINT_URL  HF_TOKEN"
    fi
done

# ----------------------------------------------------------------------
# 1. AWS CLI (S3-compatible R2 access)
# ----------------------------------------------------------------------
if ! command -v aws >/dev/null 2>&1; then
    log "installing AWS CLI v2"
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq unzip curl
    fi
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip
    (cd /tmp && unzip -q awscli.zip && ./aws/install --update)
    rm -rf /tmp/awscli.zip /tmp/aws
fi
log "aws: $(aws --version 2>&1 | head -1)"

# ----------------------------------------------------------------------
# 2. Pull + extract code tarball (or git clone if MYLLM_REPO_URL set)
# ----------------------------------------------------------------------
mkdir -p /workspace
cd /workspace

if [[ -n "${MYLLM_REPO_URL:-}" ]]; then
    log "git-cloning from $MYLLM_REPO_URL"
    rm -rf "$WORKDIR"
    git clone --depth 1 "$MYLLM_REPO_URL" "$WORKDIR"
else
    log "pulling code tarball $CODE_KEY from R2"
    aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp \
        "s3://$S3_BUCKET/$CODE_KEY" /tmp/llm-build-code.tar.gz
    rm -rf "$WORKDIR"
    tar -xzf /tmp/llm-build-code.tar.gz -C /workspace
    rm -f /tmp/llm-build-code.tar.gz
fi
[[ -f "$WORKDIR/pyproject.toml" ]] || die "$WORKDIR/pyproject.toml not found after extract"

# ----------------------------------------------------------------------
# 3. Python venv + deps (CPU-only — much lighter than GPU bootstrap)
# ----------------------------------------------------------------------
cd "$WORKDIR"
PY_BIN="${MYLLM_PYTHON:-python3.11}"
command -v "$PY_BIN" >/dev/null || PY_BIN="python3"
log "creating venv with $PY_BIN"
"$PY_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel >/dev/null
pip install -r requirements.txt
log "deps installed; python=$(python --version 2>&1)"

# ----------------------------------------------------------------------
# 4. Pull tokenizer
# ----------------------------------------------------------------------
mkdir -p artifacts
if [[ ! -f artifacts/tokenizer_v1.json ]]; then
    log "pulling tokenizer $TOKENIZER_KEY"
    aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp \
        "s3://$S3_BUCKET/$TOKENIZER_KEY" artifacts/tokenizer_v1.json
fi
log "tokenizer size: $(stat -c%s artifacts/tokenizer_v1.json) bytes"

# ----------------------------------------------------------------------
# 5. Optional: pre-built decontamination index
# ----------------------------------------------------------------------
DECON_KEY="${MYLLM_DECON_INDEX_KEY:-decontamination/index_v1.json}"
if aws --endpoint-url "$S3_ENDPOINT_URL" s3 ls "s3://$S3_BUCKET/$DECON_KEY" \
        >/dev/null 2>&1; then
    log "pulling pre-built decontamination index"
    aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp \
        "s3://$S3_BUCKET/$DECON_KEY" artifacts/decontamination_index.json
else
    log "no pre-built decon index found; build will skip per-source decontamination "
    log "(or live-build is possible but slow; pass --no-decontam to be explicit)"
fi

# ----------------------------------------------------------------------
# 6. Determine source list (default = all from pretrain_mix.yaml)
# ----------------------------------------------------------------------
if [[ -n "$SOURCES_OVERRIDE" ]]; then
    # shellcheck disable=SC2206
    SOURCES=($SOURCES_OVERRIDE)
else
    SOURCES=( $(python -c "
import yaml
cfg = yaml.safe_load(open('configs/data/pretrain_mix.yaml'))
for s in cfg.get('sources', []):
    print(s['dataset'])
") )
fi
log "building ${#SOURCES[@]} sources: ${SOURCES[*]}"

# ----------------------------------------------------------------------
# 7. Per-source builds (sequential — each saturates Rust tokenizers'
#    internal Rayon parallelism on this many cores)
# ----------------------------------------------------------------------
mkdir -p /workspace/corpus/sources

BUILD_ARGS_COMMON=(
    --output-root /workspace/corpus/sources
    --sequence-length "$SEQUENCE_LENGTH"
    --sequences-per-shard "$SEQUENCES_PER_SHARD"
    --r2-prefix "$SOURCES_R2_PREFIX"
)
if [[ "$DELETE_LOCAL" == "true" ]]; then
    BUILD_ARGS_COMMON+=( --delete-local-after-upload )
fi
if [[ -n "$SAMPLE_LIMIT" ]]; then
    BUILD_ARGS_COMMON+=( --sample-limit "$SAMPLE_LIMIT" )
fi
if [[ -n "$HF_REVISION" ]]; then
    BUILD_ARGS_COMMON+=( --hf-revision "$HF_REVISION" )
fi
if [[ -f artifacts/decontamination_index.json ]]; then
    # decontamination is config-driven — set decontamination.index_path
    # in pretrain_mix.yaml's decontamination block to use it.
    log "decontamination index present at artifacts/decontamination_index.json"
fi

SUMMARY_DIR=/workspace/corpus/build_summaries
mkdir -p "$SUMMARY_DIR"

for src in "${SOURCES[@]}"; do
    log "==== source: $src ===="
    src_safe=$(echo "$src" | tr '/' '_')
    summary_path="$SUMMARY_DIR/${src_safe}.json"
    # Use --revision-id with the date-stamp so the manifest records when
    # this source was built. HF revision pin is separate (--hf-revision).
    rev_id="build-$(date -u +%Y%m%d)"
    python scripts/build_packed_corpus.py \
        --source "$src" \
        --revision-id "$rev_id" \
        "${BUILD_ARGS_COMMON[@]}" \
        > "$summary_path" || {
            log "FAILED: $src — see $summary_path"; exit 1
        }
    log "done: $src — summary at $summary_path"
done

# ----------------------------------------------------------------------
# 8. Compose mixed-training corpus.
#    With --delete-local-after-upload, the per-source corpora aren't on
#    disk anymore — compose has to read from R2. For now: re-download
#    each source's manifest+shards on demand, OR build with delete=false
#    and run compose locally. We default to local compose; for true
#    stream-to-R2 mode, the operator should set MYLLM_DELETE_LOCAL=false
#    when they intend to compose on this same pod.
# ----------------------------------------------------------------------
if [[ "$DELETE_LOCAL" == "true" ]]; then
    log "MYLLM_DELETE_LOCAL=true — per-source corpora streamed to R2;"
    log "compose pass needs local copies. Either:"
    log "  - re-run this script with MYLLM_DELETE_LOCAL=false to keep local, OR"
    log "  - manually: aws s3 sync s3://$S3_BUCKET/$SOURCES_R2_PREFIX/ /workspace/corpus/sources/"
    log "  - and then: python scripts/compose_mixed_corpus.py ..."
    log "skipping compose pass on this pod."
else
    log "==== composing mixed-training corpus ===="
    mkdir -p /workspace/corpus/train
    python scripts/compose_mixed_corpus.py \
        --sources-root /workspace/corpus/sources \
        --output-dir /workspace/corpus/train \
        --sequences-per-shard "$SEQUENCES_PER_SHARD" \
        > "$SUMMARY_DIR/compose.json"
    log "compose done — uploading to R2"
    aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
        /workspace/corpus/train/ \
        "s3://$S3_BUCKET/$TRAIN_R2_PREFIX/"
    log "training corpus uploaded to s3://$S3_BUCKET/$TRAIN_R2_PREFIX/"
fi

# ----------------------------------------------------------------------
# 9. Upload build summaries to R2 for the operator's record
# ----------------------------------------------------------------------
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
    "$SUMMARY_DIR/" \
    "s3://$S3_BUCKET/$CORPUS_NAME/build_summaries/"

log "DONE. Build summaries at s3://$S3_BUCKET/$CORPUS_NAME/build_summaries/"
log "Stop the pod from the RunPod dashboard to halt billing."
