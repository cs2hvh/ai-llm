#!/usr/bin/env bash
# GPU pod bootstrap — analogous to pod_launch_cpu.sh but for the JAX/GPU
# training path. Sets up the venv, installs jax[cuda12] + GPU-only deps,
# runs a JAX-sees-GPUs sanity check, validates required env vars.
#
# Usage on the pod after SSH-ing in:
#
#   1. Run system pkg bootstrap (one-time, idempotent):
#        bash scripts/pod_setup_apt.sh
#
#   2. Export R2 + HF secrets:
#        export AWS_ACCESS_KEY_ID=...
#        export AWS_SECRET_ACCESS_KEY=...
#        export S3_BUCKET=llm-data
#        export S3_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
#        export HF_TOKEN=hf_...
#
#   3. (Optional) override defaults:
#        export MYLLM_REPO_URL=https://github.com/cs2hvh/ai-llm.git
#        export MYLLM_REPO_BRANCH=main
#        export MYLLM_WORKDIR=/workspace/llm-build
#        export MYLLM_INSTALL_VLLM=1     # set to 1 to also install vLLM
#                                        # for the teacher audit (Track B)
#
#   4. Run this script:
#        bash scripts/pod_launch_gpu.sh
#
# What it does (in order):
#   - Verifies nvidia-smi / driver version
#   - Clones or updates llm-build at $MYLLM_WORKDIR
#   - Creates .venv with python3.12 (fallback: python3)
#   - Installs requirements.txt (CPU baseline)
#   - Installs requirements-gpu.txt (jax[cuda12])
#   - Optionally installs vLLM (for Track B teacher audit)
#   - Pulls tokenizer + decontam indexes from R2
#   - Runs JAX-sees-GPUs sanity check (exits non-zero if no CUDA devices)
#   - Prints a "next steps" banner
set -euo pipefail

REPO_URL="${MYLLM_REPO_URL:-https://github.com/cs2hvh/ai-llm.git}"
REPO_BRANCH="${MYLLM_REPO_BRANCH:-main}"
WORKDIR="${MYLLM_WORKDIR:-/workspace/llm-build}"
INSTALL_VLLM="${MYLLM_INSTALL_VLLM:-0}"
TOKENIZER_KEY="${MYLLM_TOKENIZER_KEY:-tokenizer/myllm-spm-unigram-131k-v2.json}"
DECON_KEY_PRIMARY="${MYLLM_DECON_INDEX_KEY_PRIMARY:-decontamination/decontamination_index_8gram.json}"
DECON_KEY_SECONDARY="${MYLLM_DECON_INDEX_KEY_SECONDARY:-decontamination/decontamination_index_13gram.json}"

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*"; exit 1; }

# ----------------------------------------------------------------------
# 1. NVIDIA driver / GPUs visible
# ----------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found. Is this actually a GPU pod?"
fi

DRIVER_LINE=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
log "nvidia driver: $DRIVER_LINE"
log "gpu count: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'

if [[ "$GPU_COUNT" -lt 1 ]]; then
    die "no GPUs detected by nvidia-smi"
fi

# Soft warn on old drivers (jax[cuda12]+CUDA 12.4 wants >=550; 12.9 wants >=575)
DRIVER_MAJOR=$(echo "$DRIVER_LINE" | cut -d. -f1)
if [[ -n "$DRIVER_MAJOR" && "$DRIVER_MAJOR" -lt 535 ]]; then
    log "WARNING: driver $DRIVER_LINE is older than 535; jax[cuda12] may fail."
fi

# ----------------------------------------------------------------------
# 2. R2 + HF secrets sanity
# ----------------------------------------------------------------------
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY S3_BUCKET S3_ENDPOINT_URL HF_TOKEN; do
    if [[ -z "${!v:-}" ]]; then
        die "$v not exported. See script header for the 5 required env vars."
    fi
done

# ----------------------------------------------------------------------
# 3. AWS CLI present? (pod_setup_apt.sh installs it; this is a safety net)
# ----------------------------------------------------------------------
if ! command -v aws >/dev/null 2>&1; then
    log "aws not in PATH; falling back to install"
    bash "$(dirname "$0")/pod_setup_apt.sh"
fi

# ----------------------------------------------------------------------
# 4. Clone / refresh the repo
# ----------------------------------------------------------------------
mkdir -p /workspace
if [[ -d "$WORKDIR/.git" ]]; then
    log "repo exists at $WORKDIR; fetching latest $REPO_BRANCH"
    (cd "$WORKDIR" && git fetch origin "$REPO_BRANCH" && git reset --hard "origin/$REPO_BRANCH")
else
    log "cloning $REPO_URL @ $REPO_BRANCH -> $WORKDIR"
    rm -rf "$WORKDIR"
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
HEAD_SHA=$(git rev-parse HEAD)
log "at HEAD=$HEAD_SHA"

# ----------------------------------------------------------------------
# 5. Python venv
# ----------------------------------------------------------------------
PY_BIN="${MYLLM_PYTHON:-python3.12}"
command -v "$PY_BIN" >/dev/null || PY_BIN="python3.11"
command -v "$PY_BIN" >/dev/null || PY_BIN="python3"
command -v "$PY_BIN" >/dev/null || die "no usable python3 in PATH"
log "creating venv with $PY_BIN ($($PY_BIN --version 2>&1))"

if [[ ! -d "$WORKDIR/.venv" ]]; then
    "$PY_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel setuptools

# ----------------------------------------------------------------------
# 6. CPU baseline deps + GPU deps
# ----------------------------------------------------------------------
log "installing CPU baseline deps (requirements.txt)"
pip install --quiet -r requirements.txt

log "installing GPU deps (requirements-gpu.txt — jax[cuda12])"
# Important: don't let LD_LIBRARY_PATH from a pre-installed PyTorch CUDA
# interfere with jax's bundled libs. Unset for the pip resolution step.
LD_LIBRARY_PATH_BAK="${LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH || true
pip install --quiet -r requirements-gpu.txt
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_BAK"

# ----------------------------------------------------------------------
# 7. Optional: vLLM for the teacher audit
# ----------------------------------------------------------------------
if [[ "$INSTALL_VLLM" == "1" ]]; then
    log "installing vLLM (for Track B teacher audit)"
    # vLLM 0.6+ supports CUDA 12.1+; 0.7+ requires CUDA 12.4+.
    pip install --quiet "vllm>=0.6.0"
fi

# ----------------------------------------------------------------------
# 8. Pull tokenizer + decontam indexes
# ----------------------------------------------------------------------
mkdir -p artifacts
if [[ ! -f artifacts/tokenizer_v1.json ]]; then
    log "pulling tokenizer $TOKENIZER_KEY"
    aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp \
        "s3://$S3_BUCKET/$TOKENIZER_KEY" artifacts/tokenizer_v1.json
fi
log "tokenizer size: $(stat -c%s artifacts/tokenizer_v1.json) bytes"

for KEY_LOCAL in \
    "$DECON_KEY_PRIMARY:artifacts/decontamination_index_8gram.json" \
    "$DECON_KEY_SECONDARY:artifacts/decontamination_index_13gram.json" \
; do
    KEY="${KEY_LOCAL%%:*}"
    LOCAL="${KEY_LOCAL##*:}"
    if [[ ! -f "$LOCAL" ]]; then
        log "pulling $KEY"
        aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp "s3://$S3_BUCKET/$KEY" "$LOCAL"
    fi
done

# ----------------------------------------------------------------------
# 9. JAX-sees-GPUs sanity check
# ----------------------------------------------------------------------
log "JAX GPU sanity check"
python - <<'PY'
import jax, sys
devs = jax.devices()
print(f"jax.default_backend(): {jax.default_backend()}")
print(f"jax.devices(): {devs}")
gpu_devs = [d for d in devs if d.platform == "gpu"]
if not gpu_devs:
    print("FATAL: jax sees no GPUs. Check driver + CUDA installation.")
    sys.exit(2)
print(f"GPUs visible to JAX: {len(gpu_devs)}")
PY

# ----------------------------------------------------------------------
# 10. Done
# ----------------------------------------------------------------------
log ""
log "==============================================================="
log " GPU pod ready. Repo at: $WORKDIR (HEAD=$HEAD_SHA)"
log "==============================================================="
log " Next steps:"
log "  - FSDP gauntlet:    bash scripts/run_fsdp_gauntlet.sh"
log "  - Teacher audit:    bash scripts/run_teacher_audit.sh  (after vLLM install)"
log "  - Activate venv:    source $WORKDIR/.venv/bin/activate"
log "==============================================================="
