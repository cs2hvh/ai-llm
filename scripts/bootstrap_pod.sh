#!/usr/bin/env bash
# Bootstrap a stock RunPod pytorch image into a working myllm dev/train env.
#
# Why: until we push our custom Docker image to a registry (Docker Hub /
# GHCR), each pod needs to install the project deps fresh. This script
# does that idempotently, so SSH-and-run is the same regardless of pod.
#
# Usage on a fresh pod:
#   curl -fsSL <some-host>/bootstrap_pod.sh | bash
#   # OR copy this file in via scp/git and run:
#   bash bootstrap_pod.sh
#
# Assumptions:
#   - Stock RunPod pytorch image (Python 3.10-3.12, pip, git pre-installed).
#   - GPU image with NVIDIA driver + CUDA 12.x already installed.
#   - Network access to PyPI and Hugging Face.
#
# Exit codes:
#   0  — bootstrap complete; venv at /workspace/llm-build/.venv ready.
#   1  — install failure; logs above will show what.
#   2  — env-var sanity check failed (HF_TOKEN etc. not exported).
set -euo pipefail

REPO_URL="${MYLLM_REPO_URL:-https://github.com/<org>/llm-build.git}"
WORKDIR="${MYLLM_WORKDIR:-/workspace/llm-build}"
PY_BIN="${MYLLM_PYTHON:-python3.11}"

log() { printf '[bootstrap %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ----------------------------------------------------------------------
# 1. Sanity-check environment and tools.
# ----------------------------------------------------------------------
log "checking prerequisites"
command -v git >/dev/null 2>&1 || { log "FATAL: git not found"; exit 1; }
command -v "$PY_BIN" >/dev/null 2>&1 || PY_BIN="python3"
command -v "$PY_BIN" >/dev/null 2>&1 || { log "FATAL: no python3 found"; exit 1; }

PY_VERSION="$($PY_BIN -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
log "python = $PY_BIN ($PY_VERSION)"

# ----------------------------------------------------------------------
# 2. Clone or update the repo.
# ----------------------------------------------------------------------
if [[ -d "$WORKDIR/.git" ]]; then
    log "repo exists at $WORKDIR; pulling"
    git -C "$WORKDIR" fetch --all --prune
    git -C "$WORKDIR" pull --ff-only
elif [[ -f "$WORKDIR/pyproject.toml" ]]; then
    log "repo present at $WORKDIR (not a git checkout); skipping clone"
else
    log "cloning $REPO_URL into $WORKDIR"
    git clone --depth 1 "$REPO_URL" "$WORKDIR"
fi

cd "$WORKDIR"

# ----------------------------------------------------------------------
# 3. Create venv and install deps.
# ----------------------------------------------------------------------
if [[ ! -d .venv ]]; then
    log "creating venv at .venv"
    "$PY_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "upgrading pip / wheel"
pip install --upgrade pip wheel >/dev/null

REQS_FILE="requirements.txt"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || [[ -f /proc/driver/nvidia/version ]]; then
    if [[ -f requirements-gpu.txt ]]; then
        REQS_FILE="requirements-gpu.txt"
        log "GPU detected — using $REQS_FILE"
    fi
fi

log "installing project + deps from $REQS_FILE"
pip install -r "$REQS_FILE"
pip install -e .

# ----------------------------------------------------------------------
# 4. Sanity-check imports + GPU visibility.
# ----------------------------------------------------------------------
log "running smoke_test_env.py"
python scripts/smoke_test_env.py || {
    log "FATAL: smoke_test_env.py failed"
    exit 1
}

# ----------------------------------------------------------------------
# 5. Optional: verify required env vars are set.
# ----------------------------------------------------------------------
MISSING=()
for v in HF_TOKEN RUNPOD_API_KEY WANDB_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY S3_ENDPOINT_URL S3_BUCKET; do
    if [[ -z "${!v:-}" ]]; then
        MISSING+=("$v")
    fi
done
if (( ${#MISSING[@]} > 0 )); then
    log "WARN: the following env vars are not set: ${MISSING[*]}"
    log "      set them via 'set -a && source .env && set +a' before running training"
fi

log "bootstrap complete; venv ready at $WORKDIR/.venv"
log "next: cd $WORKDIR && source .venv/bin/activate"
