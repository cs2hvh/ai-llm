#!/usr/bin/env bash
# System-package bootstrap for a fresh pod.
#
# Installs the small constellation of tools we always want on a pod:
# editor, JSON parser, monitoring, terminal multiplexer, disk-usage,
# unzip, curl, git, jq, plus AWS CLI v2 (for R2 access).
#
# Idempotent — safe to re-run. Each package is installed only if missing.
#
# Usage (on the pod, before pod_launch_gpu.sh):
#   bash scripts/pod_setup_apt.sh
#
# Why a separate script (vs. inline in pod_launch_gpu.sh):
#   - Re-runnable on its own when you forgot to install something
#   - apt's stdout is noisy; isolating it keeps pod_launch_gpu.sh's log clean
#   - You can drop this on top of any RunPod image regardless of CUDA
set -euo pipefail

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*"; exit 1; }

# ----------------------------------------------------------------------
# 1. APT — quality-of-life + build essentials
# ----------------------------------------------------------------------
APT_PKGS=(
    # Core tools
    nano vim jq curl wget unzip git ca-certificates
    # Monitoring / debugging
    htop tmux ncdu lsof psmisc procps tree
    # Network
    iputils-ping dnsutils netcat-openbsd
    # Build (some pip wheels need these)
    build-essential pkg-config
    # Python build-deps (rare; some wheels still need these for sdist)
    python3-dev
)

# RunPod's default images run as root; sudo is usually absent and unneeded.
SUDO=""
if [[ $EUID -ne 0 ]]; then
    command -v sudo >/dev/null && SUDO="sudo" || die "not root and no sudo available"
fi

if ! command -v apt-get >/dev/null 2>&1; then
    die "apt-get not found; this script expects a Debian/Ubuntu base image"
fi

log "apt-get update"
DEBIAN_FRONTEND=noninteractive $SUDO apt-get update -qq

log "installing ${#APT_PKGS[@]} apt packages: ${APT_PKGS[*]}"
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends \
    "${APT_PKGS[@]}"

# ----------------------------------------------------------------------
# 2. AWS CLI v2 — only if not already present
# ----------------------------------------------------------------------
if ! command -v aws >/dev/null 2>&1; then
    log "installing AWS CLI v2"
    cd /tmp
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64) AWS_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" ;;
        aarch64) AWS_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" ;;
        *) die "unsupported arch for AWS CLI v2: $ARCH" ;;
    esac
    curl -fsSL "$AWS_URL" -o /tmp/awscli.zip
    unzip -q /tmp/awscli.zip -d /tmp
    $SUDO /tmp/aws/install --update
    rm -rf /tmp/awscli.zip /tmp/aws
fi
log "aws: $(aws --version 2>&1 | head -1)"

# ----------------------------------------------------------------------
# 3. Report
# ----------------------------------------------------------------------
log "system packages installed."
log "next: bash scripts/pod_launch_gpu.sh"
