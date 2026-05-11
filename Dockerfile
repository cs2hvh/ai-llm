# MyLLM training image — CUDA 12 + Python 3.11 + JAX[cuda12] + the project package.
#
# Built locally or on RunPod build infra; pushed to a registry RunPod can pull from.
# Layers ordered for cache hits: rarely-changing system deps first, then python deps,
# then project source last.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    KERAS_BACKEND=jax \
    MYLLM_LOG_FORMAT=json \
    HF_HOME=/workspace/.cache/huggingface \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# System packages — Python, build tools, SSH (RunPod ships its own sshd typically)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git gnupg \
        build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python deps — copy requirements first for build-cache friendliness
COPY requirements.txt requirements-gpu.txt ./
RUN python -m pip install -U pip setuptools wheel \
    && python -m pip install -r requirements.txt \
    && python -m pip install -r requirements-gpu.txt

# Project package — README.md is referenced from pyproject.toml so it must
# be present at metadata-generation time, even though the runtime doesn't use it.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN python -m pip install -e .

# Scripts and configs
COPY scripts/ scripts/
COPY configs/ configs/
COPY docs/ docs/

# Health/version probe — RunPod start command typically overrides this.
CMD ["python", "-c", "import myllm, jax; \
print('myllm', myllm.__version__, 'jax', jax.__version__); \
print('devices:', jax.devices())"]
