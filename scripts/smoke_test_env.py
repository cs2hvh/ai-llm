#!/usr/bin/env python3
"""Phase 0 smoke test: verify the orchestration-VM environment is ready.

Runs on CPU only. Reports installed dependency versions, checks expected env
vars (without printing their values), and confirms the project package imports.
Exits non-zero if any required check fails.
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from typing import NamedTuple


REQUIRED_PACKAGES = [
    "keras",
    "jax",
    "jaxlib",
    "numpy",
    "tokenizers",
    "datasets",
    "transformers",
    "optax",
    "orbax.checkpoint",
    "wandb",
    "runpod",
    "yaml",  # PyYAML
    "pydantic",
    "boto3",
]

OPTIONAL_ENV_VARS = [
    "HF_TOKEN",
    "RUNPOD_API_KEY",
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_ENDPOINT_URL",
    "S3_BUCKET",
]


class Check(NamedTuple):
    name: str
    ok: bool
    detail: str


def check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return Check("python>=3.11", ok, f"{platform.python_version()} on {platform.system()}")


def check_import(pkg: str) -> Check:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "unknown")
        return Check(f"import {pkg}", True, version)
    except Exception as e:  # noqa: BLE001
        return Check(f"import {pkg}", False, f"{type(e).__name__}: {e}")


def check_myllm() -> Check:
    try:
        import myllm  # noqa: F401
        from myllm.model import ModelConfig
        cfg = ModelConfig.from_yaml("configs/pilot_250m.yaml")
        return Check("myllm package + pilot config load", True, f"params≈{cfg.param_count_estimate():,}")
    except Exception as e:  # noqa: BLE001
        return Check("myllm package + pilot config load", False, f"{type(e).__name__}: {e}")


def check_env_var(var: str) -> Check:
    present = bool(os.environ.get(var))
    return Check(f"env: {var}", present, "set" if present else "missing")


def check_jax_devices() -> Check:
    try:
        import jax
        devices = jax.devices()
        return Check("jax.devices()", True, f"{len(devices)} device(s): {[d.platform for d in devices]}")
    except Exception as e:  # noqa: BLE001
        return Check("jax.devices()", False, f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 70)
    print("MyLLM Phase 0 environment smoke test")
    print("=" * 70)

    checks: list[Check] = [check_python()]
    checks += [check_import(p) for p in REQUIRED_PACKAGES]
    checks.append(check_myllm())
    checks.append(check_jax_devices())

    print("\nRequired:")
    for c in checks:
        mark = "OK" if c.ok else "FAIL"
        print(f"  [{mark:>4}] {c.name:<45} {c.detail}")

    print("\nOptional env vars (not failed if missing — just informational):")
    for v in OPTIONAL_ENV_VARS:
        c = check_env_var(v)
        mark = "set" if c.ok else "—"
        print(f"  [{mark:>4}] {c.name}")

    failed = [c for c in checks if not c.ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"FAIL: {len(failed)}/{len(checks)} required checks failed")
        return 1
    print(f"PASS: all {len(checks)} required checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
