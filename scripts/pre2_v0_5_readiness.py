#!/usr/bin/env python3
"""Check whether the pre-2 v0.5 POC launch inputs are ready."""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from myllm_pre2.config import DataMixConfig, load_data_mix_config
from myllm_pre2.data.source_registry import SourceRegistry, load_source_registry


TOKEN_BYTES = 4
METADATA_FACTOR = 1.30


def _format_tokens(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    return f"{n:,}"


def _bytes_to_gib(n: int | float) -> float:
    return float(n) / (1024**3)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recommended_local_storage_gb(target_tokens: int) -> int:
    if target_tokens <= 10_000_000_000:
        return 500
    if target_tokens <= 30_000_000_000:
        return 1_500
    return int(math.ceil((target_tokens * TOKEN_BYTES * METADATA_FACTOR * 8) / 1_000_000_000))


def _source_gate_payload(
    mix: DataMixConfig,
    registry: SourceRegistry,
    stage: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    sources_by_id = {source.source_id: source for source in registry.sources}
    stage_sources = {source.source_id for source in registry.sources_for_stage(stage)}  # type: ignore[arg-type]
    buckets: list[dict[str, Any]] = []
    issues: list[str] = []

    for bucket in mix.source_buckets:
        if not bucket.source_ids:
            issues.append(f"{bucket.bucket}: no source_ids declared")
        required_tokens = bucket.target_tokens
        if required_tokens is None and bucket.target_share is not None:
            required_tokens = int(round(mix.target_tokens * bucket.target_share))
        available_tokens = 0
        source_payloads = []
        for source_id in bucket.source_ids:
            source = sources_by_id.get(source_id)
            if source is None:
                issues.append(f"{bucket.bucket}: source_id {source_id} not found in registry")
                continue
            if source.license_status != "approved":
                issues.append(f"{bucket.bucket}: source_id {source_id} is not approved")
            if source_id not in stage_sources:
                issues.append(f"{bucket.bucket}: source_id {source_id} is not allowed for {stage}")
            available_tokens += source.estimated_tokens or 0
            source_payloads.append(
                {
                    "source_id": source.source_id,
                    "license_status": source.license_status,
                    "revision": source.revision,
                    "estimated_tokens": source.estimated_tokens,
                }
            )
        if required_tokens is not None and available_tokens < required_tokens:
            issues.append(
                f"{bucket.bucket}: estimated source tokens {_format_tokens(available_tokens)} "
                f"< required {_format_tokens(required_tokens)}"
            )
        buckets.append(
            {
                "bucket": bucket.bucket,
                "target_share": bucket.target_share,
                "target_tokens": required_tokens,
                "available_tokens": available_tokens,
                "source_ids": bucket.source_ids,
                "sources": source_payloads,
            }
        )
    return buckets, issues


def build_payload(
    *,
    mix_path: Path,
    registry_path: Path,
    build_mix_path: Path | None,
    tokenizer_path: Path,
    stage: str,
    allow_missing_tokenizer: bool,
    allow_missing_decontam: bool,
) -> tuple[dict[str, Any], int]:
    mix = load_data_mix_config(mix_path)
    registry = load_source_registry(registry_path)
    buckets, issues = _source_gate_payload(mix, registry, stage)
    repo = registry_path.resolve().parents[2]

    tokenizer: dict[str, Any] = {"path": str(tokenizer_path), "exists": tokenizer_path.exists()}
    if tokenizer_path.exists():
        tokenizer["size_bytes"] = tokenizer_path.stat().st_size
        tokenizer["sha256"] = _sha256_file(tokenizer_path)
    elif not allow_missing_tokenizer:
        issues.append(f"tokenizer artifact missing: {tokenizer_path}")
    else:
        tokenizer["waived"] = True

    decontamination: dict[str, Any] = {"enabled": False, "indexes": []}
    if build_mix_path is not None:
        with open(build_mix_path, encoding="utf-8") as f:
            build_mix = yaml.safe_load(f)
        if not isinstance(build_mix, dict):
            raise ValueError(f"{build_mix_path} did not parse to a mapping")
        decon_cfg = build_mix.get("decontamination", {}) or {}
        decontamination["enabled"] = bool(decon_cfg.get("enabled", False))
        if decontamination["enabled"]:
            raw_paths = [
                decon_cfg.get("index_path_primary"),
                decon_cfg.get("index_path_secondary"),
            ]
            if not any(raw_paths):
                raw_paths = [decon_cfg.get("index_path")]
            for raw_path in raw_paths:
                if not raw_path:
                    continue
                path = Path(str(raw_path))
                if not path.is_absolute():
                    path = repo / path
                exists = path.exists()
                record = {"path": str(path), "exists": exists}
                decontamination["indexes"].append(record)
                if not exists and not allow_missing_decontam:
                    issues.append(f"decontamination index missing: {path}")
                elif not exists:
                    record["waived"] = True

    packed_bytes = mix.target_tokens * TOKEN_BYTES
    packed_with_metadata = int(math.ceil(packed_bytes * METADATA_FACTOR))
    payload = {
        "name": mix.name,
        "stage": stage,
        "target_tokens": mix.target_tokens,
        "synthetic_cap": mix.synthetic_cap,
        "buckets": buckets,
        "tokenizer": tokenizer,
        "storage": {
            "packed_token_bytes": packed_bytes,
            "packed_token_gib": round(_bytes_to_gib(packed_bytes), 2),
            "packed_with_metadata_gib": round(_bytes_to_gib(packed_with_metadata), 2),
            "recommended_local_nvme_gb": _recommended_local_storage_gb(mix.target_tokens),
            "recommended_object_storage_gb": 1_000 if mix.target_tokens <= 10_000_000_000 else 2_000,
        },
        "decontamination": decontamination,
        "blocking_issues": issues,
        "ready": not issues,
    }
    return payload, 0 if payload["ready"] else 2


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mix",
        default=str(repo / "configs" / "data" / "pre2_mix_v0_5.yaml"),
        help="v0.5 data mix config.",
    )
    parser.add_argument(
        "--registry",
        default=str(repo / "configs" / "data" / "pre2_source_registry.yaml"),
        help="pre-2 source registry.",
    )
    parser.add_argument(
        "--build-mix",
        default=str(repo / "configs" / "data" / "pre2_v0_5_build_mix.yaml"),
        help="Build-time mix config used to locate decontamination artifacts.",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(repo / "artifacts" / "tokenizer_v1.json"),
        help="tokenizer artifact expected by the packed-corpus build.",
    )
    parser.add_argument("--stage", default="poc", choices=["poc", "proxy"])
    parser.add_argument(
        "--allow-missing-tokenizer",
        action="store_true",
        help="Report readiness without failing on missing tokenizer. Use only for dry runs.",
    )
    parser.add_argument(
        "--allow-missing-decontam",
        action="store_true",
        help="Report readiness without failing on missing decontamination indexes. Use only for dry runs.",
    )
    parser.add_argument("--yaml", action="store_true", help="Emit machine-readable YAML.")
    args = parser.parse_args()

    payload, exit_code = build_payload(
        mix_path=Path(args.mix),
        registry_path=Path(args.registry),
        build_mix_path=Path(args.build_mix) if args.build_mix else None,
        tokenizer_path=Path(args.tokenizer),
        stage=args.stage,
        allow_missing_tokenizer=args.allow_missing_tokenizer,
        allow_missing_decontam=args.allow_missing_decontam,
    )
    if args.yaml:
        print(yaml.safe_dump(payload, sort_keys=False).strip())
    else:
        status = "ready" if payload["ready"] else "blocked"
        print(
            f"{payload['name']}: {status} "
            f"target_tokens={_format_tokens(payload['target_tokens'])} "
            f"stage={payload['stage']}"
        )
        for bucket in payload["buckets"]:
            print(
                f"{bucket['bucket']}: share={bucket['target_share']:.0%} "
                f"target={_format_tokens(bucket['target_tokens'])} "
                f"sources={','.join(bucket['source_ids'])}"
            )
        storage = payload["storage"]
        print(
            "storage: "
            f"packed={storage['packed_token_gib']:.2f}GiB "
            f"packed_plus_metadata={storage['packed_with_metadata_gib']:.2f}GiB "
            f"local_nvme_min={storage['recommended_local_nvme_gb']}GB "
            f"object_storage={storage['recommended_object_storage_gb']}GB"
        )
        tokenizer = payload["tokenizer"]
        if tokenizer["exists"]:
            print(f"tokenizer: present size={tokenizer['size_bytes']} sha256={tokenizer['sha256']}")
        else:
            print(f"tokenizer: missing path={tokenizer['path']}")
        if payload["decontamination"]["enabled"]:
            for index in payload["decontamination"]["indexes"]:
                state = "present" if index["exists"] else "missing"
                print(f"decontamination_index: {state} path={index['path']}")
        for issue in payload["blocking_issues"]:
            print(f"blocking: {issue}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
