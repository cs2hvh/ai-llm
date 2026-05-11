#!/usr/bin/env python3
"""Live credential verifier — Phase 0 deliverable.

Validates each credential via a no-cost API call (whoami / list / head).
Never prints credential values, only PASS/FAIL with the API response status.
Reads credentials from environment (load via ``set -a; source .env; set +a``).

Exit code: 0 if all configured credentials pass, 1 if any fail.
"""
from __future__ import annotations

import os
import sys
from typing import NamedTuple


class Result(NamedTuple):
    name: str
    ok: bool
    detail: str


def check_hf() -> Result:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return Result("HF_TOKEN", False, "not set")
    try:
        import requests

        r = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            return Result(
                "HF_TOKEN",
                True,
                f"user={data.get('name', '?')} type={data.get('type', '?')}",
            )
        return Result("HF_TOKEN", False, f"http {r.status_code}: {r.text[:80]}")
    except Exception as e:
        return Result("HF_TOKEN", False, f"{type(e).__name__}: {e}")


def check_runpod() -> Result:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        return Result("RUNPOD_API_KEY", False, "not set")
    try:
        import requests

        # GraphQL: get current user. The myself query is cheap and read-only.
        r = requests.post(
            "https://api.runpod.io/graphql",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={"query": "{ myself { id email } }"},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            if "errors" in data:
                return Result("RUNPOD_API_KEY", False, f"api errors: {data['errors']}")
            me = data.get("data", {}).get("myself") or {}
            return Result(
                "RUNPOD_API_KEY",
                True,
                f"user_id={me.get('id', '?')[:8]}…",
            )
        return Result("RUNPOD_API_KEY", False, f"http {r.status_code}: {r.text[:80]}")
    except Exception as e:
        return Result("RUNPOD_API_KEY", False, f"{type(e).__name__}: {e}")


def check_wandb() -> Result:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        return Result("WANDB_API_KEY", False, "not set")
    try:
        import requests

        # W&B uses Basic auth (user='api', pw=key) and GraphQL via POST.
        r = requests.post(
            "https://api.wandb.ai/graphql",
            auth=("api", key),
            json={"query": "{ viewer { username } }"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            if "errors" in data:
                return Result("WANDB_API_KEY", False, f"api errors: {data['errors']}")
            user = (data.get("data") or {}).get("viewer") or {}
            return Result("WANDB_API_KEY", True, f"user={user.get('username', '?')}")
        return Result("WANDB_API_KEY", False, f"http {r.status_code}: {r.text[:80]}")
    except Exception as e:
        return Result("WANDB_API_KEY", False, f"{type(e).__name__}: {e}")


def check_r2() -> Result:
    key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    bucket = os.environ.get("S3_BUCKET")
    if not all([key_id, secret, endpoint, bucket]):
        missing = [
            n
            for n, v in [
                ("AWS_ACCESS_KEY_ID", key_id),
                ("AWS_SECRET_ACCESS_KEY", secret),
                ("S3_ENDPOINT_URL", endpoint),
                ("S3_BUCKET", bucket),
            ]
            if not v
        ]
        return Result("R2 (S3)", False, f"missing: {missing}")
    try:
        import boto3
        from botocore.config import Config

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 2},
                connect_timeout=10,
                read_timeout=10,
            ),
        )
        # head_bucket is the canonical "can we authenticate + bucket exists" probe.
        s3.head_bucket(Bucket=bucket)

        # Round-trip: write + read a tiny test object. Confirms write perms.
        key = ".myllm-credcheck"
        s3.put_object(Bucket=bucket, Key=key, Body=b"ok")
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        s3.delete_object(Bucket=bucket, Key=key)
        if body != b"ok":
            return Result("R2 (S3)", False, "round-trip body mismatch")
        return Result("R2 (S3)", True, f"bucket={bucket} read+write ok")
    except Exception as e:
        return Result("R2 (S3)", False, f"{type(e).__name__}: {e}")


def main() -> int:
    print("MyLLM credential check")
    print("=" * 70)
    checks = [check_hf(), check_runpod(), check_wandb(), check_r2()]
    failed = []
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        print(f"  [{mark}] {c.name:<20} {c.detail}")
        if not c.ok:
            failed.append(c)
    print("=" * 70)
    if failed:
        print(f"FAIL: {len(failed)}/{len(checks)} credentials failed")
        return 1
    print(f"PASS: all {len(checks)} credentials verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
