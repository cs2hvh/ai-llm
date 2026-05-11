"""S3-compatible object storage helpers (Cloudflare R2 by default).

Reads endpoint / bucket / credentials from environment:
    S3_ENDPOINT_URL, S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.

Functions are retry-wrapped on transient connection errors. Heavy operations
(checkpoint upload, dataset cache push) should use the streaming variants
to avoid materialising large blobs in memory.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from myllm.utils import get_logger
from myllm.utils.exceptions import MyLLMError

log = get_logger(__name__)


class StorageError(MyLLMError):
    """Raised by storage operations on configuration or remote errors."""


_RETRY = retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    reraise=True,
)


def _client() -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise ImportError("boto3 not installed; pip install boto3") from e

    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if not endpoint:
        raise StorageError("S3_ENDPOINT_URL not set")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3},
            connect_timeout=10,
            read_timeout=60,
        ),
    )


def _default_bucket() -> str:
    b = os.environ.get("S3_BUCKET")
    if not b:
        raise StorageError("S3_BUCKET not set")
    return b


@_RETRY
def upload_file(local_path: str | Path, key: str, bucket: str | None = None) -> str:
    bucket = bucket or _default_bucket()
    local = str(local_path)
    if not Path(local).exists():
        raise StorageError(f"local file not found: {local}")
    log.info("storage_upload", local=local, bucket=bucket, key=key)
    _client().upload_file(local, bucket, key)
    return f"s3://{bucket}/{key}"


@_RETRY
def download_file(key: str, local_path: str | Path, bucket: str | None = None) -> str:
    bucket = bucket or _default_bucket()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    log.info("storage_download", bucket=bucket, key=key, local=str(local_path))
    _client().download_file(bucket, key, str(local_path))
    return str(local_path)


@_RETRY
def upload_bytes(data: bytes, key: str, bucket: str | None = None) -> str:
    bucket = bucket or _default_bucket()
    log.info("storage_upload_bytes", bucket=bucket, key=key, size=len(data))
    _client().put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


@_RETRY
def download_bytes(key: str, bucket: str | None = None) -> bytes:
    bucket = bucket or _default_bucket()
    log.info("storage_download_bytes", bucket=bucket, key=key)
    return _client().get_object(Bucket=bucket, Key=key)["Body"].read()


def exists(key: str, bucket: str | None = None) -> bool:
    bucket = bucket or _default_bucket()
    try:
        _client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def list_keys(prefix: str = "", bucket: str | None = None) -> Iterator[str]:
    bucket = bucket or _default_bucket()
    paginator = _client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def upload_directory(
    local_dir: str | Path,
    key_prefix: str,
    bucket: str | None = None,
) -> int:
    """Upload every file under ``local_dir`` to ``key_prefix/<relpath>``.

    Returns the number of files uploaded. Used for sharded checkpoint dirs.
    """
    bucket = bucket or _default_bucket()
    root = Path(local_dir)
    if not root.is_dir():
        raise StorageError(f"not a directory: {root}")
    count = 0
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            upload_file(path, f"{key_prefix.rstrip('/')}/{rel}", bucket)
            count += 1
    log.info(
        "storage_upload_directory",
        local=str(root),
        bucket=bucket,
        prefix=key_prefix,
        files=count,
    )
    return count


def download_directory(
    key_prefix: str,
    local_dir: str | Path,
    bucket: str | None = None,
) -> int:
    bucket = bucket or _default_bucket()
    root = Path(local_dir)
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    for key in list_keys(prefix=key_prefix, bucket=bucket):
        rel = key[len(key_prefix) :].lstrip("/")
        if not rel:
            continue
        local_path = root / rel
        download_file(key, local_path, bucket)
        count += 1
    log.info(
        "storage_download_directory",
        bucket=bucket,
        prefix=key_prefix,
        local=str(root),
        files=count,
    )
    return count
