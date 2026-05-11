"""Atomic IO and hashing helpers.

All write helpers write to a sibling temp file then ``os.replace`` so
concurrent readers never see a half-written file.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, data: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        _silent_unlink(tmp)
        raise


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        _silent_unlink(tmp)
        raise


def write_json_atomic(path: str | Path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=True))


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _silent_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
