"""Persistent processed-shard manifest.

Each completed shard writes a JSON manifest atomically. Re-running the
pipeline reads existing manifests and skips shards that are already done,
making the pipeline resumable across pod failures.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from myllm.data.types import ProcessedShardManifest
from myllm.utils.io import read_json, write_json_atomic


class ManifestStore:
    """Filesystem-backed store of processed-shard manifests.

    Layout: ``<root>/<dataset>/<shard_id>.json``
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, dataset: str, shard_id: str) -> Path:
        # Datasets contain '/' (HF org/name); flatten to keep one dir per dataset.
        safe = dataset.replace("/", "__")
        return self.root / safe / f"{shard_id}.json"

    def has(self, dataset: str, shard_id: str) -> bool:
        return self._path_for(dataset, shard_id).exists()

    def write(self, m: ProcessedShardManifest) -> None:
        path = self._path_for(m.dataset, m.shard_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, asdict(m))

    def read(self, dataset: str, shard_id: str) -> ProcessedShardManifest:
        data = read_json(self._path_for(dataset, shard_id))
        return ProcessedShardManifest(**data)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
