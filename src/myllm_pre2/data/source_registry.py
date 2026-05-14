"""Pre-2 data source registry and license gate."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


LicenseStatus = Literal["approved", "needs_review", "blocked", "excluded"]
TrainingStage = Literal["canary", "poc", "proxy", "mainline"]
SourceType = Literal[
    "web",
    "code",
    "math",
    "books_reference",
    "qa_documentation",
    "multilingual",
    "synthetic",
    "fixture",
]


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=3)
    display_name: str = Field(min_length=1)
    provider: str | None = None
    source_type: SourceType
    buckets: list[str] = Field(min_length=1)
    license_status: LicenseStatus
    license_expression: str | None = None
    terms_url: str | None = None
    revision: str | None = None
    estimated_tokens: int | None = Field(default=None, gt=0)
    allowed_stages: list[TrainingStage] = Field(default_factory=list)
    required_controls: list[str] = Field(default_factory=list)
    synthetic: bool = False
    synthetic_origin: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check_license_gate(self) -> "SourceEntry":
        if self.license_status == "approved":
            if not self.license_expression:
                raise ValueError(f"{self.source_id}: approved sources require license_expression")
            if not self.revision:
                raise ValueError(f"{self.source_id}: approved sources require a pinned revision")
        elif self.allowed_stages:
            raise ValueError(
                f"{self.source_id}: only approved sources may declare allowed_stages"
            )

        if self.license_status in {"blocked", "excluded"} and not self.notes:
            raise ValueError(f"{self.source_id}: blocked/excluded sources require notes")
        if self.synthetic and not self.synthetic_origin:
            raise ValueError(f"{self.source_id}: synthetic sources require synthetic_origin")
        if not self.synthetic and self.synthetic_origin:
            raise ValueError(f"{self.source_id}: synthetic_origin requires synthetic=true")
        return self


class SourceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: float | str
    status: Literal["planning", "active"]
    name: str
    sources: list[SourceEntry] = Field(min_length=1)
    required_controls: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_sources(self) -> "SourceRegistry":
        ids = [source.source_id for source in self.sources]
        duplicates = [source_id for source_id, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate source_id values: {', '.join(sorted(duplicates))}")
        return self

    def sources_for_stage(self, stage: TrainingStage) -> list[SourceEntry]:
        return [
            source
            for source in self.sources
            if source.license_status == "approved" and stage in source.allowed_stages
        ]

    def estimated_tokens_for_stage(self, stage: TrainingStage) -> int:
        return sum(source.estimated_tokens or 0 for source in self.sources_for_stage(stage))

    def status_counts(self) -> dict[str, int]:
        counts = Counter(source.license_status for source in self.sources)
        return {status: counts.get(status, 0) for status in ("approved", "needs_review", "blocked", "excluded")}

    def bucket_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for source in self.sources:
            for bucket in source.buckets:
                counts[bucket] += 1
        return dict(sorted(counts.items()))


def load_source_registry(path: str | Path) -> SourceRegistry:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return SourceRegistry.model_validate(data)
