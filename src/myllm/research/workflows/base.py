"""Shared helpers for workflows."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, TypeVar

from myllm.research.client import ResearchClient, SubagentResult, UsageStats
from myllm.utils import get_logger

log = get_logger(__name__)

T = TypeVar("T")
SubagentInput = TypeVar("SubagentInput")


@dataclass
class WorkflowResult:
    """Top-level container for any workflow's output."""

    summary: str  # the orchestrator's synthesized answer
    subagent_outputs: dict[str, SubagentResult]  # keyed by subagent id
    usage: UsageStats  # aggregated across orchestrator + subagents


def run_subagents_parallel(
    client: ResearchClient,
    inputs: list[tuple[str, SubagentInput]],
    subagent_fn: Callable[[ResearchClient, SubagentInput], SubagentResult],
    *,
    max_workers: int = 5,
) -> dict[str, SubagentResult]:
    """Run ``subagent_fn`` on each (id, input) pair in parallel.

    ``max_workers`` caps the concurrency — Anthropic enforces per-key
    rate limits, so blasting 20 concurrent subagents will start hitting
    429s. 5 is a safe default for Tier 1; bump to 10-20 on higher tiers.

    Failures are captured in the SubagentResult.error field — they
    don't kill sibling subagents (per the Anthropic blog: "subagent
    failures shouldn't kill the run").
    """
    results: dict[str, SubagentResult] = {}
    log.info(
        "research_subagents_start",
        n=len(inputs),
        max_workers=max_workers,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_id = {
            ex.submit(subagent_fn, client, payload): subagent_id
            for subagent_id, payload in inputs
        }
        for future in as_completed(future_to_id):
            subagent_id = future_to_id[future]
            try:
                results[subagent_id] = future.result()
            except Exception as e:  # noqa: BLE001
                log.error(
                    "research_subagent_unhandled_error",
                    subagent_id=subagent_id,
                    error=str(e),
                )
                results[subagent_id] = SubagentResult(
                    content="", usage=UsageStats(), error=str(e),
                )
    log.info("research_subagents_done", n=len(results))
    return results


def aggregate_usage(
    primary: SubagentResult, others: dict[str, SubagentResult]
) -> UsageStats:
    """Sum the orchestrator's usage with every subagent's."""
    total = UsageStats()
    total.add(primary.usage)
    for r in others.values():
        total.add(r.usage)
    return total
