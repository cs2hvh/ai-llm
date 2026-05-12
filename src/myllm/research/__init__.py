"""Multi-agent research library for the MyLLM project.

Pattern adapted from Anthropic's blog post on multi-agent research systems:
https://www.anthropic.com/engineering/multi-agent-research-system

The library is intentionally narrow: three pre-baked workflows that match
this project's recurring task shapes, not a general-purpose orchestrator.

  - ``verify_candidates``: parallel multi-candidate verification
    (e.g. "compare 5 teacher models against these license + modality
    criteria"). Each candidate gets its own subagent + a structured
    return shape; the orchestrator synthesizes a comparison table.

  - ``multi_source_lookup``: parallel fact-finding across web sources
    (e.g. "what does each of these 8 papers say about X?"). Subagents
    fetch + summarize; orchestrator synthesizes with citations.

  - ``parallel_audit``: parallel file-level review (e.g. "audit these
    5 files for problem X"). Subagents read files via a local
    ``file_read`` tool; orchestrator produces a punch list.

When NOT to use this library:
  - Sequential reasoning tasks (code writing, debugging) — single chat
    is faster and cheaper.
  - One-off questions where Claude Code's built-in Agent tool already
    suffices in-session.

Cost characteristic: per the source blog post, multi-agent workflows
use ~15× more tokens than a single chat turn. Workflows here typically
land at $0.50-$2.00 per run with Opus 4.7 throughout. The disk cache
makes re-runs free for unchanged inputs.

See ``docs/multi_agent_research.md`` for the design rationale + usage.
"""
from myllm.research.client import (
    ResearchClient,
    SubagentResult,
    UsageStats,
)
from myllm.research.workflows.multi_source_lookup import multi_source_lookup
from myllm.research.workflows.parallel_audit import parallel_audit
from myllm.research.workflows.verify_candidates import verify_candidates

__all__ = [
    "ResearchClient",
    "SubagentResult",
    "UsageStats",
    "verify_candidates",
    "multi_source_lookup",
    "parallel_audit",
]
