"""parallel_audit workflow: parallel file-level code review.

Use case: you have N files in the repo and a single audit question
("does this file have race conditions?", "does it handle resume
correctly?", "are there stale comments?"). Each subagent reads ONE file
via the ``file_read`` tool and produces a punch list. The orchestrator
ranks issues by severity and produces a single consolidated punch list.

Example:

    result = parallel_audit(
        paths=[
            "src/myllm/training/loop.py",
            "src/myllm/training/checkpoint.py",
            "src/myllm/data/teacher_cache.py",
        ],
        audit_question=(
            "Find every place where resume-from-checkpoint could "
            "silently corrupt state — e.g. cursors that aren't saved, "
            "namedtuples that get flattened, off-by-one between data "
            "position and teacher cache position."
        ),
        repo_root="/root/llm-build",
    )
"""
from __future__ import annotations

from pathlib import Path

from myllm.research.client import ResearchClient, SubagentResult
from myllm.research.tools import FILE_READ_TOOL, make_file_read_handler
from myllm.research.workflows.base import (
    WorkflowResult,
    aggregate_usage,
    run_subagents_parallel,
)
from myllm.utils import get_logger

log = get_logger(__name__)


_ORCHESTRATOR_SYSTEM = """\
You are consolidating audit findings from multiple subagents who each
reviewed one file of the MyLLM project.

Produce a single punch list:
  1. Each item: SEVERITY (P0/P1/P2) + file path + line range + 1-line
     description.
  2. Group by severity, P0 first.
  3. After the punch list, write a 3-5 line OVERALL ASSESSMENT covering
     systemic patterns (vs file-specific bugs).
  4. If the audit question was substantially answered by NO file showing
     the problem, say so explicitly — "Audit found no instances of <X>
     across the N files reviewed."

Be concise. Output markdown.
"""

_SUBAGENT_SYSTEM = """\
You are auditing ONE file of the MyLLM project for a specific concern.

  1. Use the ``file_read`` tool to retrieve the file's contents.
  2. Read it carefully.
  3. Identify every occurrence of the audited concern, with line
     numbers (or line ranges).
  4. For each occurrence: severity (P0/P1/P2), 1-line description, and
     a suggested fix (if obvious).

P0 = silent-corruption or data-loss bug.
P1 = correctness bug with observable failure mode.
P2 = code-smell, minor staleness, missed-opportunity.

Output format (markdown):

    ## File: <path>

    ### Findings

    - **P0** at line <N> (or <N-M>): <description>. Fix: <suggestion>.
    - **P1** at line <N>: ...
    - ...

    ### Summary
    <2-3 sentence verdict: how clean is this file relative to the audit?>

If you find NO instances of the audited concern, say:

    ### Findings
    None. The file does not appear to be affected by this concern.

Do NOT speculate beyond what the file content shows. Do NOT review the
file for unrelated concerns.
"""


def _subagent_fn(client: ResearchClient, payload: dict) -> SubagentResult:
    path = payload["path"]
    audit_question = payload["audit_question"]
    repo_root = payload["repo_root"]
    user_msg = (
        f"# Audit brief\n\n"
        f"## File to audit\n\n`{path}`\n\n"
        f"## Audit question\n\n{audit_question}\n\n"
        f"Use file_read to retrieve the file content (the path is "
        f"already relative to the repo root). Then produce the "
        f"structured report per your system prompt."
    )
    return client.call(
        system=_SUBAGENT_SYSTEM,
        user=user_msg,
        tools=[FILE_READ_TOOL],
        tool_handlers={"file_read": make_file_read_handler(repo_root)},
        max_tokens=4096,
    )


def parallel_audit(
    paths: list[str],
    audit_question: str,
    *,
    repo_root: str | Path = ".",
    client: ResearchClient | None = None,
    max_workers: int = 5,
) -> WorkflowResult:
    """Audit N files in parallel against a single question. See module docstring."""
    if not paths:
        raise ValueError("paths list is empty")
    if not audit_question.strip():
        raise ValueError("audit_question is empty")

    client = client or ResearchClient()
    repo_root = Path(repo_root).resolve()

    log.info(
        "parallel_audit_start",
        n_paths=len(paths),
        repo_root=str(repo_root),
    )

    subagent_inputs = [
        (
            path,
            {
                "path": path,
                "audit_question": audit_question,
                "repo_root": str(repo_root),
            },
        )
        for path in paths
    ]
    subagent_results = run_subagents_parallel(
        client, subagent_inputs, _subagent_fn, max_workers=max_workers,
    )

    report_sections = []
    for path, r in subagent_results.items():
        if r.error:
            report_sections.append(
                f"### File `{path}` (FAILED)\n\nError: {r.error}\n"
            )
        else:
            report_sections.append(f"### File `{path}`\n\n{r.content}\n")

    orchestrator_user = (
        f"# Consolidation brief\n\n"
        f"## Audit question\n\n{audit_question}\n\n"
        f"## Subagent reports ({len(subagent_results)} files)\n\n"
        + "\n---\n\n".join(report_sections)
        + "\n\nProduce the consolidated punch list per your system prompt."
    )
    orchestrator_result = client.call(
        system=_ORCHESTRATOR_SYSTEM,
        user=orchestrator_user,
        max_tokens=4096,
    )

    return WorkflowResult(
        summary=orchestrator_result.content,
        subagent_outputs=subagent_results,
        usage=aggregate_usage(orchestrator_result, subagent_results),
    )
