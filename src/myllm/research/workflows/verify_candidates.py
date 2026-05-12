"""verify_candidates workflow: parallel multi-candidate verification.

Use case: you have N candidates (teacher models, datasets, GPU SKUs,
HF dataset versions, ...) and a fixed set of criteria they need to meet.
Each subagent verifies one candidate independently and returns a
structured report. The orchestrator synthesizes a comparison table +
recommendation.

Example:

    result = verify_candidates(
        candidates=[
            {"id": "olmo-3-32b",     "hf_id": "allenai/Olmo-3-1125-32B"},
            {"id": "qwen3-14b",      "hf_id": "Qwen/Qwen3-14B-Base"},
            {"id": "mixtral-8x22b",  "hf_id": "mistralai/Mixtral-8x22B-v0.1"},
        ],
        criteria=[
            "License must be Apache-2.0, MIT, or equivalently permissive",
            "Must be a base text-only model (not instruct-tuned, not multimodal)",
            "Must publish weight + training details on HuggingFace",
            "Should be ≥ 20B params dense or ≥ 50B MoE",
        ],
        context="Selecting a second distillation teacher for MyLLM v1...",
    )

Each subagent is told the candidate + the criteria + has access to
web_fetch so it can read HF model cards and license files. The
orchestrator gets all subagent reports and produces a final synthesis.
"""
from __future__ import annotations

from typing import Any

from myllm.research.client import ResearchClient, SubagentResult
from myllm.research.tools import WEB_FETCH_TOOL, make_web_fetch_handler
from myllm.research.workflows.base import (
    WorkflowResult,
    aggregate_usage,
    run_subagents_parallel,
)
from myllm.utils import get_logger

log = get_logger(__name__)


_ORCHESTRATOR_SYSTEM = """\
You are a senior engineer running a verification audit for the MyLLM project.

You will receive a list of candidates, each verified independently by a
subagent. Your job is to synthesize their findings into:
  1. A comparison table (one row per candidate, columns = criteria, cells
     = PASS / FAIL / UNCLEAR + a 1-line note).
  2. A recommendation: which candidates pass, which to reject, which need
     further investigation (and what specifically).
  3. Any factual disagreement or uncertainty across the subagent reports.

Bias toward conservatism: a candidate is PASS only if a subagent
verified it directly (not "probably permissive"). Flag UNCLEAR liberally.

Be concise. The reader is a solo engineer who needs to act on this; do
not pad with restatements of the brief.
"""

_SUBAGENT_SYSTEM = """\
You are verifying ONE candidate for the MyLLM project. The orchestrator
will combine your report with other subagents' reports.

For each criterion in the brief:
  1. State the criterion verbatim.
  2. Report PASS / FAIL / UNCLEAR.
  3. Cite the source(s) (URL + relevant quote) that support your verdict.
  4. Note any caveats or open questions.

You have access to a ``web_fetch`` tool — USE IT to verify claims
against the candidate's HuggingFace model card, license file, paper, or
official documentation. Do NOT rely on training-data knowledge alone:
licenses and modality details change.

Output format (markdown):

    ## Candidate: <id>

    ### Criterion 1: <verbatim criterion>
    **Verdict:** PASS|FAIL|UNCLEAR
    **Evidence:** <URL + short quote>
    **Notes:** <caveats>

    ### Criterion 2: ...

    ## Summary
    <2-3 sentence verdict + any cross-cutting concerns>

If a fetch fails or evidence is missing, mark UNCLEAR — never guess.
"""


def _subagent_fn(client: ResearchClient, payload: dict) -> SubagentResult:
    candidate = payload["candidate"]
    criteria = payload["criteria"]
    context = payload.get("context", "")

    user_msg = (
        f"# Verification brief\n\n"
        f"{context}\n\n"
        f"## Candidate to verify\n\n"
        f"```json\n{candidate!s}\n```\n\n"
        f"## Criteria (verify each)\n\n"
        + "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
        + "\n\nUse web_fetch to verify against the candidate's HuggingFace "
        "model card and any linked license files. Return the structured "
        "report described in your system prompt."
    )
    return client.call(
        system=_SUBAGENT_SYSTEM,
        user=user_msg,
        tools=[WEB_FETCH_TOOL],
        tool_handlers={"web_fetch": make_web_fetch_handler()},
        max_tokens=4096,
    )


def verify_candidates(
    candidates: list[dict[str, Any]],
    criteria: list[str],
    *,
    context: str = "",
    client: ResearchClient | None = None,
    max_workers: int = 5,
) -> WorkflowResult:
    """Verify N candidates against M criteria. See module docstring."""
    if not candidates:
        raise ValueError("candidates list is empty")
    if not criteria:
        raise ValueError("criteria list is empty")
    # Each candidate must carry an "id" so subagent results can be keyed.
    for c in candidates:
        if "id" not in c:
            raise ValueError(f"candidate missing 'id': {c}")

    client = client or ResearchClient()

    log.info(
        "verify_candidates_start",
        n_candidates=len(candidates),
        n_criteria=len(criteria),
    )

    # Parallel subagent verification.
    subagent_inputs = [
        (
            c["id"],
            {"candidate": c, "criteria": criteria, "context": context},
        )
        for c in candidates
    ]
    subagent_results = run_subagents_parallel(
        client, subagent_inputs, _subagent_fn, max_workers=max_workers,
    )

    # Build the synthesis prompt: orchestrator sees every subagent's report.
    report_sections = []
    for cid, r in subagent_results.items():
        if r.error:
            report_sections.append(
                f"### Subagent report for `{cid}` (FAILED)\n\n"
                f"Error: {r.error}\n"
            )
        else:
            report_sections.append(
                f"### Subagent report for `{cid}`\n\n{r.content}\n"
            )

    orchestrator_user = (
        f"# Verification synthesis\n\n"
        f"## Original brief\n\n{context}\n\n"
        f"## Criteria\n\n"
        + "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
        + "\n\n## Subagent reports\n\n"
        + "\n---\n\n".join(report_sections)
        + "\n\nProduce the comparison table + recommendation per your "
        "system prompt."
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
