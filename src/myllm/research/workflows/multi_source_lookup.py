"""multi_source_lookup workflow: parallel fact-finding across web sources.

Use case: you have a question and a list of sources (papers, blog posts,
HF model cards, governance docs) that might answer it. Each subagent
reads ONE source and reports what it says about the question. The
orchestrator synthesizes a final answer with per-source citations.

This is the "verify-before-locking" pattern (per the user's standing
guidance) industrialized.

Example:

    result = multi_source_lookup(
        question="What is the standard practice for n-gram size in "
                 "benchmark decontamination? What hash function is used?",
        sources=[
            "https://arxiv.org/abs/2402.16819",          # OLMo 2 tech report
            "https://huggingface.co/blog/smollm3",       # SmolLM3 post
            "https://github.com/EleutherAI/lm-eval-harness/...",
        ],
    )

The orchestrator's synthesis will include "Source [1] says ... Source [2]
says ..." inline citations so downstream readers can verify.
"""
from __future__ import annotations

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
You are synthesizing findings from multiple sources to answer a single
question for the MyLLM project.

Your output should:
  1. State the answer concisely (1-3 paragraphs).
  2. Cite sources INLINE using [1], [2], etc., matching the order in
     which they appear in the brief. Every factual claim must carry a
     citation.
  3. End with a "Sources" list mapping [N] → URL + 1-line description.
  4. Call out disagreements: if sources say different things, name the
     disagreement explicitly.
  5. Mark UNCERTAINTY: if no source directly answers the question, say
     so — do not synthesize a confident answer from training-data
     knowledge.

Be concise. Output markdown.
"""

_SUBAGENT_SYSTEM = """\
You are reading ONE source to answer ONE question for the MyLLM project.

Use the ``web_fetch`` tool to retrieve the source. Then:
  1. Quote the most relevant passage(s) verbatim (with location if
     possible — section heading, paragraph number).
  2. State what this source says about the question, in your own words.
  3. Flag whether this source ACTUALLY addresses the question or is
     only tangentially related.
  4. Note any caveats or limitations the source itself raises.

Output format (markdown):

    ## Source
    <URL>

    ## Relevant quotes
    > <verbatim quote 1>
    > <verbatim quote 2>

    ## What this source says
    <2-4 sentences in your own words>

    ## Direct answer to the question?
    YES | PARTIAL | NO + 1-line justification

If the fetch fails, report the error explicitly and produce NO content
("Direct answer to the question?: NO — fetch failed").
"""


def _subagent_fn(client: ResearchClient, payload: dict) -> SubagentResult:
    question = payload["question"]
    url = payload["url"]
    user_msg = (
        f"# Question\n\n{question}\n\n"
        f"# Source to read\n\n{url}\n\n"
        "Use web_fetch to retrieve it, then produce the structured "
        "report described in your system prompt."
    )
    return client.call(
        system=_SUBAGENT_SYSTEM,
        user=user_msg,
        tools=[WEB_FETCH_TOOL],
        tool_handlers={"web_fetch": make_web_fetch_handler()},
        max_tokens=3072,
    )


def multi_source_lookup(
    question: str,
    sources: list[str],
    *,
    client: ResearchClient | None = None,
    max_workers: int = 5,
) -> WorkflowResult:
    """Answer ``question`` by reading ``sources`` in parallel. See module docstring."""
    if not question.strip():
        raise ValueError("question is empty")
    if not sources:
        raise ValueError("sources list is empty")

    client = client or ResearchClient()

    log.info("multi_source_lookup_start", n_sources=len(sources))

    subagent_inputs = [
        (f"source_{i+1}", {"question": question, "url": url})
        for i, url in enumerate(sources)
    ]
    subagent_results = run_subagents_parallel(
        client, subagent_inputs, _subagent_fn, max_workers=max_workers,
    )

    # Build the synthesis brief: orchestrator sees every subagent + the URLs
    # in numbered order so [N] citations resolve.
    numbered_reports = []
    for i, url in enumerate(sources):
        sid = f"source_{i+1}"
        r = subagent_results[sid]
        if r.error:
            numbered_reports.append(
                f"### Source [{i+1}] {url} (FAILED: {r.error})\n"
            )
        else:
            numbered_reports.append(
                f"### Source [{i+1}] {url}\n\n{r.content}\n"
            )

    orchestrator_user = (
        f"# Synthesis brief\n\n"
        f"## Question\n\n{question}\n\n"
        f"## Subagent reports (one per source)\n\n"
        + "\n---\n\n".join(numbered_reports)
        + "\n\nProduce the synthesized answer with inline [N] citations "
        "per your system prompt."
    )
    orchestrator_result = client.call(
        system=_ORCHESTRATOR_SYSTEM,
        user=orchestrator_user,
        max_tokens=3072,
    )

    return WorkflowResult(
        summary=orchestrator_result.content,
        subagent_outputs=subagent_results,
        usage=aggregate_usage(orchestrator_result, subagent_results),
    )
