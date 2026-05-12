# Multi-agent research library

A small, narrow-scope implementation of the orchestrator-with-parallel-
subagents pattern from
[Anthropic's blog post](https://www.anthropic.com/engineering/multi-agent-research-system).

Three pre-baked workflows that match this project's recurring task shapes
— **not** a general-purpose research agent.

## When to use

- **`verify_candidates`** — N candidate models / datasets / SKUs against M criteria. Each subagent verifies one candidate with `web_fetch` access; the orchestrator synthesizes a comparison table + recommendation.
- **`multi_source_lookup`** — One question, N web sources. Subagents fetch + summarize in parallel; the orchestrator synthesizes an answer with inline `[N]` citations.
- **`parallel_audit`** — N local files, one audit question. Subagents read files via `file_read` (sandboxed to the repo root); the orchestrator returns a consolidated punch list ranked by severity.

## When NOT to use

- Sequential reasoning (writing code, debugging) — a single Claude Code session is faster and cheaper.
- One-off questions where Claude Code's built-in `Agent` tool already suffices.
- Tasks where the subagents would each need state from the previous one (the pattern is for parallel-breadth, not serial-depth).

## Cost characteristic

Per the source blog post: multi-agent workflows use ~15× more tokens than a single chat turn. Typical workflows here land at:

| Workflow | Typical cost (Opus 4.7) | When |
|---|---|---|
| `verify_candidates` (5 candidates × 6 criteria) | $1.50–3.00 | once per major HP/teacher decision |
| `multi_source_lookup` (8 sources) | $1.00–2.00 | "verify-before-locking" workflow industrialized |
| `parallel_audit` (5 files) | $1.00–2.50 | one-off forensic sweeps (e.g., "find resume bugs across the training stack") |

**Disk cache makes re-runs free** for unchanged inputs (cache key = SHA256 of full request payload).

## Architecture

```
scripts/research_cli.py            — CLI entrypoint
src/myllm/research/
    client.py                      — Anthropic SDK wrapper + cost + cache + tool loop
    tools.py                       — web_fetch + file_read (sandboxed) tool handlers
    workflows/
        base.py                    — parallel subagent runner + usage aggregator
        verify_candidates.py       — multi-candidate verification workflow
        multi_source_lookup.py     — parallel fact-finding workflow
        parallel_audit.py          — parallel file-level audit workflow
tests/test_research_lib.py         — 29 tests, no real API calls (mocked SDK)
briefs/example_teacher_verify.yaml — example brief for the verify workflow
```

### One model throughout: Opus 4.7

Per user preference (2026-05-12), both orchestrator and subagents run on
`claude-opus-4-7`. Anthropic's blog recommends a split (orchestrator on
Opus, subagents on Haiku) to save cost ~3-5×; we chose quality-first
because our workflows are typically load-bearing (license verification,
data-corruption audits) where a missed nuance in a Haiku subagent would
defeat the purpose.

If cost becomes an issue, the split is a one-line change in
`ResearchClient.__init__` + a per-workflow override.

### Failure handling

Three layers of defense, matching the source blog post's "subagent
failures shouldn't kill the run" principle:

1. **API errors** are caught in `ResearchClient.call` → returned as a
   `SubagentResult.error` field instead of propagating.
2. **Subagent crashes** (exceptions in `subagent_fn`) are caught in
   `run_subagents_parallel` → wrapped in an error-flagged `SubagentResult`.
3. **Tool-use loops** are capped at `max_tool_iterations=8` so a
   misbehaving tool can't burn the token budget.

Tests pin all three behaviors.

### Cache safety

Cache key = SHA256 of `(model, system, user_messages, tools, max_tokens, temperature)`. Any prompt-template change invalidates the cache for that subagent — a deliberately strict policy because false-positive cache hits silently return stale answers, and that's the silent-corruption bug class this library is meant to *catch*, not introduce. (See `feedback_verify_before_locking.md` in memory.)

## Usage

### Verify candidates

```bash
python scripts/research_cli.py verify \
    --brief briefs/example_teacher_verify.yaml \
    --output-dir artifacts/research/teacher_v2_audit
```

Brief format: YAML with `context`, `criteria` (list of strings), and
`candidates` (list of dicts with required `id` field; other fields
get passed verbatim into the subagent prompt).

### Multi-source lookup

```bash
python scripts/research_cli.py lookup \
    --question "What n-gram size do OLMo 2, SmolLM3, and Llama 2 use for benchmark decontamination? What hash function?" \
    --source https://arxiv.org/abs/2402.16819 \
    --source https://huggingface.co/blog/smollm3 \
    --source https://ai.meta.com/blog/llama-2/ \
    --output-dir artifacts/research/decon_ngram_lookup
```

### Parallel audit

```bash
python scripts/research_cli.py audit \
    --audit "Find every place where resume-from-checkpoint could silently corrupt state — cursors that aren't saved, namedtuples that get flattened, off-by-one between data position and teacher cache position." \
    --path src/myllm/training/loop.py \
    --path src/myllm/training/checkpoint.py \
    --path src/myllm/data/teacher_cache.py \
    --path src/myllm/training/decay_phase.py \
    --output-dir artifacts/research/resume_audit
```

### Output

The CLI prints the synthesized answer to stdout. With `--output-dir`, it
also persists:

```
<output-dir>/
    summary.md          — orchestrator's final synthesis
    usage.json          — token + USD cost breakdown
    subagents/
        <id>.md         — per-subagent raw output
```

### Programmatic use

```python
from myllm.research import verify_candidates

result = verify_candidates(
    candidates=[{"id": "x", "hf_id": "org/x"}, ...],
    criteria=["License must be Apache-2.0 ...", ...],
    context="Selecting a teacher for ...",
)

print(result.summary)             # orchestrator output
print(result.usage.summary())     # "4 model calls, 12,345 input + 2,310 output tokens, $0.3527"
for cid, r in result.subagent_outputs.items():
    print(cid, r.content)
```

## Prerequisites

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # required
pip install anthropic httpx           # already in requirements.txt
```

## What this library deliberately doesn't do

- **No general-purpose orchestrator.** If your task doesn't fit one of
  the three workflows, write a new workflow file (it's ~100-150 LOC
  including prompt templates) rather than hacking a generic one.
- **No memory across runs.** Each run is independent. (Anthropic's blog
  describes a memory mechanism for long horizons; ours are short.)
- **No citation agent.** Sources are usually a small, named set we
  already have in hand; we don't need a separate citation-verification
  step.
- **No model routing.** One model (`claude-opus-4-7`) for everything.
  See "One model throughout" above for the rationale.

## See also

- Anthropic blog post: <https://www.anthropic.com/engineering/multi-agent-research-system>
- `feedback_verify_before_locking.md` (auto-memory) — the "verify
  external claims via WebFetch before locking" rule this library
  industrializes.
