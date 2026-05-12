"""Anthropic API client wrapper + cost tracking + disk cache.

Single responsibility: take a (system, user, tools) triple and return
the model's response while:
  - retrying on transient errors with exponential backoff
  - tracking input/output token usage + USD cost per call
  - persisting completed call results to a disk cache so re-runs are free
    when inputs are unchanged

Design choices:
  - One model for everything (Opus 4.7) — user preference. Simpler than
    a routing layer; quality-first.
  - Cache key = SHA256(model + system + user_messages + tools). The cache
    is conservative: any prompt-template change invalidates everything.
    That's the right tradeoff — false-positive cache hits would silently
    return stale data, which is the bug pattern this library is meant to
    avoid (see ``feedback_verify_before_locking.md`` in memory).
  - Tool use is handled here via the agentic-loop pattern: if the
    response stops with ``stop_reason=="tool_use"``, we run the tool,
    append the result, and recall the model. ``max_tool_iterations``
    caps the loop so a misbehaving tool can't burn the budget.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from myllm.utils import get_logger

log = get_logger(__name__)

# Opus 4.7 pricing per 1M tokens (USD). Update if Anthropic changes pricing.
# Source: Anthropic pricing page; cross-check before billing-load runs.
_PRICE_PER_MTOK_INPUT_USD = 15.0
_PRICE_PER_MTOK_OUTPUT_USD = 75.0

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 4096

# Hard cap so a runaway tool-use loop can't pile up cost. Per the source
# blog post: "we need to ensure that subagent failures don't kill the run".
DEFAULT_MAX_TOOL_ITERATIONS = 8


@dataclass
class UsageStats:
    """Aggregated token usage + USD cost across a workflow run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    n_calls: int = 0
    n_cache_hits: int = 0
    n_tool_iterations: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def usd_cost(self) -> float:
        return (
            self.input_tokens * _PRICE_PER_MTOK_INPUT_USD / 1_000_000.0
            + self.output_tokens * _PRICE_PER_MTOK_OUTPUT_USD / 1_000_000.0
        )

    def add(self, other: "UsageStats") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.n_calls += other.n_calls
        self.n_cache_hits += other.n_cache_hits
        self.n_tool_iterations += other.n_tool_iterations

    def summary(self) -> str:
        cache_line = (
            f" ({self.n_cache_hits} cached)" if self.n_cache_hits else ""
        )
        return (
            f"{self.n_calls} model calls{cache_line}, "
            f"{self.input_tokens:,} input + {self.output_tokens:,} output tokens, "
            f"${self.usd_cost:.4f}"
        )


@dataclass
class SubagentResult:
    """Container for a subagent's structured output.

    The ``content`` is the model's final text response (after any
    tool-use loop). ``raw_blocks`` is the full sequence of content
    blocks from the last assistant turn — useful when the caller wants
    structured tool-use traces, not just the text.
    """

    content: str
    usage: UsageStats
    raw_blocks: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    error: str | None = None  # set if the call ultimately failed


# Tool handler protocol: a callable that takes the tool input dict and
# returns a string result (which becomes the tool_result content sent
# back to the model).
ToolHandler = Callable[[dict], str]


class ResearchClient:
    """Wrapper around anthropic.Anthropic with retry + cache + tool loop.

    Usage:
        client = ResearchClient()
        result = client.call(
            system="You are a careful verifier.",
            user="Verify the following 3 claims...",
            tools=[WEB_FETCH_TOOL],
            tool_handlers={"web_fetch": handle_web_fetch},
        )
        print(result.content, result.usage.usd_cost)
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        # Lazy import so users who don't run the research lib don't need
        # anthropic installed.
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required for ResearchClient; "
                "install with `pip install anthropic`"
            ) from e

        self._anthropic_module = anthropic
        self.model = model
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it before constructing "
                "ResearchClient."
            )
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- #
    # Cache
    # ------------------------------------------------------------------- #
    def _cache_key(self, payload: dict) -> str:
        """Stable hash of the request payload for cache lookup."""
        # json.dumps with sort_keys + separators gives a canonical form.
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def _cache_put(self, key: str, value: dict) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(value))
        except OSError as e:
            log.warning("research_cache_write_failed", key=key, error=str(e))

    # ------------------------------------------------------------------- #
    # Main entry
    # ------------------------------------------------------------------- #
    def call(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict] | None = None,
        tool_handlers: dict[str, ToolHandler] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        temperature: float = 1.0,
    ) -> SubagentResult:
        """Single agent turn (with optional tool-use loop).

        Returns a SubagentResult containing the assembled final text +
        usage stats. On unrecoverable error, ``error`` is set and
        ``content`` is best-effort (may be empty).
        """
        tools = tools or []
        tool_handlers = tool_handlers or {}

        # Cache key based on the full request (system + user + tools).
        cache_payload = {
            "model": self.model,
            "system": system,
            "user": user,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        key = self._cache_key(cache_payload)
        cached = self._cache_get(key)
        if cached is not None:
            log.info("research_cache_hit", key=key[:16])
            usage = UsageStats(
                input_tokens=cached.get("usage", {}).get("input_tokens", 0),
                output_tokens=cached.get("usage", {}).get("output_tokens", 0),
                n_calls=0,
                n_cache_hits=1,
            )
            return SubagentResult(
                content=cached["content"],
                usage=usage,
                raw_blocks=cached.get("raw_blocks", []),
                stop_reason=cached.get("stop_reason"),
            )

        messages: list[dict] = [{"role": "user", "content": user}]
        usage = UsageStats()
        final_content = ""
        final_blocks: list[dict] = []
        stop_reason: str | None = None

        for iteration in range(max_tool_iterations + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=temperature,
                )
            except Exception as e:  # noqa: BLE001
                log.error(
                    "research_call_failed",
                    iteration=iteration,
                    error=str(e),
                )
                return SubagentResult(
                    content=final_content,
                    usage=usage,
                    raw_blocks=final_blocks,
                    stop_reason=stop_reason,
                    error=str(e),
                )

            usage.n_calls += 1
            usage.input_tokens += response.usage.input_tokens
            usage.output_tokens += response.usage.output_tokens
            stop_reason = response.stop_reason
            content_blocks = [_block_to_dict(b) for b in response.content]
            final_blocks = content_blocks

            # Collect text blocks for the running final content (so
            # text emitted before tool calls isn't lost).
            text_pieces = [b["text"] for b in content_blocks if b["type"] == "text"]
            if text_pieces:
                final_content = "\n".join(text_pieces)

            if stop_reason != "tool_use":
                break

            # Otherwise: dispatch each tool_use block, append results,
            # loop back to the model.
            usage.n_tool_iterations += 1
            tool_use_blocks = [b for b in content_blocks if b["type"] == "tool_use"]
            messages.append({"role": "assistant", "content": content_blocks})
            tool_results = []
            for tu in tool_use_blocks:
                handler = tool_handlers.get(tu["name"])
                if handler is None:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": f"ERROR: no handler for tool {tu['name']!r}",
                        "is_error": True,
                    })
                    continue
                try:
                    result_str = handler(tu.get("input", {}))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result_str,
                    })
                except Exception as e:  # noqa: BLE001
                    log.error(
                        "research_tool_handler_failed",
                        tool=tu["name"],
                        error=str(e),
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": f"ERROR: {e}",
                        "is_error": True,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            log.warning(
                "research_tool_loop_exhausted",
                max_tool_iterations=max_tool_iterations,
            )

        # Persist to cache only on success (error path skipped).
        result = SubagentResult(
            content=final_content,
            usage=usage,
            raw_blocks=final_blocks,
            stop_reason=stop_reason,
        )
        self._cache_put(
            key,
            {
                "content": result.content,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                },
                "stop_reason": stop_reason,
                "raw_blocks": final_blocks,
                "cached_at": time.time(),
            },
        )
        return result


def _block_to_dict(block: Any) -> dict:
    """Coerce a SDK content block to a plain dict (so it's cache-able)."""
    # The SDK returns typed objects; convert via model_dump if available,
    # else fall back to manual.
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    # Best-effort fallback.
    return {
        "type": getattr(block, "type", "unknown"),
        "text": getattr(block, "text", None),
        "id": getattr(block, "id", None),
        "name": getattr(block, "name", None),
        "input": getattr(block, "input", None),
    }
