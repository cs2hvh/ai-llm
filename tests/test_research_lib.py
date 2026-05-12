"""Tests for the multi-agent research library.

No real API calls — we monkeypatch the anthropic SDK so tests run fast,
deterministically, and without burning real tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from myllm.research.client import (
    ResearchClient,
    SubagentResult,
    UsageStats,
    _block_to_dict,
)
from myllm.research.tools import (
    WebFetchConfig,
    make_file_read_handler,
    make_web_fetch_handler,
)
from myllm.research.workflows.base import (
    aggregate_usage,
    run_subagents_parallel,
)


# --------------------------------------------------------------------------- #
# Mock anthropic SDK
# --------------------------------------------------------------------------- #
@dataclass
class _MockTextBlock:
    type: str = "text"
    text: str = ""

    def model_dump(self, exclude_none: bool = False) -> dict:
        return {"type": self.type, "text": self.text}


@dataclass
class _MockToolUseBlock:
    type: str = "tool_use"
    id: str = "toolu_1"
    name: str = "web_fetch"
    input: dict | None = None

    def model_dump(self, exclude_none: bool = False) -> dict:
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "input": self.input or {},
        }


@dataclass
class _MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class _MockResponse:
    content: list
    usage: _MockUsage
    stop_reason: str = "end_turn"


def _patch_anthropic_client(client: ResearchClient, response_sequence: list):
    """Replace client._client.messages.create with a function that returns
    successive entries from response_sequence on each call."""
    iterator = iter(response_sequence)

    def _create(**kwargs):
        try:
            return next(iterator)
        except StopIteration:
            pytest.fail("model called more times than test expected")

    client._client = MagicMock()
    client._client.messages.create = _create


@pytest.fixture
def client(tmp_path, monkeypatch):
    """ResearchClient with API key + cache wired up to tmp_path."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    c = ResearchClient(cache_dir=tmp_path / "cache")
    return c


# --------------------------------------------------------------------------- #
# UsageStats
# --------------------------------------------------------------------------- #
class TestUsageStats:
    def test_usd_cost_uses_per_million_pricing(self):
        u = UsageStats(input_tokens=1_000_000, output_tokens=1_000_000)
        # $15/M in + $75/M out per Opus 4.7 pricing.
        assert u.usd_cost == pytest.approx(90.0)

    def test_add_aggregates(self):
        a = UsageStats(input_tokens=100, output_tokens=50, n_calls=1)
        b = UsageStats(input_tokens=200, output_tokens=80, n_calls=1)
        a.add(b)
        assert a.input_tokens == 300
        assert a.output_tokens == 130
        assert a.n_calls == 2

    def test_summary_mentions_cache_hits_only_when_nonzero(self):
        u = UsageStats(n_calls=3, n_cache_hits=0)
        assert "cached" not in u.summary()
        u2 = UsageStats(n_calls=3, n_cache_hits=2)
        assert "2 cached" in u2.summary()


# --------------------------------------------------------------------------- #
# ResearchClient — single-turn call
# --------------------------------------------------------------------------- #
class TestResearchClientCall:
    def test_simple_text_response(self, client):
        _patch_anthropic_client(client, [
            _MockResponse(
                content=[_MockTextBlock(text="hello world")],
                usage=_MockUsage(input_tokens=50, output_tokens=10),
            ),
        ])
        result = client.call(system="sys", user="hi")
        assert result.content == "hello world"
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 10
        assert result.usage.n_calls == 1
        assert result.error is None

    def test_api_error_returns_error_field_not_raises(self, client):
        def _boom(**kwargs):
            raise RuntimeError("simulated API down")
        client._client.messages.create = _boom
        result = client.call(system="sys", user="hi")
        assert result.error is not None
        assert "simulated API down" in result.error

    def test_cache_hit_skips_second_api_call(self, client):
        _patch_anthropic_client(client, [
            _MockResponse(
                content=[_MockTextBlock(text="cached answer")],
                usage=_MockUsage(input_tokens=50, output_tokens=10),
            ),
        ])
        first = client.call(system="sys", user="question")
        # Second call with identical inputs must NOT hit the API again
        # (the response_sequence only has one entry — a second call
        # would StopIteration and fail the test).
        second = client.call(system="sys", user="question")
        assert first.content == second.content == "cached answer"
        assert second.usage.n_cache_hits == 1
        assert second.usage.n_calls == 0  # cache hit, no API call

    def test_cache_miss_on_different_user_message(self, client):
        _patch_anthropic_client(client, [
            _MockResponse(
                content=[_MockTextBlock(text="answer A")],
                usage=_MockUsage(),
            ),
            _MockResponse(
                content=[_MockTextBlock(text="answer B")],
                usage=_MockUsage(),
            ),
        ])
        r1 = client.call(system="sys", user="Q1")
        r2 = client.call(system="sys", user="Q2")
        assert r1.content == "answer A"
        assert r2.content == "answer B"
        # Both real calls — no cache hits.
        assert r1.usage.n_cache_hits == 0
        assert r2.usage.n_cache_hits == 0


# --------------------------------------------------------------------------- #
# ResearchClient — tool-use loop
# --------------------------------------------------------------------------- #
class TestToolUseLoop:
    def test_tool_use_triggers_handler_and_continues(self, client):
        _patch_anthropic_client(client, [
            # Turn 1: model asks for a tool.
            _MockResponse(
                content=[_MockToolUseBlock(input={"url": "https://example.com"})],
                usage=_MockUsage(input_tokens=80, output_tokens=20),
                stop_reason="tool_use",
            ),
            # Turn 2: model sees tool result + emits final text.
            _MockResponse(
                content=[_MockTextBlock(text="final after tool")],
                usage=_MockUsage(input_tokens=120, output_tokens=30),
                stop_reason="end_turn",
            ),
        ])
        captured = {}

        def _handler(inp: dict) -> str:
            captured["got"] = inp
            return "TOOL RESULT BODY"

        result = client.call(
            system="sys", user="please fetch",
            tools=[{"name": "web_fetch", "description": "x",
                    "input_schema": {"type": "object", "properties": {}, "required": []}}],
            tool_handlers={"web_fetch": _handler},
        )
        assert result.content == "final after tool"
        assert captured["got"] == {"url": "https://example.com"}
        # Two model calls (initial + after-tool).
        assert result.usage.n_calls == 2
        assert result.usage.n_tool_iterations == 1

    def test_missing_tool_handler_produces_error_tool_result(self, client):
        """If the model calls a tool we don't have a handler for, we
        must not crash — return an is_error result and let the model
        recover."""
        _patch_anthropic_client(client, [
            _MockResponse(
                content=[_MockToolUseBlock(name="unknown_tool", input={})],
                usage=_MockUsage(),
                stop_reason="tool_use",
            ),
            _MockResponse(
                content=[_MockTextBlock(text="gave up after error")],
                usage=_MockUsage(),
                stop_reason="end_turn",
            ),
        ])
        result = client.call(
            system="sys", user="...",
            tools=[{"name": "unknown_tool", "description": "x",
                    "input_schema": {"type": "object", "properties": {}, "required": []}}],
            tool_handlers={},  # no handler
        )
        # Did not raise; model recovered.
        assert result.content == "gave up after error"
        assert result.error is None

    def test_tool_loop_capped_at_max_iterations(self, client):
        """Infinite tool-use loop must be capped — otherwise a buggy
        tool could burn the token budget."""
        # 4 turns: tool_use, tool_use, tool_use, tool_use. With max_tool_iterations=2
        # we expect only 3 model calls (initial + 2 tool rounds).
        responses = [
            _MockResponse(
                content=[_MockToolUseBlock(input={"url": "https://x"})],
                usage=_MockUsage(),
                stop_reason="tool_use",
            ),
            _MockResponse(
                content=[_MockToolUseBlock(input={"url": "https://x"})],
                usage=_MockUsage(),
                stop_reason="tool_use",
            ),
            _MockResponse(
                content=[_MockToolUseBlock(input={"url": "https://x"})],
                usage=_MockUsage(),
                stop_reason="tool_use",
            ),
        ]
        _patch_anthropic_client(client, responses)
        result = client.call(
            system="sys", user="...",
            tools=[{"name": "web_fetch", "description": "x",
                    "input_schema": {"type": "object", "properties": {}, "required": []}}],
            tool_handlers={"web_fetch": lambda inp: "ok"},
            max_tool_iterations=2,
        )
        # Loop exited cleanly (no exception), didn't make a 4th call.
        assert result.usage.n_calls == 3


# --------------------------------------------------------------------------- #
# file_read tool — path-traversal safety
# --------------------------------------------------------------------------- #
class TestFileReadHandler:
    def test_reads_file_within_root(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "x.py").write_text("hello = 1\n")
        handler = make_file_read_handler(tmp_path)
        assert "hello = 1" in handler({"path": "src/x.py"})

    def test_rejects_absolute_paths(self, tmp_path):
        handler = make_file_read_handler(tmp_path)
        out = handler({"path": "/etc/passwd"})
        assert out.startswith("ERROR:")
        assert "absolute" in out.lower()

    def test_rejects_path_traversal(self, tmp_path):
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("SECRET")
        handler = make_file_read_handler(tmp_path)
        out = handler({"path": "../outside.txt"})
        assert out.startswith("ERROR:")
        assert "escape" in out.lower()
        # And the secret was NOT returned.
        assert "SECRET" not in out

    def test_missing_file_returns_error(self, tmp_path):
        handler = make_file_read_handler(tmp_path)
        assert "not found" in handler({"path": "nope.py"}).lower()

    def test_oversized_file_returns_error_not_truncated(self, tmp_path):
        big = tmp_path / "big.txt"
        big.write_text("x" * 300_000)
        handler = make_file_read_handler(tmp_path)
        out = handler({"path": "big.txt"})
        assert out.startswith("ERROR:")
        assert "too large" in out


# --------------------------------------------------------------------------- #
# web_fetch tool — basic shape (no real HTTP)
# --------------------------------------------------------------------------- #
class TestWebFetchHandler:
    def test_rejects_non_http_urls(self):
        handler = make_web_fetch_handler()
        for bad in ["", "file:///etc/passwd", "ftp://x", None, 42]:
            out = handler({"url": bad})
            assert out.startswith("ERROR:"), f"should reject {bad!r}: got {out!r}"

    def test_fetch_returns_decoded_body(self, monkeypatch):
        """We monkeypatch httpx.Client to return a canned response."""
        import httpx

        class _FakeResponse:
            status_code = 200
            headers = {"content-type": "text/plain; charset=utf-8"}
            content = b"hello there"

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url): return _FakeResponse()

        monkeypatch.setattr(httpx, "Client", _FakeClient)
        handler = make_web_fetch_handler()
        out = handler({"url": "https://example.com"})
        assert "hello there" in out
        assert "HTTP 200" in out

    def test_fetch_rejects_binary_content_type(self, monkeypatch):
        import httpx

        class _FakeResponse:
            status_code = 200
            headers = {"content-type": "application/octet-stream"}
            content = b"\x00\x01\x02"

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url): return _FakeResponse()

        monkeypatch.setattr(httpx, "Client", _FakeClient)
        handler = make_web_fetch_handler()
        out = handler({"url": "https://example.com/x.bin"})
        assert out.startswith("ERROR:")
        assert "binary" in out.lower()

    def test_fetch_truncates_at_max_bytes(self, monkeypatch):
        import httpx

        class _FakeResponse:
            status_code = 200
            headers = {"content-type": "text/plain"}
            content = b"x" * 1_000_000

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url): return _FakeResponse()

        monkeypatch.setattr(httpx, "Client", _FakeClient)
        handler = make_web_fetch_handler(WebFetchConfig(max_bytes=1024))
        out = handler({"url": "https://example.com"})
        assert "truncated at 1024" in out


# --------------------------------------------------------------------------- #
# Workflow base helpers
# --------------------------------------------------------------------------- #
class TestRunSubagentsParallel:
    def test_runs_all_subagents_and_returns_keyed_results(self):
        def _fn(c, payload):
            return SubagentResult(content=f"answered {payload['n']}", usage=UsageStats())
        results = run_subagents_parallel(
            client=None,  # not used by _fn
            inputs=[("a", {"n": 1}), ("b", {"n": 2}), ("c", {"n": 3})],
            subagent_fn=_fn,
            max_workers=2,
        )
        assert set(results) == {"a", "b", "c"}
        assert results["b"].content == "answered 2"

    def test_subagent_failure_does_not_kill_siblings(self):
        def _fn(c, payload):
            if payload["fail"]:
                raise RuntimeError("subagent crashed")
            return SubagentResult(content="ok", usage=UsageStats())
        results = run_subagents_parallel(
            client=None,
            inputs=[
                ("good", {"fail": False}),
                ("bad", {"fail": True}),
                ("also_good", {"fail": False}),
            ],
            subagent_fn=_fn,
            max_workers=3,
        )
        assert results["good"].content == "ok"
        assert results["also_good"].content == "ok"
        # The failing subagent must surface in `error`, not crash.
        assert results["bad"].error is not None
        assert "crashed" in results["bad"].error


class TestAggregateUsage:
    def test_sums_orchestrator_plus_subagents(self):
        orch = SubagentResult(content="x", usage=UsageStats(input_tokens=100, output_tokens=50))
        subs = {
            "a": SubagentResult(content="", usage=UsageStats(input_tokens=200, output_tokens=80)),
            "b": SubagentResult(content="", usage=UsageStats(input_tokens=150, output_tokens=70)),
        }
        total = aggregate_usage(orch, subs)
        assert total.input_tokens == 450
        assert total.output_tokens == 200


# --------------------------------------------------------------------------- #
# Workflows — end-to-end with a mock client (no real API calls)
# --------------------------------------------------------------------------- #
class TestVerifyCandidatesWorkflow:
    def test_dispatches_one_subagent_per_candidate(self, monkeypatch, tmp_path):
        from myllm.research.workflows import verify_candidates as vc_mod

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        calls: list[dict] = []

        def _fake_call(self, **kwargs):
            calls.append(kwargs)
            # Return a stub result distinguishable by the user prompt.
            return SubagentResult(
                content=f"subagent saw: {kwargs['user'][:50]}",
                usage=UsageStats(input_tokens=80, output_tokens=20, n_calls=1),
            )

        monkeypatch.setattr(ResearchClient, "call", _fake_call)
        result = vc_mod.verify_candidates(
            candidates=[
                {"id": "a", "hf_id": "x/a"},
                {"id": "b", "hf_id": "x/b"},
                {"id": "c", "hf_id": "x/c"},
            ],
            criteria=["criterion 1", "criterion 2"],
            context="test context",
            client=ResearchClient(cache_dir=tmp_path),
            max_workers=3,
        )
        # 3 subagents + 1 orchestrator synthesis = 4 calls
        assert len(calls) == 4
        # Orchestrator's user prompt should include each candidate id.
        orch_user = calls[-1]["user"]
        for cid in ("a", "b", "c"):
            assert cid in orch_user
        assert "subagent saw" in orch_user  # subagent output forwarded
        assert len(result.subagent_outputs) == 3

    def test_rejects_empty_inputs(self, monkeypatch, tmp_path):
        from myllm.research.workflows import verify_candidates as vc_mod

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        c = ResearchClient(cache_dir=tmp_path)
        with pytest.raises(ValueError, match="candidates"):
            vc_mod.verify_candidates(candidates=[], criteria=["x"], client=c)
        with pytest.raises(ValueError, match="criteria"):
            vc_mod.verify_candidates(
                candidates=[{"id": "a"}], criteria=[], client=c,
            )

    def test_rejects_candidate_without_id(self, monkeypatch, tmp_path):
        from myllm.research.workflows import verify_candidates as vc_mod

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        c = ResearchClient(cache_dir=tmp_path)
        with pytest.raises(ValueError, match="missing 'id'"):
            vc_mod.verify_candidates(
                candidates=[{"hf_id": "x/y"}],
                criteria=["x"],
                client=c,
            )


class TestMultiSourceLookupWorkflow:
    def test_passes_question_and_one_source_per_subagent(self, monkeypatch, tmp_path):
        from myllm.research.workflows import multi_source_lookup as msl_mod

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        calls: list[dict] = []

        def _fake_call(self, **kwargs):
            calls.append(kwargs)
            return SubagentResult(
                content=f"saw question, prompt len={len(kwargs['user'])}",
                usage=UsageStats(n_calls=1),
            )

        monkeypatch.setattr(ResearchClient, "call", _fake_call)
        result = msl_mod.multi_source_lookup(
            question="What is X?",
            sources=[
                "https://example.com/a",
                "https://example.com/b",
            ],
            client=ResearchClient(cache_dir=tmp_path),
        )
        # 2 subagent + 1 orchestrator = 3 calls
        assert len(calls) == 3
        # Orchestrator sees numbered citations [1] and [2].
        orch_user = calls[-1]["user"]
        assert "[1]" in orch_user and "[2]" in orch_user
        assert "https://example.com/a" in orch_user
        assert "https://example.com/b" in orch_user


class TestParallelAuditWorkflow:
    def test_dispatches_one_subagent_per_path(self, monkeypatch, tmp_path):
        from myllm.research.workflows import parallel_audit as pa_mod

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        calls: list[dict] = []

        def _fake_call(self, **kwargs):
            calls.append(kwargs)
            return SubagentResult(content="findings: none", usage=UsageStats(n_calls=1))

        monkeypatch.setattr(ResearchClient, "call", _fake_call)
        result = pa_mod.parallel_audit(
            paths=["src/myllm/a.py", "src/myllm/b.py"],
            audit_question="any race conditions?",
            repo_root=tmp_path,
            client=ResearchClient(cache_dir=tmp_path / "cache"),
        )
        # 2 subagent + 1 orchestrator = 3 calls
        assert len(calls) == 3
        # Audit question reaches every subagent + the orchestrator.
        for c in calls:
            assert "race conditions" in c["user"]
        assert set(result.subagent_outputs) == {"src/myllm/a.py", "src/myllm/b.py"}


# --------------------------------------------------------------------------- #
# Block coercion (cache serialization)
# --------------------------------------------------------------------------- #
class TestBlockToDict:
    def test_pydantic_block_with_model_dump(self):
        block = _MockTextBlock(text="hi")
        out = _block_to_dict(block)
        assert out == {"type": "text", "text": "hi"}

    def test_fallback_for_attr_only_block(self):
        class _NoDump:
            type = "text"
            text = "fallback"
            id = None
            name = None
            input = None

        out = _block_to_dict(_NoDump())
        assert out["type"] == "text"
        assert out["text"] == "fallback"
