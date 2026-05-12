"""Tools subagents can use during a workflow.

Two tools, kept tight on scope:

  - ``web_fetch(url)``: HTTP GET with timeout + size cap. Returns the
    response body (text only — binary content is rejected). Used by
    ``multi_source_lookup`` for parallel source reading.

  - ``file_read(path)``: read a repo-local file. Restricted to a
    configured root (defaults to the repo root). Used by
    ``parallel_audit`` for per-file review.

Both tools log every call so the operator can audit what the subagents
actually did.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from myllm.utils import get_logger

log = get_logger(__name__)

# Anthropic tool-definition dicts (passed to messages.create(tools=...)).

WEB_FETCH_TOOL: dict = {
    "name": "web_fetch",
    "description": (
        "Fetch the textual content of a URL via HTTP GET. Returns the "
        "response body decoded as UTF-8. Use this to read pages, "
        "documentation, model cards, or papers cited in your task. "
        "Times out at 20 seconds and caps response size at 500KB."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http:// or https:// URL to fetch.",
            },
        },
        "required": ["url"],
    },
}

FILE_READ_TOOL: dict = {
    "name": "file_read",
    "description": (
        "Read a file from the repository. Path must be RELATIVE to the "
        "repository root and must not escape it (no '..' segments). "
        "Returns the file's full contents as UTF-8. Use for source-code "
        "audits, config reviews, or docs inspection."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path RELATIVE to the repository root "
                    "(e.g. 'src/myllm/training/loop.py')."
                ),
            },
        },
        "required": ["path"],
    },
}


# --------------------------------------------------------------------------- #
# Handlers (called by ResearchClient.call when the model emits a tool_use)
# --------------------------------------------------------------------------- #
@dataclass
class WebFetchConfig:
    timeout_sec: float = 20.0
    max_bytes: int = 500_000
    user_agent: str = "MyLLM-Research/0.1 (+contact: harshit.hv@samatva.com)"


def make_web_fetch_handler(config: WebFetchConfig | None = None):
    """Build a web_fetch handler bound to the given config.

    Returned callable accepts the tool input dict and returns a string.
    Errors are returned as ``ERROR: ...`` strings, never raised — the
    model handles its own retry/fallback logic.
    """
    cfg = config or WebFetchConfig()

    def _handler(tool_input: dict) -> str:
        try:
            import httpx
        except ImportError:
            return "ERROR: httpx not installed (pip install httpx)"

        url = (tool_input or {}).get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return f"ERROR: invalid url {url!r}; must be absolute http/https"
        log.info("web_fetch_start", url=url)
        try:
            with httpx.Client(
                timeout=cfg.timeout_sec,
                follow_redirects=True,
                headers={"User-Agent": cfg.user_agent},
            ) as client:
                resp = client.get(url)
        except httpx.HTTPError as e:
            log.warning("web_fetch_failed", url=url, error=str(e))
            return f"ERROR: HTTP request failed: {e}"

        ct = resp.headers.get("content-type", "")
        if "html" not in ct and "text" not in ct and "json" not in ct and "xml" not in ct:
            log.warning("web_fetch_binary_rejected", url=url, content_type=ct)
            return f"ERROR: refusing to read binary content-type {ct!r}"

        body = resp.content[: cfg.max_bytes]
        try:
            text = body.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return f"ERROR: could not decode body as UTF-8 (content-type {ct!r})"

        suffix = ""
        if len(resp.content) > cfg.max_bytes:
            suffix = (
                f"\n\n[truncated at {cfg.max_bytes} bytes; "
                f"full size {len(resp.content)} bytes]"
            )
        log.info(
            "web_fetch_done",
            url=url,
            status=resp.status_code,
            bytes=len(body),
        )
        return f"HTTP {resp.status_code} {ct}\n\n{text}{suffix}"

    return _handler


def make_file_read_handler(repo_root: str | Path):
    """Build a file_read handler restricted to ``repo_root``.

    Path traversal attempts (``..``, absolute paths, symlinks pointing
    outside the root) are rejected. The handler is intentionally
    read-only — there is no file_write counterpart in this library.
    """
    root = Path(repo_root).resolve()

    def _handler(tool_input: dict) -> str:
        rel = (tool_input or {}).get("path")
        if not isinstance(rel, str) or not rel.strip():
            return f"ERROR: invalid path {rel!r}"
        if rel.startswith("/"):
            return f"ERROR: absolute paths not allowed; pass a path relative to {root}"
        # Resolve against the root, then check containment.
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return f"ERROR: path {rel!r} escapes repo root {root}"
        if not target.exists():
            return f"ERROR: file not found: {rel}"
        if not target.is_file():
            return f"ERROR: not a regular file: {rel}"
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"ERROR: read failed: {e}"
        log.info("file_read_done", path=rel, bytes=len(text))
        # Cap at 200KB so a single file_read can't crowd out other
        # context. Large files are unusual in this repo; if a real one
        # comes up we'll surface a useful error rather than silently
        # truncating mid-function.
        if len(text) > 200_000:
            return (
                f"ERROR: file too large ({len(text)} bytes); read a "
                "specific section by line range instead, or pre-extract "
                "the relevant region."
            )
        return text

    return _handler
