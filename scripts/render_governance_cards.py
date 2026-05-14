#!/usr/bin/env python3
"""Render governance AUTO blocks from live configs when present.

Older governance cards used ``*_template.md`` files with AUTO blocks. The
current pre-2 governance cards are live drafts, but this script is kept so
AUTO blocks can be reintroduced without rewriting the renderer. If template
files are absent, the script falls back to the current rendered card files.

Auto-rendered blocks:
  - model_card  → ``architecture`` (from configs/base_1b.yaml)
  - data_card   → ``sources``      (from configs/data/pretrain_mix.yaml)

Static prose (intended use, eval gates, environmental impact, etc.) is
NOT touched. Editors should add new auto-blocks here when more fields
become config-derived (e.g. teacher table, eval gate set).

Usage:
    python scripts/render_governance_cards.py                  # in-place render
    python scripts/render_governance_cards.py --check          # CI: fail if stale
    python scripts/render_governance_cards.py \\
        --model-config configs/pilot_250m.yaml \\
        --output-suffix _pilot                                 # render pilot variant

The ``--check`` mode is intended for CI: it renders the cards in memory
and exits non-zero if the on-disk file would change. This catches the
"someone updated configs but forgot to regenerate the card" failure mode.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_GOVERNANCE = _REPO / "docs" / "governance"

_AUTO_RE = re.compile(
    r'<!-- AUTO:start name="(?P<name>[a-z0-9_-]+)" -->.*?<!-- AUTO:end -->',
    re.DOTALL,
)


# --------------------------------------------------------------------------- #
# Render blocks
# --------------------------------------------------------------------------- #
def render_architecture_block(model_cfg: dict, base_cfg: dict | None) -> str:
    """Build the architecture bullets from a model config.

    ``base_cfg`` is the base_1b.yaml — used for v1-specific token budgets.
    When rendering a pilot/wind-tunnel card, pass base_cfg=None and the
    block falls back to the model's own ``target_tokens`` if present.
    """
    layers = model_cfg["layers"]
    hidden = model_cfg["hidden_dim"]
    ffn = model_cfg["ffn_dim"]
    ffn_ratio = ffn // hidden if hidden else 0
    n_heads = model_cfg["num_heads"]
    n_kv = model_cfg.get("num_kv_heads", n_heads)
    gqa = f"GQA {n_heads}:{n_kv}" if n_kv != n_heads else f"MHA {n_heads}"
    rope_base = model_cfg.get("rope_base", "n/a")
    context = model_cfg["context_length"]
    tied = model_cfg.get("tie_embeddings", False)

    norm = model_cfg.get("norm", "rmsnorm").upper().replace("RMSNORM", "RMSNorm")
    activation = model_cfg.get("activation", "swiglu").lower().replace("swiglu", "SwiGLU")
    qk = model_cfg.get("qk_norm", False)
    precision = model_cfg.get("mixed_precision", "bfloat16")

    arch_line = (
        f"{layers} layers × {hidden} hidden × {ffn_ratio}× FFN × {gqa} × "
        f"{'tied embeddings × ' if tied else ''}"
        f"{norm} + {activation} + RoPE (base {int(rope_base):,})"
        f"{' + QK-norm' if qk else ''}"
    )

    # Token budget: prefer base_cfg.target_tokens_v1; else model's own target.
    if base_cfg and "target_tokens_v1" in base_cfg:
        tokens = base_cfg["target_tokens_v1"]
        tokens_str = f"{tokens // 1_000_000_000_000}T (target for v1)"
    elif "target_tokens" in model_cfg:
        t = model_cfg["target_tokens"]
        tokens_str = _format_token_count(t)
    else:
        tokens_str = "TBD"

    raw_vocab = model_cfg.get("vocab_size")
    if raw_vocab is None:
        tok_vocab = "TBD"
    elif raw_vocab >= 1000:
        # Round to nearest 1k so 131072 reads as 131k (not 128k from
        # integer floor-division).
        tok_vocab = f"{round(raw_vocab / 1000)}k"
    else:
        tok_vocab = str(raw_vocab)
    # 2026-05-12: prior code used 'context_extension_yarn_target' which is a
    # typo not present in ModelConfig's schema (the dataclass field is
    # 'context_extension_target'). Aligned to match.
    extension = model_cfg.get("context_extension_target")
    context_note = (
        f"{context // 1024}k natively"
        + (f" — long-context anneal to {extension // 1024}k via YaRN" if extension else "")
    )

    bullets = [
        f"- **Architecture**: {arch_line}",
        f"- **Training tokens**: {tokens_str}",
        f"- **Tokenizer**: SentencePiece Unigram, {tok_vocab} vocab, "
        "NFKC + Metaspace, byte_fallback",
        f"- **Context length**: {context_note}",
        f"- **Mixed precision**: {precision}",
    ]
    return "\n".join(bullets)


def _format_token_count(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.1f}T"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    return str(n)


def render_sources_block(data_cfg: dict) -> str:
    """Build the per-source table from the pretrain_mix yaml."""
    sources = data_cfg.get("sources", [])
    if not sources:
        return "_(no sources configured)_"

    # License lookup keyed by dataset name. Authoritative log lives in
    # license_register.md — this map mirrors it for table rendering.
    license_map = {
        "HuggingFaceFW/fineweb-edu": "ODC-By 1.0",
        "bigcode/the-stack-v2": "BigCode Open RAIL-M (T&Cs accepted 2026-05-11)",
        "wikimedia/wikipedia": "CC-BY-SA 4.0",
        "pg19": "Public domain (Project Gutenberg pre-1919)",
        "allenai/peS2o": "ODC-By 1.0",
        "open-web-math/open-web-math": "ODC-By 1.0",
        "HuggingFaceH4/stack-exchange-preferences": "CC-BY-SA 4.0",
        "ai4bharat/sangraha": "CC-BY-4.0",
        "mc4": "ODC-By 1.0",
        "allenai/c4": "ODC-By 1.0",
    }

    notes_map = {
        "HuggingFaceFW/fineweb-edu": "High-quality educational subset; "
        "absorbs Nemotron-CC share pending NVIDIA approval",
        "bigcode/the-stack-v2": "Code",
        "wikimedia/wikipedia": "English Wikipedia snapshot",
        "pg19": "Books",
        "allenai/peS2o": "Academic",
        "open-web-math/open-web-math": "Math; absorbs proof-pile-2 share "
        "(dropped due to loader fragility)",
        "HuggingFaceH4/stack-exchange-preferences": "Q&A (question field only)",
        "ai4bharat/sangraha": "Hindi sovereign hedge",
        "mc4": "Secondary language",
    }

    lines = [
        "| # | HF dataset | Configured share | License | Revision pinned? | Notes |",
        "|---|---|---|---|---|---|",
    ]
    total = 0.0
    for i, src in enumerate(sources, 1):
        ds = src["dataset"]
        share = float(src["share"])
        total += share
        cfg_name = src.get("config_name")
        split = src.get("split")
        suffix_bits = []
        if cfg_name:
            suffix_bits.append(cfg_name)
        if split and split != "train":
            suffix_bits.append(f"split={split}")
        suffix = f" ({', '.join(suffix_bits)})" if suffix_bits else ""
        license_str = license_map.get(ds, "see license_register.md")
        notes = notes_map.get(ds, "")
        lines.append(
            f"| {i} | {ds}{suffix} | {share * 100:.1f}% | {license_str} | "
            f"⏳ B2 work | {notes} |"
        )
    lines.append("")
    lines.append(
        f"**Total**: {total * 100:.1f}% (validated at run-start; mixture "
        "sampler is token-weighted per P0-6 fix)"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
RENDERERS = {
    # name → (which configs it needs, render fn taking those configs)
    "architecture": ("architecture", "model+base", render_architecture_block),
    "sources": ("sources", "data", render_sources_block),
}


def render_template(template_text: str, *, configs: dict) -> str:
    """Substitute every ``<!-- AUTO:start name="X" -->...<!-- AUTO:end -->``.

    Unknown auto-block names raise to fail loudly rather than silently
    leaving stale content.
    """
    def _replace(match: re.Match) -> str:
        name = match.group("name")
        if name == "architecture":
            block = render_architecture_block(configs["model"], configs.get("base"))
        elif name == "sources":
            block = render_sources_block(configs["data"])
        else:
            raise ValueError(
                f"unknown AUTO block name: {name!r}. Add a renderer in "
                f"scripts/render_governance_cards.py."
            )
        return (
            f'<!-- AUTO:start name="{name}" -->\n{block}\n<!-- AUTO:end -->'
        )

    return _AUTO_RE.sub(_replace, template_text)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-config",
        default=str(_REPO / "configs" / "base_1b.yaml"),
        help="Model yaml. Default: configs/base_1b.yaml.",
    )
    p.add_argument(
        "--base-config",
        default=str(_REPO / "configs" / "base_1b.yaml"),
        help="Base 1B yaml (for token-budget fields). Same as --model-config "
        "when rendering the base card.",
    )
    p.add_argument(
        "--data-config",
        default=str(_REPO / "configs" / "data" / "pretrain_mix.yaml"),
    )
    p.add_argument(
        "--templates-dir",
        default=str(_GOVERNANCE),
        help="Directory containing governance templates. Falls back to "
        "model_card_v1.md and data_card_v1.md if templates are absent.",
    )
    p.add_argument(
        "--output-dir",
        default=str(_GOVERNANCE),
        help="Where to write rendered cards.",
    )
    p.add_argument(
        "--output-suffix",
        default="",
        help="Suffix on output filenames before the .md extension. "
        "E.g. '_pilot' → 'model_card_v1_pilot.md'. Default writes back "
        "to the template file in-place.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="CI mode: render in memory and exit 1 with a diff if any "
        "rendered AUTO block differs from on-disk content.",
    )
    args = p.parse_args()

    configs = {
        "model": _load_yaml(Path(args.model_config)),
        "base": _load_yaml(Path(args.base_config)),
        "data": _load_yaml(Path(args.data_config)),
    }

    templates_dir = Path(args.templates_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("model_card_v1_template.md", "model_card_v1.md", f"model_card_v1{args.output_suffix}.md"),
        ("data_card_v1_template.md", "data_card_v1.md", f"data_card_v1{args.output_suffix}.md"),
    ]

    diffs: list[str] = []
    for template_name, fallback_name, output_name in targets:
        template_path = templates_dir / template_name
        if not template_path.exists():
            template_path = templates_dir / fallback_name
        if not template_path.exists():
            print(f"missing template: {template_path}", file=sys.stderr)
            return 2
        template = template_path.read_text()
        rendered = render_template(template, configs=configs)

        out_path = output_dir / output_name
        if args.check:
            on_disk = out_path.read_text() if out_path.exists() else ""
            if on_disk != rendered:
                diffs.extend(
                    difflib.unified_diff(
                        on_disk.splitlines(keepends=True),
                        rendered.splitlines(keepends=True),
                        fromfile=str(out_path),
                        tofile=f"{out_path} (rendered)",
                    )
                )
        else:
            out_path.write_text(rendered)
            print(f"rendered {out_path}")

    if args.check and diffs:
        sys.stdout.writelines(diffs)
        print("\n[check] governance cards are stale; run without --check to fix.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
