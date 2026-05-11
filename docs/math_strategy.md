# Math Strategy for MyLLM

**Status:** v0.1, locked 2026-05-10. Revise after Phase 5.

Math is the single biggest quality differentiator between modern small LLMs
(SmolLM2, Qwen 2.5, Phi-3.5) and prior-generation small models. The recipe
is well-established: high-quality math data, phased curriculum (more math
late), and a reasoning-tuning post-training phase. This doc maps how MyLLM
handles math across the full lifecycle.

## 1. Data sources

### Pretraining (open licenses)

| Dataset | License | Description | Approx tokens |
|---|---|---|---|
| `open-web-math/open-web-math` | ODC-BY | Web math, deduplicated | ~14B |
| `EleutherAI/proof-pile-2` | mixed permissive | arXiv math, formal math, math web | ~55B |
| `MathPile` | CC-BY-NC-SA | Curated math web + arXiv | ~9B |
| `bigcode/the-stack-v2` (math-heavy subset) | permissive | Numerical-method code | varies |
| `Skywork/Skywork-OR1-RL-Data` (subset) | research-friendly | RL-friendly math problems | varies |

Total available math pretraining: ~80B tokens of high-quality, permissively-licensed material. Plenty for a 1B model.

### SFT (instruction-tuned math)

| Dataset | Use |
|---|---|
| `nvidia/OpenMathInstruct-2` | Large-scale math problem + step-by-step solution pairs |
| `meta-math/MetaMathQA` | Augmented GSM8K + MATH problems |
| `microsoft/orca-math-word-problems-200k` | Word problems with detailed solutions |
| `lighteval/MATH` (train split) | High-school competition math |
| `gsm8k` (train split) | Grade-school word problems |

### Reasoning tuning (Phase 8)

When the teacher API arrives, we synthesise:
- Step-by-step CoT for problems the base model gets wrong
- Multiple solution paths per problem (DPO pairs from quality differential)
- Verifiable problems with answer checking → RL/GRPO reward signal

## 2. Pretraining curriculum (mix shares)

| Phase | Math share | Rationale |
|---|---|---|
| Phase 4 (base pretrain, bulk) | **7%** | Modern baseline. v0 plan had 5% — bumped after research review. |
| Phase 5 (continued pretrain "decay") | **14%** | Mirrors SmolLM2's empirically-validated bump. Math share doubles in the last 5–10% of training. Cheap, measurable GSM8K gain. |

The data pipeline's `MixtureSampler` already supports re-weighting per phase. The schedule lands as YAML configs in Phase 5.

## 3. Math-specific quality filter (Phase 3 deliverable)

`MathQualityFilter` (to be implemented in `src/myllm/data/filters.py`):

- **LaTeX validity check** — drop documents where ≥30% of `$...$` or `\\(...\\)` blocks are malformed
- **Math symbol density** — for documents tagged as "math," reject if math-symbol-fraction (Greek letters, math operators, digits, common LaTeX commands) is < 5%
- **Equation density** — at least 1 equation per 1000 chars for "math-pure" documents
- **Garbled-text detection** — reject documents where word-boundary entropy is below threshold (often a sign of OCR'd math from PDF that lost spacing)

Roughly 100 LOC. Slots into the existing `Filter` ABC.

## 4. Tokenizer considerations

Byte-level BPE handles arbitrary UTF-8 cleanly, so math symbols (∑, ∫, π, ∂, ≤, ≥, ∈, ⊂, etc.) are not a vocab concern. What we want to verify in Phase 1 validation:

- Common math operators encode efficiently (1–2 tokens, not 4+)
- LaTeX commands like `\\frac`, `\\sum`, `\\int`, `\\sqrt` get reasonable tokens
- Hindi math notation (Devanagari digits ०१२३४५६७८९) — given Hindi is in our language mix, verify these tokenize well
- Number-splitting behavior: digits should split per-digit (or per-3-digits) consistently — this measurably helps arithmetic. Our tokenizer config already enables `digit_split: true` in `pre_tokenizer`. ✓

The tokenizer compression validation (`per_language_compression_floor: 0.85`) doesn't cover math directly. **Add a math-corpus benchmark to Phase 1 validation:** measure bytes-per-token on a held-out math sample and require ≥ 0.90 of cl100k baseline.

## 5. Evaluation

| Benchmark | Purpose | Gate |
|---|---|---|
| GSM8K (8-shot CoT) | Grade-school arithmetic + reasoning | GATE 2, 4 |
| MATH (4-shot) | Competition math | GATE 4 |
| MMLU-stem (math subset) | Knowledge-style math | GATE 2 |
| MGSM | GSM8K in 11 languages (incl. Hindi) | GATE 4 |
| HumanEval, MBPP | Code (math-adjacent for numerical methods) | GATE 2, 4 |
| Numerical-only mini-suite (in-house) | Pure arithmetic, modular sanity | continuous |

**Targets at the 1.24B class** (rough; refined when we see pilot signal):

| Bench | Target after base | Target after SFT | Target after reasoning tuning |
|---|---|---|---|
| GSM8K (8-shot) | 8–15% | 25–40% | 45–60% |
| MATH (4-shot) | 1–3% | 5–12% | 12–20% |
| HumanEval | 18–25% | 30–40% | 35–45% |

For comparison: Llama 3.2 1B reports GSM8K ≈ 44.4% (8-shot, with instruction tuning + reasoning), MATH ≈ 30%. Reaching parity at 1.24B with our token budget is plausible but not guaranteed.

## 6. Tool-use math (Phase 10, optional)

For arithmetic that's tedious in raw token space (e.g., 7-digit multiplication), give the model a Python-eval tool. This trades raw "model knows arithmetic" for "model knows when to call the calculator." Most modern assistants do this. Implementation lands in Phase 10 alongside general tool-use SFT.

## 7. Open questions

1. Use of synthetic math data from the teacher API once it arrives — what's the right share vs. real datasets?
2. Should the pilot include the decay-phase math bump or stay uniform? **Recommend uniform 7% for pilot** (we want to see clean signal at one mix level before testing curriculum effects).
3. How aggressive should the math-quality filter be at the 1B scale? Loose filtering preserves volume; tight filtering raises quality. Default: tight on synthetic-looking docs, loose on academic.
