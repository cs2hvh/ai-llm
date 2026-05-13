# Stage 3 Data-Prep Rust Migration Plan — DRAFT v0.1

**Status**: 🟡 DRAFT, awaiting user + reviewer cross-verification before Session B starts.
**Author**: Session A (Claude), 2026-05-13, after 3 parallel research agents.
**Owner once approved**: Session B (works on a fork in a separate folder with their own git remote).
**Scope target**: rewrite the Python data-prep hot paths in Rust + adopt one modern algorithmic upgrade, so the Stage 3 (600B-1T token) build is feasible on our single 128-core node.
**Effort target**: 3-4 weeks of focused work by Session B (~120 hours).
**Speedup target**: 5B-token build wall time **2.5 h → 15-25 min** (≈10× E2E); 1T-token build wall time **70-100 days → 2-5 days** (≈30-50× E2E).

---

## 0. Why this plan exists (1-paragraph context)

Our current data pipeline is single-thread Python per source, GIL-bound. For Stage 1 pilot (5B tokens, ~2.5 hr) this is fine. For Stage 3 base v1 (600B-1T tokens), naive scaling = 70-100 days of CPU wall — unacceptable. Frontier labs (Llama 3, DeepSeek-V3, OLMo 2, DCLM) get past this with either huge clusters or Rust hot paths or both. We don't have a cluster; we do have 128 cores idle. The right answer is to migrate the per-doc hot path into Rust and pick up one or two algorithmic upgrades that have become standard in 2024-2025.

**This is parallel work to Stage 1/2** — Session B forks the repo + works independently. Session A keeps owning the Stage 1 pilot + Stage 2 prep on the main branch. Merge story documented in §10.

---

## 1. Research that backs this plan

Three parallel research agents ran on 2026-05-13. Key findings (compressed):

- **Datatrove (HF, used to build FineWeb) is 97% Python**. Its "speed" comes from Slurm cluster scale, not from native code. **Single-node won't scale better than our current setup.** Don't adopt as a framework. Reuse selectively.
- **Dolma (AI2, used to build OLMo 2's 3T corpus) has its dedup hot path in Rust** (`src/bloom_filter.rs`, `src/deduper.rs`). This is the architecture pattern we want.
- **DCLM-Baseline** benchmarked MinHash vs Suffix-Array vs Bloom-Filter-Fuzzy (BFF) and picked **BFF** for everything beyond ~10 TB.
- **`rensa`** ([github.com/beowolx/rensa](https://github.com/beowolx/rensa)) is a Rust MinHash crate with PyO3 bindings, ~40× faster than `datasketch`. Drop-in compatible.
- **Suffix-array global dedup** (Lee et al. 2021, `google-research/deduplicate-text-datasets`) would need ~24 TB RAM for our 1T target — **not viable on our 251 GB node**.
- **fastText quality classifiers** are standard in 2024-2025 (Llama 3, OLMo 2, DCLM all use them). [FineWeb-Edu classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) gives 2k docs/sec/CPU and was trained on 450k Llama-3-70B annotations.
- **PyO3 0.28.3 + maturin 1.9+** is the canonical Python-Rust interop for this kind of work. Same pattern as `huggingface/tokenizers`, `pola-rs/polars`, `rensa`.
- **`xxhash-rust` (xxh3)** is the safe default for hashing. `gxhash` is faster but needs AES-NI hardware fallback — don't gate the build on it.
- **DCLM Bloom-Filter-Fuzzy (`revbucket/bff`)** for global dedup: ~2-3 days for 1T tokens on a 128-core box with sharding. Already battle-tested.

Full agent reports available — ping Session A if you want to read the raw research.

---

## 2. Goals + non-goals

### Goals
1. **Single-node 1T-token build in ≤5 days wall** (vs current ~70-100 days projected).
2. **Output bit-for-bit equivalent to current Python pipeline** for any code path that's NOT explicitly an algorithmic upgrade. (Algorithm upgrades like BFF dedup intentionally diverge; they need separate quality validation.)
3. **No regression in train-time correctness**: the packed shards, manifests, provenance schemas (`DocSpan`, `SequenceMeta`, `CorpusManifest`) stay binary-identical. Training pipeline doesn't care if data came from Python or Rust workers.
4. **Maintainable**: Rust code lives in a clearly scoped subdirectory; build via `maturin` integrated with existing `pyproject.toml`; CI builds wheels for at least linux x86_64.
5. **Reviewable in chunks**: each phase ships independently with passing tests + perf delta. Don't land a giant single PR.

### Non-goals
1. ❌ **NOT** rewriting the whole project in Rust. JAX/Keras model code stays Python; Orbax checkpoints stay Python; HF dataset enumeration stays Python.
2. ❌ **NOT** switching to Datatrove or any other framework. Our current architecture (per-source build → compose pass → packed shards) is sound; we're upgrading the hot path under it.
3. ❌ **NOT** GPU dedup (NVIDIA NeMo-Curator). We have a CPU box. GPU dedup is a separate optimization for ephemeral GPU workers later.
4. ❌ **NOT** training our own fastText quality classifier. Adopt FineWeb-Edu's + DCLM's. This is decided per §6.
5. ❌ **NOT** changing the dedup algorithm in Phase 1. Phase 1 is "same algorithm, faster implementation." Algorithm change (MinHash → BFF) is a separate Phase 5.

### Out of scope for this plan (handled elsewhere)
- Per-source val loss enhancement (reviewer P0-1) — Session A or future work on main branch.
- Forward-only eval_step for FSDP — Session A, Stage 2 prep.
- Source-sharding via `HF dataset.shard` — DEPRECATED by this plan (Rust hot path makes per-source parallelism > 1 unnecessary).

---

## 3. Architecture

### 3.1 Repo layout (in the fork)

```
ai-llm/                                  # fork root (Session B's clone)
├── pyproject.toml                       # existing — pure Python package
├── src/myllm/                           # existing Python source (UNCHANGED EXCEPT where noted)
│   ├── data/
│   │   ├── build.py                     # MINOR: swap inner-loop calls to native module
│   │   ├── dedupe.py                    # MINOR: route through native MinHash when available
│   │   ├── decontamination.py           # MINOR: route through native n-gram check
│   │   ├── filters.py                   # MINOR: route through native filter pass
│   │   └── (rest unchanged)
│   └── (rest unchanged)
├── crates/                              # NEW — Rust workspace
│   └── myllm_dataprep/
│       ├── Cargo.toml                   # workspace root
│       ├── core/                        # pure Rust (no PyO3), unit-tested via cargo
│       │   ├── Cargo.toml
│       │   ├── benches/
│       │   │   └── perf.rs              # criterion benchmarks
│       │   └── src/
│       │       ├── lib.rs
│       │       ├── minhash.rs           # MinHash signature + LSH index
│       │       ├── decontam.rs          # n-gram Bloom set membership
│       │       ├── filters.rs           # length / repetition / symbol_ratio / PII
│       │       ├── bff.rs               # (Phase 5) Bloom-Filter-Fuzzy global dedup
│       │       └── fasttext.rs          # (Phase 6) fastText classifier wrapper
│       └── python/                      # PyO3 bindings crate
│           ├── Cargo.toml
│           ├── pyproject.toml           # maturin build config (separate from repo root)
│           ├── python/
│           │   └── myllm_dataprep/
│           │       ├── __init__.py
│           │       └── _native.pyi      # type stubs for IDE/mypy
│           └── src/
│               └── lib.rs               # #[pymodule] wrappers, GIL release blocks
└── tests/
    ├── test_native_minhash.py           # NEW — Rust vs Python equivalence
    ├── test_native_decontam.py          # NEW — same
    ├── test_native_filters.py           # NEW — same
    └── (existing tests unchanged)
```

**Why a separate Cargo workspace under `crates/`**: clean isolation; Rust developers don't need to navigate the Python source tree. Same pattern as `huggingface/tokenizers` (`tokenizers/` core + `bindings/python/` wrapper) and `polars` (`crates/` workspace + `py-polars/`).

**Why a separate `pyproject.toml` inside `crates/myllm_dataprep/python/`**: that's where maturin builds the wheel. Repo-root `pyproject.toml` stays the place where `pip install -e .` installs the pure-Python project. After Session B's work lands, operators do `pip install -e .` (Python) + `pip install -e crates/myllm_dataprep/python` (Rust extension) OR we publish the Rust crate as a separate PyPI package `myllm-dataprep` that the main package depends on.

### 3.2 Python-side integration pattern

The existing Python modules (`dedupe.py`, `decontamination.py`, `filters.py`) keep their public APIs unchanged. Inside each, we add a "use native if available, fall back to Python" shim:

```python
# src/myllm/data/dedupe.py
try:
    from myllm_dataprep import MinHasher as _NativeMinHasher  # Rust impl
    _USE_NATIVE = True
except ImportError:
    _USE_NATIVE = False  # pure-Python fallback

class Deduplicator:
    def __init__(self, config: MinHashConfig | None = None) -> None:
        if _USE_NATIVE:
            self._hasher = _NativeMinHasher(...)
        else:
            self._hasher = MinHasher(self.config)
        ...
```

This pattern means:
- Existing CPU dev box workflows keep working with pure-Python (no Rust build needed for casual development).
- Production builds opt into Rust via `pip install myllm-dataprep`.
- Existing tests keep running against both paths (with a fixture switching impl).
- Migration is **incremental** — we can ship phase by phase without flag-day cutovers.

### 3.3 Interop choices (locked per Agent 2's research)

- **PyO3 0.28.3**: free-threaded Python 3.13t support, `py.detach()` for GIL release.
- **maturin 1.9+**: wheel build; abi3-py310 → one wheel per platform, not per Python version.
- **Batch API at the boundary**: `compute_signatures_batch(docs: list[str]) -> np.ndarray` not per-doc calls. PyO3 FFI has ~200ns overhead; at 600M calls that's 2 minutes wasted. Batch in chunks of 1k-10k.
- **GIL release inside Rust**: every `pyfunction` wraps its compute in `py.detach(|| { ... })` so multiple Python threads can call concurrently.
- **numpy zero-copy** via [`rust-numpy`](https://github.com/PyO3/rust-numpy) for returning bulk signature arrays without copy.

---

## 4. Phased work breakdown

Each phase is a separate landable PR. Each phase has:
- Concrete files to create / modify
- Acceptance criteria (functional + perf)
- A test plan
- Estimated effort

### Phase 0: Project scaffolding (1-2 days)

**Goal**: get the empty Rust workspace building and shipping a Python-importable extension.

**Files**:
- `crates/myllm_dataprep/Cargo.toml` (workspace root)
- `crates/myllm_dataprep/core/Cargo.toml` + `src/lib.rs` with a single placeholder function `hello() -> &'static str`
- `crates/myllm_dataprep/python/Cargo.toml`, `pyproject.toml`, `src/lib.rs` with `#[pymodule]` exposing `hello()`
- `crates/myllm_dataprep/python/python/myllm_dataprep/__init__.py` re-exporting from `_native`
- GitHub Actions workflow: `maturin build --release --zig` for linux x86_64
- Documentation: `crates/myllm_dataprep/README.md` explaining the build

**Acceptance**:
- `maturin develop --release` in `crates/myllm_dataprep/python/` succeeds
- `python -c "from myllm_dataprep import hello; print(hello())"` prints something
- CI builds + uploads wheel artifact
- All existing tests still pass

**Why first**: shake out build system before writing any real code.

### Phase 1: MinHash signature in Rust (3-4 days)

**Goal**: replace the `signature()` hot loop in `src/myllm/data/dedupe.py` with a Rust implementation, ≥40× speedup.

**Algorithm**: same as current (112 perms × `xxhash64`-with-seeds, take min per perm over all 5-shingles). NOT changing dedup algorithm; just porting the inner loop.

**Implementation choices**:
- Use `xxhash-rust` crate with xxh3 (NOT xxh64 — xxh3 is 5-10× faster). **BUT** — `xxhash64` is the algorithm our existing index files use. So:
  - Phase 1a: keep xxh64 (bit-for-bit compatible with current files). Speedup ~40× from GIL escape + native loop.
  - Phase 1b (optional, later): migrate index file format to xxh3. Speedup additional ~5-10×.
- Use rayon for batch parallelism inside Rust.
- `release_gil` block around the compute loop.

**Files**:
- `crates/myllm_dataprep/core/src/minhash.rs` (~200 LoC)
- `crates/myllm_dataprep/python/src/lib.rs` — add MinHasher pymodule (~80 LoC)
- `crates/myllm_dataprep/core/benches/minhash.rs` — criterion bench
- `tests/test_native_minhash.py` (~150 LoC) — equivalence + perf tests
- `src/myllm/data/dedupe.py` — add the `_USE_NATIVE` shim (~10 LoC)

**Acceptance**:
- For ≥10k docs from FineWeb-Edu, Rust `signature()` returns bit-identical 112-tuple to Python `signature()`.
- Microbench: ≥40× speedup on a 1k-doc batch (target: <10ms total vs ~280s in Python).
- E2E: re-run a 100M-token build with `--no-decontam`, verify shard manifests bit-identical to baseline build.
- All existing dedup tests pass under both paths (fixture parameterized).

### Phase 2: N-gram decontam in Rust (2-3 days)

**Goal**: replace the per-doc n-gram check in `src/myllm/data/decontamination.py` with a Rust implementation. ≥20× speedup.

**Algorithm**: same (8-gram + 13-gram, hash each, check `HashSet<u64>` membership). Existing JSON index files stay the same.

**Implementation choices**:
- `hashbrown::HashSet<u64>` (Rust stdlib HashMap successor, ~2× faster).
- Pre-hash with the same `BuildHasher` as the set; skip rehashing on lookup.
- Load `decontamination_index_*.json` on first use (Python loads → passes set across boundary, OR Rust reads JSON directly via `serde_json`).
- Batch API: `check_batch(docs: list[str]) -> list[set[str]]` returning per-doc matches per benchmark.

**Files**:
- `crates/myllm_dataprep/core/src/decontam.rs` (~150 LoC)
- `crates/myllm_dataprep/python/src/lib.rs` — add DecontamIndex pymodule (~60 LoC)
- `tests/test_native_decontam.py` — equivalence + perf
- `src/myllm/data/decontamination.py` — `_USE_NATIVE` shim

**Acceptance**:
- For ≥1000 docs (mix of contaminated + clean), Rust returns same `dict[benchmark_id, n_matches]` as Python.
- Microbench: ≥20× speedup (target: ~0.2ms/doc vs ~10ms/doc).
- E2E: 100M-token build with decontam enabled produces same `docs_contaminated` count ± rounding tolerance.

### Phase 3: Filter chain in Rust (3-4 days)

**Goal**: replace the per-doc length/repetition/symbol-ratio/PII filters in `src/myllm/data/filters.py` with Rust. ≥10× speedup.

**Implementation choices**:
- `regex` crate for PII patterns (already uses `aho-corasick` internally for literal-prefix optimization).
- `memchr` for byte-level whitespace/newline scanning.
- One-pass byte iteration to compute length / category ratios / repetition window simultaneously.
- Each filter sub-rule keeps its current threshold semantics for output equivalence.

**Files**:
- `crates/myllm_dataprep/core/src/filters.rs` (~400 LoC)
- `crates/myllm_dataprep/python/src/lib.rs` — add Filter pymodule (~80 LoC)
- `tests/test_native_filters.py` — equivalence + perf
- `src/myllm/data/filters.py` — `_USE_NATIVE` shim

**Acceptance**:
- For ≥10k docs covering all filter rejection cases, Rust + Python agree on `(passed: bool, reject_reason: str | None)` for every doc.
- Microbench: ≥10× speedup.
- E2E: full pilot corpus build with both paths produces same `docs_filtered` count.

### Phase 4: Integration + E2E benchmarking (2-3 days)

**Goal**: with all 3 hot paths in Rust, run a real 5B-token build on the same source list and compare wall time + output integrity to baseline.

**Tasks**:
- Add a `--use-native` flag to `scripts/build_packed_corpus.py` (default: auto-detect via `_USE_NATIVE`).
- Update `scripts/run_parallel_builds.py` to forward the flag.
- Run 5B build using existing `pretrain_mix_pilot.yaml` with all-Rust hot paths.
- Diff resulting per-source manifests against Stage 1 build (should be bit-identical for filter/decontam outcomes; MinHash dedupe outputs may differ in DOC ORDER due to non-deterministic parallelism but the SET of kept docs should match).

**Acceptance**:
- 5B build wall ≤ 30 min (vs current 2.5 hr). **Stretch: ≤ 15 min**.
- Per-source manifests bit-identical OR fully explained (e.g. dedup ordering differences).
- Memory peak < 100 GB during build (currently ~50 GB).
- No regression in `pytest -q` suite.

### Phase 5: Algorithmic upgrade — Bloom Filter Fuzzy (BFF) global dedup (5-7 days, OPTIONAL for Stage 2 prep; REQUIRED for Stage 3)

**Goal**: replace per-source MinHash with global BFF (Bloom-Filter-Fuzzy) dedup. Both algorithm change AND scale change (per-source → global).

**Why**: DCLM-Baseline benchmarked MinHash vs SuffixArray vs BFF and picked BFF for everything past ~10 TB. SlimPajama is the standard reference. For 1T tokens, MinHash starts hitting memory ceilings (in-memory LSH index gets huge for cross-source dedup).

**Implementation choices**:
- Port `revbucket/bff` core algorithm to our `crates/myllm_dataprep/core/src/bff.rs` (~600 LoC).
- Default params from DCLM: `min_ngram=5`, `max_ngram=13`, `filtering_threshold=0.8`, `fp_rate=0.01`.
- Sharding strategy: hash document by content-hash mod N_SHARDS (default 16); each shard fits in <200 GB RAM.
- Output: per-document keep/drop flag in a sidecar manifest; original packed shards unchanged.

**Files**:
- `crates/myllm_dataprep/core/src/bff.rs` (~600 LoC)
- `crates/myllm_dataprep/python/src/lib.rs` — BFF pymodule
- `scripts/run_global_dedup.py` (new) — post-build pass that reads all per-source shards, runs BFF, emits sidecar drop list
- `tests/test_native_bff.py` — equivalence with DCLM reference if accessible, otherwise property tests
- `docs/dedup_strategy.md` (new) — document the algorithm change + the per-source-vs-global tradeoff for the reviewer

**Acceptance**:
- BFF run on a 5B-token build completes in <30 min.
- Drop rate within 5% of MinHash dedup for the same corpus (sanity check).
- Sidecar manifest format documented + readable from `compose_mixed_corpus.py`.

**HOLD if**: Stage 2 doesn't need global dedup (per-source MinHash on Rust may be enough). Decide before starting this phase based on Stage 1 pilot results.

### Phase 6: fastText quality classifier (3-4 days, OPTIONAL but recommended)

**Goal**: add a quality-score column to the filter chain using FineWeb-Edu + DCLM fastText classifiers. Drop docs below configurable threshold.

**Implementation choices**:
- Use `fasttext` Rust crate (`fasttext-rs`) inside the existing Rust filter pipeline → no Python boundary cost per doc.
- Load both FineWeb-Edu (`HuggingFaceFW/fineweb-edu-classifier`) and DCLM (`mlfoundations/dclm/quality_classifier`) classifier files.
- Per-doc: emit score for each classifier; drop if below threshold (configurable in pretrain_mix yaml).
- Default thresholds from public docs: FineWeb-Edu ≥3.0 (their "high educational value" bucket), DCLM ≥0.033.

**Files**:
- `crates/myllm_dataprep/core/src/fasttext.rs` (~150 LoC)
- `scripts/download_quality_classifiers.py` (new) — fetch the two classifier files
- `configs/data/pretrain_mix_pilot.yaml` — new `quality_classifier:` section
- `tests/test_native_fasttext.py`

**Acceptance**:
- Per-doc score generation at ≥2k docs/sec/core (matches FineWeb-Edu's published number).
- Drop rate at default thresholds within 10% of FineWeb-Edu's published rate on a CommonCrawl sample.

### Phase 7: Documentation + handoff (1-2 days)

- Update `docs/PROJECT_OVERVIEW.md` to reflect new data-prep architecture.
- Write `docs/data_prep_architecture.md` documenting the Python↔Rust integration, the BFF dedup choice, the fastText classifier choice.
- Migration guide for operators: how to set up the build environment, what flags to pass.
- Performance benchmark report: before/after on 5B + projected 1T.

---

## 5. Verification protocol

### Output equivalence (Phases 1-3)

For each migrated hot path, the acceptance criterion is **bit-for-bit output equivalence with the Python implementation** for the same input. Specifically:

```python
# tests/test_native_<X>.py pattern
@pytest.mark.parametrize("impl", ["python", "rust"])
def test_<X>_output(impl):
    if impl == "rust":
        monkeypatch.setattr(module, "_USE_NATIVE", True)
    else:
        monkeypatch.setattr(module, "_USE_NATIVE", False)
    # run pipeline on fixed corpus, compare outputs to canonical baseline
```

For MinHash: signatures must be bit-identical (same 112-tuple of u64s).
For decontam: match dict must be identical (same benchmark IDs, same match counts).
For filters: `(passed, reason)` per doc must be identical.

### Behavioral equivalence (Phase 4)

Run full 5B-token build with Rust hot paths. Compare per-source manifest against baseline:
- `docs_seen` MUST match exactly (same HF stream, deterministic iteration).
- `docs_filtered` MUST match exactly (deterministic filter rules).
- `docs_contaminated` MUST match exactly (deterministic decontam).
- `docs_deduped` MAY differ (MinHash LSH order-dependent across parallel workers).
- `tokens_emitted` should match within 0.1% (same tokenizer).
- `sequences_emitted` should match within 0.1%.

### Performance benchmarks

Each phase ships with a `criterion` micro-benchmark in `core/benches/` + a Python-side `pytest-benchmark` test. CI tracks regression against the previous commit.

### Phase 5+ has DIFFERENT acceptance

For BFF dedup (Phase 5), output WILL differ from MinHash (different algorithm). The acceptance criterion is "drop rate ± 5% of MinHash, no regression in downstream training quality" — the latter requires a small Stage-1-like pilot to validate, so Phase 5 implicitly requires a pilot run.

---

## 6. Locked decisions (for the reviewer to verify)

| # | Decision | Rationale | Alternative considered | Reversibility |
|---|---|---|---|---|
| D1 | **Use PyO3 + maturin** for Python-Rust interop | Standard for HF tokenizers, polars, rensa; abi3 wheel matrix is manageable | ctypes (too much boilerplate); subprocess IPC (too much overhead per call) | High — could swap interop if we hit PyO3-specific issues |
| D2 | **Keep MinHash algorithm in Phase 1** (just port to Rust) | Bit-for-bit compatible with existing index files; small surface; immediate ~40× win | Jump straight to BFF (algorithm change risk) | High — Phase 5 will deprecate it anyway |
| D3 | **Use `xxhash-rust` xxh64** in Phase 1, NOT xxh3 or gxhash | Bit-identical to existing Python xxhash output; bridge compatibility | xxh3 (5-10× faster but breaks compat); gxhash (no AES-NI fallback) | Medium — Phase 1b can migrate to xxh3 with a one-time corpus rebuild |
| D4 | **Inline decontam in Rust** (not post-hoc batch) | Preserves audit trail per-doc; avoids separate pipeline; simpler operator model | Post-hoc decontam (faster build but worse traceability) | High — Phase 5 could move to post-hoc if perf becomes a blocker |
| D5 | **Adopt fastText classifiers from FineWeb-Edu + DCLM**, not train our own | Frontier-lab standard practice; 1B teams don't train quality classifiers; lift is below ablation noise | Train our own (10-20× more engineering work, no clear quality win) | Medium — could train custom later if quality plateaus |
| D6 | **BFF (Bloom-Filter-Fuzzy) global dedup** in Phase 5, NOT MinHash global | DCLM benchmarked all three and picked BFF; SlimPajama scale validation; memory-efficient; faster | MinHash global (in-memory LSH won't fit for 1T); Suffix-Array (won't fit either) | Low — picking BFF locks the algorithm |
| D7 | **DO NOT adopt Datatrove as framework** | 97% Python in hot path; designed for Slurm not single-node; we'd inherit our current bottleneck | Adopt Datatrove + Rust extensions (too much rework) | Medium — could revisit if we ever get a real cluster |
| D8 | **Output equivalence is bit-for-bit for Phases 1-3** | Trust requires it; alternative is silent quality degradation | "Statistically equivalent" (looser; harder to debug regressions) | Low — Phase 5+ explicitly breaks this for algorithm changes |
| D9 | **Repo layout: `crates/myllm_dataprep/` subdir with separate Cargo workspace + Python wrapper** | Same as huggingface/tokenizers, polars; clean separation | Top-level Rust files (clutter); monorepo with python+rust in same Cargo project (build complexity) | Medium — could restructure but it's load-bearing |
| D10 | **Session B works on a fork in a separate folder with their own git remote** | Doesn't conflict with Stage 1 main-branch work; PR back when ready | Branch in same repo (merge conflicts); coordinate via this doc | Medium — could re-merge to single-repo after Phase 1 ships if conflicts are manageable |

---

## 7. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Wheel matrix complexity (Linux x86_64 + arm64 + macOS + Windows × Python 3.10-3.13 = 16 wheels) | High | Medium | Use abi3-py310 (collapse Python axis to 1); ship linux x86_64 only in Phase 0, add others later |
| R2 | GIL release discipline — forgetting `py.detach()` means zero parallelism gain | Medium | High | Stress test: spawn 8 Python threads calling extension, diff results vs single-thread; explicit linting in code review |
| R3 | Bit-equivalence harder than expected (Python's float / int / hash quirks) | Medium | Medium | Build small "golden" test corpus committed to repo; run before each PR; if quirk found, document + decide whether to match or document deviation |
| R4 | xxh64 vs xxh3 confusion — accidentally using the wrong one breaks index file compatibility | Medium | High | Explicit tests asserting hash output matches known Python value; CI guard |
| R5 | Compile time (1k LoC Rust + PyO3 + regex + hashbrown = ~150 crates, 3-5 min cold compile) | High | Low | sccache in CI; iterate on small testable units; keep binding crate small |
| R6 | Session B's fork diverges too far from main, painful to merge | Medium | High | Tight PR-per-phase discipline (don't accrete >1 phase before merge); rebase against main weekly; coordinate in CLAUDE_COLLAB.md |
| R7 | BFF + Bloom filter memory blowup at 1T scale (filters get huge) | Medium | High | Validate at 5B + 20B scales first; pre-compute memory requirements; shard strategy documented |
| R8 | fastText classifier dependencies (binaries from HF) may not be redistributable | Low | Medium | Check FineWeb-Edu + DCLM classifier licenses before Phase 6 (likely Apache-2 / similar but verify) |
| R9 | User direction changes mid-migration (already saw scope shifts on 2026-05-13) | High | Medium | Plan in phases so partial work is still useful; align with reviewer before starting each phase |
| R10 | Reviewer pushback on algorithm changes (Phase 5 BFF in particular) | Medium | High | Submit Phase 5 design doc separately for reviewer approval BEFORE coding |

---

## 8. Timeline + budget

**Total estimate**: 3-4 weeks of focused engineering by Session B (~80-120 hours).

| Phase | Effort | Wall time (1 eng full-time) |
|---|---|---|
| Phase 0 | 1-2 days | 1-2 days |
| Phase 1 | 3-4 days | 3-4 days |
| Phase 2 | 2-3 days | 2-3 days |
| Phase 3 | 3-4 days | 3-4 days |
| Phase 4 | 2-3 days | 2-3 days |
| Phase 5 (optional, Stage 3) | 5-7 days | 5-7 days |
| Phase 6 (optional, recommended) | 3-4 days | 3-4 days |
| Phase 7 | 1-2 days | 1-2 days |
| **Phases 0-4 (MVP for Stage 2)** | **11-16 days** | **~2.5 weeks** |
| **Phases 0-7 (full migration for Stage 3)** | **20-29 days** | **~4-6 weeks** |

**Compute budget**: minimal — all work runs on the existing 128-core dev box. CI runs need GitHub Actions standard runners (free for public repos; small monthly cost for private). No GPU needed.

**Reviewer involvement**: cross-verify this plan (1 hr); review Phase 1 PR (1 hr); review Phase 5 design doc separately (1 hr); review final integration PR (2 hr). Total ~5 hr of reviewer time across the project.

---

## 9. Acceptance criteria for the whole migration

A successful migration is one where, by the end of Phase 7:

1. **Performance**: 5B build wall ≤ 20 min (vs 2.5 hr today). 1T build projected ≤ 5 days (vs 70-100).
2. **Correctness**: bit-for-bit output equivalence on Phases 1-3; documented algorithmic-equivalence on Phase 5; full test suite passes.
3. **Maintainability**: Rust crate is buildable from a fresh clone in <10 min with `pip install -e .` + `pip install -e crates/myllm_dataprep/python/`. Documentation exists at `docs/data_prep_architecture.md`.
4. **Reversibility**: `_USE_NATIVE=False` shim still works for any operator who can't or won't build Rust.
5. **Reviewed**: reviewer signed off on the algorithm choices (especially Phase 5 BFF).

---

## 10. Coordination with Session A (main branch)

Session A continues working on the main branch (Stage 1 pilot, Stage 2 prep, reviewer's P0 items). Session B works on the fork. Coordination protocol:

- **Sync point: weekly** (or at the end of each phase). Session B rebases against main; if any of the files Session B touches (esp. `src/myllm/data/{dedupe,decontamination,filters}.py`) were modified on main, resolve conflicts.
- **Conflict-likely files** Session A may touch: `src/myllm/data/build.py`, `dedupe.py`, `decontamination.py`, `filters.py`, `scripts/build_packed_corpus.py`. Session A should AVOID changing these on main during the migration unless absolutely necessary; if they must, document the change in `docs/CLAUDE_COLLAB.md` so Session B sees it on the next sync.
- **Conflict-unlikely files** Session A is likely to touch: `src/myllm/training/*`, `scripts/run_pretrain.py`, `configs/*`. Session B should AVOID these.
- **PR back to main**: at the end of each phase. Session A reviews + merges. The `_USE_NATIVE` shim means each phase landing is non-breaking (Python fallback still works).
- **Live doc**: `docs/CLAUDE_COLLAB.md` "Session B" subsection should reflect current phase + ETA + blockers.

---

## 11. Open questions for reviewer + user (cross-verify before approving)

1. **Phase order**: should Phase 5 (BFF) happen before or after Phase 6 (fastText)? My draft says BFF first (more impactful for Stage 3), but reasonable to argue fastText first (gives better data even if dedup stays per-source).
2. **Skip Phase 5 entirely?**: if Stage 2 (1B rehearsal, 10-30B tokens) shows per-source MinHash is fine, can we defer BFF until we actually do Stage 3? My draft says yes, but the trigger for "do BFF now" should be explicit.
3. **Reviewer's preference on framework adoption**: my draft DOES NOT adopt Datatrove. Reviewer should verify this is acceptable (they may push for adopting it for governance / community-validation reasons even at the perf cost).
4. **Hardware target for `gxhash`**: if our deploy box always has AES-NI, we could use gxhash for an extra 5-10× hash perf. Currently I've vetoed it (D3) for safety. User to confirm hardware.
5. **License audit for fastText classifiers**: FineWeb-Edu's is on HuggingFace; need to confirm Apache-2 / MIT before bundling.
6. **Wheel matrix scope**: ship just linux x86_64 (the dev box + RunPod default), or invest in arm64 + macOS for laptop dev workflow? My draft ships x86_64 only in Phase 0 + adds others later.
7. **What "BFF spec" do we follow exactly?**: DCLM's `revbucket/bff` is one canonical implementation. Different fork variants exist (`ai2-fuzzy-substr` branch). Reviewer to recommend.
8. **Should Session B's fork merge back into main repo (one ownership) or stay separate (two repos)?** My draft assumes merge-back PR per phase. Reviewer to verify this is operationally OK.

---

## 12. References

Sourced from 3 parallel research agents on 2026-05-13:

### Frameworks + frontier patterns
- [HuggingFace datatrove](https://github.com/huggingface/datatrove)
- [AI2 Dolma](https://github.com/allenai/dolma) ([dedup docs](https://github.com/allenai/dolma/blob/main/docs/deduplication.md))
- [DCLM toolkit](https://github.com/mlfoundations/dclm)
- [The FineWeb Datasets (2406.17557)](https://arxiv.org/html/2406.17557v1)
- [Llama 3 paper (2407.21783)](https://ar5iv.labs.arxiv.org/html/2407.21783)
- [DeepSeek-V3 Tech Report (2412.19437)](https://arxiv.org/abs/2412.19437)
- [OLMo 2 paper (2501.00656)](https://arxiv.org/pdf/2501.00656)
- [DataComp-LM benchmark (2406.11794)](https://arxiv.org/abs/2406.11794)

### Algorithms
- [Lee et al. 2021 — Deduplicating Training Data (2107.06499)](https://arxiv.org/abs/2107.06499)
- [google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)
- [LSHBloom paper (2411.04257)](https://arxiv.org/abs/2411.04257)
- [SlimPajama blog](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)
- [revbucket/bff (BFF dedup)](https://github.com/revbucket/bff)
- [Fraunhofer 2024 dedup eval](https://link.springer.com/chapter/10.1007/978-3-031-82481-4_27)

### Rust crates + interop
- [PyO3](https://github.com/PyO3/pyo3) ([free-threading guide](https://pyo3.rs/v0.28.2/free-threading))
- [Maturin](https://github.com/PyO3/maturin) ([layout guide](https://www.maturin.rs/project_layout.html))
- [huggingface/tokenizers](https://github.com/huggingface/tokenizers)
- [pola-rs/polars](https://github.com/pola-rs/polars)
- [beowolx/rensa (Rust MinHash)](https://github.com/beowolx/rensa)
- [xxhash-rust](https://crates.io/crates/xxhash-rust)
- [hashbrown](https://github.com/rust-lang/hashbrown)
- [BurntSushi/aho-corasick](https://github.com/BurntSushi/aho-corasick)
- [BurntSushi/memchr](https://github.com/BurntSushi/memchr)
- [rust regex docs](https://docs.rs/regex/latest/regex/)

### Classifiers
- [FineWeb-Edu classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier)
- [fasttext-rs](https://github.com/dfdx/fasttext-rs)

---

## END OF DRAFT

Reviewer + user, please mark §6 (Locked decisions) with explicit ack / push-back per row, and answer §11 (Open questions) before Session B starts Phase 0. Once approved, Session B claims this work in `docs/CLAUDE_COLLAB.md` and begins.
