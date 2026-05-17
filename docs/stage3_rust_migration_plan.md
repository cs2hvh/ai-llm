# Stage 3 Data-Prep Rust Migration Plan — v0.2.1

**Status**: 🟢 LOCKED 2026-05-13 after external reviewer pass + v0.2.1 cleanup of stale upstream references. Implementer can start.
**Author**: Session A v0.1 (2026-05-13 morning) + external reviewer pass v0.2 (2026-05-13 evening) + v0.2.1 cleanup pass (2026-05-13 evening, Session B).
**Owner**: implementer working in this same repo on feature branches, one PR per phase (see D10).
**Scope target**: hand-port Python data-prep hot paths to Rust + adopt Dolma's Rust bloom filter as a library + ship Python-side pre-Rust wins, so the Stage 3 (600B-1T token) build is feasible on our single 128-core node.
**Effort target**: 3-4 weeks of focused implementer work (~80-120 hours).
**Speedup target**: 5B-token build wall time **2.5 h → 15-25 min** (≈10× E2E); 1T-token build wall time **70-100 days → 2-5 days** (≈30-50× E2E).

---

## v0.2.1 changelog (cleanup deltas)

Post-lock cleanup pass — the v0.2 deltas were correct but didn't propagate to every section. Implementer reading top-to-bottom would hit contradictions. Fixed:

- **§1 Research bullets**: rensa / revbucket-bff / PyO3-free-threaded bullets rewritten to match the v0.2 locked decisions (D3 / D6 / D11). Old language preserved with `(v0.2 SUPERSEDED — see Dx)` prefixes so the audit trail stays readable.
- **§3.3 Interop**: PyO3 line corrected — `py.detach()` is correct, but the same line claimed free-threaded support (which D11 rejected). Decoupled.
- **Phase 1**: lead sentence said "use xxh3 (NOT xxh64 — xxh3 is faster)" — exact opposite of D3. Rewritten to `xxhash-rust` xxh64 as the locked choice; xxh3 stays as a deferred Phase 1b option.
- **D5**: "fastText from FineWeb-Edu + DCLM" — FineWeb-Edu isn't fastText (see Phase 6 split). Rewritten.
- **D10 ↔ §11 Q8 conflict**: D10 said "fork in separate folder with own remote", Q8 said "feature-branches-in-same-repo". Locked to feature-branches-in-same-repo; D10 rewritten; Q8 marked resolved.
- **Phase 5 HOLD trigger**: was vague ("if per-source MinHash is sufficient quality"); now concrete (per-source val_loss spread > 0.15 nats, OR memorization-probe extraction > 1%).
- **R7 + R10 risks**: were written for v0.1's from-scratch BFF port; rewritten for Phase 5's Dolma-library / DCLM-vendored path with Option C rented-box.
- **§8 Timeline**: Phase 5 5-7 → 5 days (Dolma kills the port); Phase 6 3-4 → 8 days (split 6a + 6b); Phase 0+ added.
- **§9 Acceptance #1**: clarified that the 1T ≤ 5 day budget = ~4 day local build + ~1 day rented-box dedup pass.
- **Phase 0+ sub-items**: renamed 3a/3b/3c → 0+a/0+b/0+c (visually parsed as Phase 3 sub-items otherwise).
- **§12 References**: added Dolma, DCLM/dedup/bff, messense/fasttext-rs, optimum, rust-numpy; tagged rensa + revbucket/bff as rejected; bumped PyO3 URL to v0.28.3.
- **D11**: moved from mid-§6 to end (was breaking numeric order between D6 and D7).

---

## v0.2 changelog (deltas from v0.1)

External reviewer (the implementer-to-be) caught 3 material flaws + 2 simplifications hiding in v0.1. All applied here:

- **D3 (hashing)**: KEEP xxh64 + hand-port. **DROP rensa adoption** — rensa hardcodes a non-xxh64 hasher, would silently invalidate our existing R2 decontam indexes. Realistic speedup 20-40× (matches the dolma/datasketch deltas), not the 608× rensa README claims.
- **D6 (BFF source)**: CHANGE from `revbucket/bff` (frozen 2024-04) → **`mlfoundations/dclm/dedup/bff`** (MIT, active 2025 commits, same author lineage, vendored copy).
- **D11 (concurrency, NEW)**: GIL-build Python + `py.detach()` + rayon inside Rust. **Single `abi3-py310` wheel**. Do NOT ship free-threaded (cp313t/cp314t) wheels — costs 10% single-thread perf + 15-20% memory for zero gain in our pipeline. PyO3 0.28 uses `Python::detach` (renamed from `allow_threads` in 0.27); implementer must consult 2026 docs, not 0.27-era blog posts.
- **Phase 5 (BFF)**: Use **Dolma as a Python library** (`pip install dolma` exposes AI2's Rust `bloom_filter.rs` + `deduper.rs` via PyO3) — kills the from-scratch 1500-2500 LoC BFF port. `by_ngram` mode is algorithmically equivalent to BFF. Effort ~5 days, not 5-7.
- **Phase 5 RAM decision**: **Option C — rent a 1 TB-RAM box (AWS x2gd.metal / u-3tb1) for ONE dedup pass at 1T scale** (~$30-50). 251 GB local fits ≤210B ngrams at fp=0.01 globally; 1T needs 1.20 TB. Per-snapshot (Option A) loses cross-source dedup quality; hash-partitioned sequential (Option B) adds 2-4 days of I/O. Option C matches DCLM-Baseline recipe verbatim and fits our "ephemeral workers" pattern.
- **Phase 6**: SPLIT into 6a/6b. **6a DCLM via `fasttext-rs`** (classic fastText, MIT) — unchanged from v0.1. **6b FineWeb-Edu via ONNX export + `ort` (Rust) or `candle`** — FineWeb-Edu is a transformer (Snowflake-arctic-embed + linear head), NOT classic fastText. `fasttext-rs` cannot load it. Community port (`kenhktsui/fineweb-edu-fasttext-classifier`) has 68% accuracy / Spearman 0.58 vs original — material quality regression, rejected.
- **Phase 0+ (NEW)**: Pre-Rust **Python wins** that de-risk before any Rust is written:
  - **0+a**: `src/myllm/data/build.py:125-162` — pending-list linear scan → sorted + binary search on `buf_start`. O(n) → O(log n) per sequence.
  - **0+b**: `src/myllm/data/filters.py:88-99` — repetition filter materializes `tuple(words[i:i+ngram_n])` for every position + builds a Counter. Replace with one-pass rolling window. Strictly O(n·ngram_n) Python tuple churn → O(n) one-pass scan.
  - **0+c**: Compose multiprocessing-on-shards. Defers Rusting compose UNLESS measured source-share drift > 2% across N workers. Two partitioning strategies to test: (i) range-partition by source share (guaranteed bounded drift, uneven worker sizes); (ii) shard-partition by output range (even worker sizes, drift can stack). Implementer benches both on 5B corpus before locking.
- **Compose Rusting**: deferred. Measured 92 seq/sec single-threaded CPU/GIL-bound on 2026-05-13 (not I/O-bound — reviewer's instinct was wrong here; `vmstat 0 bi 0 bo`). ~2 hr one-time per build. **Mandatory fix (multiprocessing OR Rust) before the 600B base run**, not the Stage 1 pilot.

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
- **`rensa`** ([github.com/beowolx/rensa](https://github.com/beowolx/rensa)) — *(v0.2 SUPERSEDED — see D3.)* Ships its own non-xxh64 multiply-shift hasher; would silently invalidate our existing R2 decontam indexes. Plan now hand-ports MinHash with `xxhash-rust` xxh64 for bit-compatibility. Realistic 20-40× speedup vs the 608× README claim (microbenchmark cherry-pick that doesn't apply to our workload).
- **Suffix-array global dedup** (Lee et al. 2021, `google-research/deduplicate-text-datasets`) would need ~24 TB RAM for our 1T target — **not viable on our 251 GB node**.
- **fastText quality classifiers** are standard in 2024-2025 (Llama 3, OLMo 2, DCLM all use them). [FineWeb-Edu classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) gives 2k docs/sec/CPU and was trained on 450k Llama-3-70B annotations.
- **PyO3 0.28.3 + maturin 1.9+** is the canonical Python-Rust interop. Same pattern as `huggingface/tokenizers`, `pola-rs/polars`. Free-threaded Python 3.13t/3.14t NOT in scope — see D11; plan uses GIL-build + `py.detach()` + rayon. Single `abi3-py310` wheel.
- **`xxhash-rust` xxh64** is the locked choice for MinHash (matches existing R2 index files — see D3). xxh3 is faster but requires a one-time corpus rebuild; deferred to optional Phase 1b. `gxhash` ruled out (needs AES-NI fallback).
- **Global dedup**: *(v0.2 SUPERSEDED — see D6 + Phase 5.)* `revbucket/bff` was frozen 2024-04. Plan now sources BFF from **`mlfoundations/dclm/dedup/bff`** (MIT, active 2025 commits) OR drives **Dolma's `by_ngram` mode as a Python library** (`pip install dolma` exposes AI2's Rust dedup via PyO3) — the latter killing the from-scratch port effort.

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

- **PyO3 0.28.3**: `py.detach()` for GIL release (renamed from `Python::allow_threads` in 0.28.0 — implementer must reference 2026 docs, not 0.27-era posts). Free-threaded Python NOT used — see D11.
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

### Phase 0+: Pre-Rust Python wins (NEW in v0.2, 1-2 days, do FIRST)

**Goal**: ship the Python-side algorithmic fixes BEFORE any Rust scaffolding. De-risks Phase 1 by separating "is the algorithm bad?" from "is Python slow?". If these alone are enough for Stage 2 throughput, the Rust migration can be paced more deliberately.

**0+a — `src/myllm/data/build.py:125-162` pending-list refactor**:
- Current: `for d in pending:` linear scan every sequence — O(n) per sequence, O(n²) over the corpus.
- Fix: keep `pending` sorted by `buf_start`; binary-search the cutoff. Most pending docs end before `sequence_length`, so the carry-over list stays small.
- Acceptance: same `_PendingDoc` carry semantics; same DocSpan output; 5-10× speedup on a synthetic stress test (>10k pending entries).

**0+b — `src/myllm/data/filters.py:88-99` one-pass repetition**:
- Current: builds `tuple(words[i:i+ngram_n])` for every position + Counter — O(n·ngram_n) Python tuple churn.
- Fix: rolling-hash one-pass; track top-share via streaming max while iterating words.
- Acceptance: same `FilterDecision(passed, reason, value)` output bit-for-bit; 5-10× speedup on a 10k-doc bench.

**0+c — Compose multiprocessing**:
- Current: 92 seq/sec single-threaded CPU/GIL-bound (measured 2026-05-13).
- Fix: split output range across N workers; each worker runs its own deficit-driven scheduler.
- Two strategies to bench: (i) range-partition by source share, (ii) shard-partition by output range.
- Acceptance: source-share drift ≤ 2% vs single-threaded baseline; ≥N/2 speedup at N=8 workers; same output schema.
- **Decision criterion**: if both strategies' drift > 2%, defer to Phase 4 Rust port. Otherwise compose stays in Python.

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
- Use `xxhash-rust` crate with **xxh64** (bit-compatible with existing R2 decontam indexes — see D3). Speedup ~20-40× from GIL escape + native loop + rayon parallelism.
- Phase 1b (optional, deferred): migrate index file format to xxh3 after a corpus rebuild. Additional ~5-10× perf, but breaks back-compat with existing R2 files. Decide only if Phase 1a doesn't hit the wall-time target.
- Use rayon for batch parallelism inside Rust.
- `py.detach()` block around the compute loop (NOT `allow_threads` — renamed in PyO3 0.28).

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

**Goal**: with all 3 hot paths in Rust + Phase 0+ Python wins applied, run a real 5B-token build on the same source list and compare wall time + output integrity to baseline.

**Tasks**:
- Add a `--use-native` flag to `scripts/build_packed_corpus.py` (default: auto-detect via `_USE_NATIVE`).
- Update `scripts/run_parallel_builds.py` to forward the flag.
- Run 5B build using existing `pretrain_mix_pilot.yaml` with all-Rust hot paths.
- Diff resulting per-source manifests against Stage 1 build (should be bit-identical for filter/decontam outcomes; MinHash dedupe outputs may differ in DOC ORDER due to non-deterministic parallelism but the SET of kept docs should match).
- **Compose Rusting decision point**: if Phase 0+ multiprocessing-compose passes the drift-≤-2% gate, compose stays Python and Phase 4 is just the per-source-build benchmark. If it failed, Phase 4 includes a `compose.rs` port (additional ~2 days).

**Acceptance**:
- 5B build wall ≤ 30 min (vs current 2.5 hr). **Stretch: ≤ 15 min**.
- Per-source manifests bit-identical OR fully explained (e.g. dedup ordering differences).
- Memory peak < 100 GB during build (currently ~50 GB).
- No regression in `pytest -q` suite.

### Phase 5: Global BFF dedup via Dolma library (v0.2: ~5 days, was 5-7)

**Goal**: replace per-source MinHash with global BFF dedup for Stage 3. Uses **Dolma as a Python dependency** — we don't port BFF from scratch.

**Why this is a library call, not a port**: `pip install dolma` ships AI2's Rust `bloom_filter.rs` + `deduper.rs` as a PyO3 extension module. Apache-2.0, actively maintained (v1.2.1 July 2025). The `by_ngram` mode is algorithmically equivalent to BFF. We get an industrial-quality, maintained implementation for the cost of an import.

**Backup source**: if Dolma's Python API doesn't expose the params we need, fall back to vendoring `mlfoundations/dclm/dedup/bff` (MIT, active 2025 commits, same author lineage as the dormant revbucket/bff that v0.1 cited).

**RAM math (v0.2 correction)**:

Standard BFF params (`max_ngram=13`, `fp=0.01`) need 9.59 bits per ngram:
- 1T ngrams → **1.20 TB RAM** for a single global bloom filter
- 600B ngrams → 719 GB
- 251 GB local box → fits ~210B ngrams at fp=0.01 globally

**Decision (v0.2 LOCKED)**: **Option C — rent a 1 TB-RAM box for ONE dedup pass**.
- Cost: ~$30-50 (AWS x2gd.metal or u-3tb1, single pass)
- Tradeoff: cleanest, matches DCLM-Baseline recipe verbatim, fits "ephemeral workers" pattern
- Alternatives rejected: (A) per-snapshot 16-shard dedup — no cross-source dedup, matches FineWeb-by-dump model but loses global quality win; (B) hash-partitioned sequential — fits 251 GB but adds 2-4 days I/O and is custom code on top of Dolma.

**Implementation choices**:
- Drive Dolma's by_ngram mode via Python. Configure: `min_ngram=5`, `max_ngram=13`, `filtering_threshold=0.8`, `fp_rate=0.01`.
- Tune `min_ngram_size` + `expected_ngram_count` on a held-out 5B subset BEFORE the full 1T pass (DCLM issue #71 cautionary tale: default params lost 98% of data for one user).
- Local 5B/20B tuning runs on the dev box; 1T pass on rented 1 TB-RAM box.
- Output: per-document keep/drop flag in a sidecar manifest; original packed shards unchanged.

**Files**:
- `scripts/run_global_dedup.py` (new) — Python driver over Dolma's API; reads R2 sources, calls Dolma, emits sidecar drop list
- `tests/test_global_dedup.py` — equivalence on a small fixture; param-tuning regression test
- `docs/dedup_strategy.md` (new) — document the Dolma choice + DCLM-recipe params + the rented-box procedure

**Acceptance**:
- 5B dedup pass completes locally in <30 min.
- Drop rate within 5% of per-source MinHash on the same corpus (sanity check).
- Sidecar manifest format documented + readable from `compose_mixed_corpus.py`.
- Tuning bench on 5B corpus shows drop-rate plateau within expected range (no DCLM-#71-style data loss).
- 1T pass procedure scripted end-to-end (R2 source pull → rented-box dedup → R2 sidecar upload → box teardown).

**GO trigger** (v0.2.1): start Phase 5 if EITHER:
- Stage 2 1B rehearsal shows per-source `val_loss` spread > **0.15 nats** across sources (signals duplicate-driven memorization on one source), OR
- 5B sample's memorization-probe extraction rate > **1%** (k=50, l=50 protocol from Pythia).

**HOLD otherwise** — don't pull Phase 5 forward unless Stage 3 demonstrably needs cross-source dedup. Per-source MinHash (Phase 1) is the fallback.

### Phase 6: Quality classifiers — DCLM + FineWeb-Edu (v0.2 SPLIT into 6a / 6b)

v0.1 assumed both classifiers were classic fastText. **They're not** — FineWeb-Edu is a transformer (Snowflake-arctic-embed + linear head served via `transformers`). `fasttext-rs` cannot load it. v0.2 splits the work:

#### Phase 6a — DCLM classifier via fasttext-rs (3 days)

**Status**: unchanged from v0.1.

- Source: `mlfoundations/fasttext-oh-eli5` (classic fastText, MIT license).
- Rust crate: `messense/fasttext-rs` confirmed working.
- Per-doc throughput target: ≥2k docs/sec/core.
- Default threshold: ≥0.033 (DCLM-Baseline value).
- Files: `crates/myllm_dataprep/core/src/fasttext_dclm.rs` (~100 LoC), `tests/test_native_fasttext_dclm.py`.

#### Phase 6b — FineWeb-Edu classifier via ONNX (5 days)

**Choice (v0.2 LOCKED)**: **ONNX export + `ort` (Rust) or `candle`** for the real transformer. Community fastText port (`kenhktsui/fineweb-edu-fasttext-classifier`) rejected — 68% accuracy / Spearman 0.58 vs original is a material quality regression on labels 3-5 (the educational-value bucket we actually care about).

**Implementation choices**:
- Export the FineWeb-Edu classifier (Snowflake-arctic-embed encoder + linear head) to ONNX using `transformers.onnx` or `optimum`.
- Load via `ort` (Rust ONNX runtime) inside the filter pipeline, or use `candle` if `ort` has packaging issues.
- Throughput: 50-200 docs/sec/core on CPU (GPU preferred for full-corpus passes; fits ephemeral-GPU-worker pattern).
- Default threshold: ≥3.0 (FineWeb-Edu "high educational value" bucket).
- Quality target: published numbers (no community-port regression).

**Files**:
- `scripts/export_fineweb_edu_to_onnx.py` (new) — one-time export from HF checkpoint to ONNX
- `crates/myllm_dataprep/core/src/fineweb_edu_classifier.rs` (~150 LoC)
- `tests/test_native_fineweb_edu_classifier.py` — equivalence to HF reference on small fixture
- `configs/data/pretrain_mix_pilot.yaml` — `quality_classifier:` section with both 6a + 6b thresholds

**Acceptance (6a + 6b combined)**:
- DCLM scoring at ≥2k docs/sec/core; FineWeb-Edu scoring at ≥50 docs/sec/core (CPU) or ≥500 docs/sec (GPU).
- Drop rate at default thresholds within 10% of published numbers on a CommonCrawl sample.
- License audit: DCLM is MIT (verified). FineWeb-Edu classifier license needs explicit verification before bundling — Apache-2 likely but confirm in Phase 6b kickoff.

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
| D3 | **Use `xxhash-rust` xxh64 + hand-port MinHash** (v0.2: NO rensa) | rensa hardcodes a non-xxh64 multiply-shift hasher → would silently invalidate existing R2 decontam indexes (silent-corruption class bug). Realistic speedup is 20-40× from GIL-escape + native loop, not the 608× rensa README claims. Hand-port keeps bit-compatibility. | rensa (broken for us); xxh3 (5-10× faster but breaks compat); gxhash (no AES-NI fallback) | Medium — Phase 1b can migrate to xxh3 with a one-time corpus rebuild |
| D4 | **Inline decontam in Rust** (not post-hoc batch) | Preserves audit trail per-doc; avoids separate pipeline; simpler operator model | Post-hoc decontam (faster build but worse traceability) | High — Phase 5 could move to post-hoc if perf becomes a blocker |
| D5 | **Adopt published quality classifiers — DCLM (fastText, MIT) + FineWeb-Edu (transformer via ONNX)** — not train our own | Frontier-lab standard practice; 1B teams don't train quality classifiers; lift is below ablation noise. NOTE: FineWeb-Edu is NOT classic fastText — see Phase 6b. | Train our own (10-20× more engineering work, no clear quality win); community fastText port of FineWeb-Edu (rejected — 32% accuracy gap on labels 3-5) | Medium — could train custom later if quality plateaus |
| D6 | **BFF (Bloom-Filter-Fuzzy) global dedup** in Phase 5, sourced from **`mlfoundations/dclm/dedup/bff`** (MIT, active 2025 commits) AND/OR Dolma's `by_ngram` mode as a Python library | DCLM benchmarked all three and picked BFF; SlimPajama scale validation; memory-efficient; faster. v0.2 corrects v0.1's revbucket/bff (frozen 2024-04). Dolma exposes equivalent algorithm as a PyO3 Python library (`pip install dolma`) — see Phase 5 for which path. | MinHash global (in-memory LSH won't fit for 1T); Suffix-Array (won't fit either) | Low — picking BFF locks the algorithm |
| D7 | **DO NOT adopt Datatrove as framework** | 97% Python in hot path; designed for Slurm not single-node; we'd inherit our current bottleneck | Adopt Datatrove + Rust extensions (too much rework) | Medium — could revisit if we ever get a real cluster |
| D8 | **Output equivalence is bit-for-bit for Phases 1-3** | Trust requires it; alternative is silent quality degradation | "Statistically equivalent" (looser; harder to debug regressions) | Low — Phase 5+ explicitly breaks this for algorithm changes |
| D9 | **Repo layout: `crates/myllm_dataprep/` subdir with separate Cargo workspace + Python wrapper** | Same as huggingface/tokenizers, polars; clean separation | Top-level Rust files (clutter); monorepo with python+rust in same Cargo project (build complexity) | Medium — could restructure but it's load-bearing |
| D10 | **Implementer works in this same repo on feature branches, one PR per phase** | Simpler than v0.1's separate-remote idea; CLAUDE_COLLAB doc handles claim coordination; PR-per-phase keeps merges small; `_USE_NATIVE` shim means every phase landing is non-breaking | Separate-remote fork (v0.1) — added remote complexity without coordination benefit; mono-PR at end (too big to review) | High — could fork later if conflict frequency demands |
| D11 | **GIL-build Python + `py.detach()` + rayon inside Rust. Single `abi3-py310` wheel.** Do NOT ship free-threaded (cp313t/cp314t) wheels. (NEW in v0.2) | This is what tokenizers + polars actually do in 2026. Free-threaded 3.13t/3.14t costs 10% single-thread perf + 15-20% memory for zero gain in our pipeline. PyO3 0.28 renamed `Python::allow_threads` → `Python::detach` — 2026 docs only, no 0.27-era posts. | Free-threaded wheels (regression + complexity); subinterpreters (immature in 3.13) | High — could add freethreaded wheels later if PEP 779 Phase 3+ makes them mainstream |

---

## 7. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Wheel matrix complexity (Linux x86_64 + arm64 + macOS + Windows × Python 3.10-3.13 = 16 wheels) | High | Medium | Use abi3-py310 (collapse Python axis to 1); ship linux x86_64 only in Phase 0, add others later |
| R2 | GIL release discipline — forgetting `py.detach()` means zero parallelism gain | Medium | High | Stress test: spawn 8 Python threads calling extension, diff results vs single-thread; explicit linting in code review |
| R3 | Bit-equivalence harder than expected (Python's float / int / hash quirks) | Medium | Medium | Build small "golden" test corpus committed to repo; run before each PR; if quirk found, document + decide whether to match or document deviation |
| R4 | xxh64 vs xxh3 confusion — accidentally using the wrong one breaks index file compatibility | Medium | High | Explicit tests asserting hash output matches known Python value; CI guard |
| R5 | Compile time (1k LoC Rust + PyO3 + regex + hashbrown = ~150 crates, 3-5 min cold compile) | High | Low | sccache in CI; iterate on small testable units; keep binding crate small |
| R6 | Implementer's feature branch diverges too far from main, painful to merge | Medium | Medium | Tight PR-per-phase discipline (don't accrete >1 phase before merge); rebase against main daily during active development; coordinate in CLAUDE_COLLAB.md |
| R7 | BFF bloom filter RAM at 1T scale (1.20 TB needed for global at fp=0.01; our 251 GB box fits ~210B ngrams) | High | High | Phase 5 Option C — rent 1 TB-RAM box ($30-50, AWS x2gd.metal or u-3tb1) for ONE dedup pass. Tune params on local 5-20B subset FIRST to avoid DCLM-#71 failure mode (98% data loss with mis-tuned defaults). |
| R8 | fastText classifier dependencies (binaries from HF) may not be redistributable | Low | Medium | Check FineWeb-Edu + DCLM classifier licenses before Phase 6 (likely Apache-2 / similar but verify) |
| R9 | User direction changes mid-migration (already saw scope shifts on 2026-05-13) | High | Medium | Plan in phases so partial work is still useful; align with reviewer before starting each phase |
| R10 | Reviewer pushback on Dolma as runtime dependency vs vendoring `mlfoundations/dclm/dedup/bff` | Medium | Medium | Plan supports both (D6); decide at Phase 5 kickoff based on Dolma's exposed Python API surface for our params. Vendor DCLM/dedup/bff into `third_party/bff/` if Dolma's API doesn't expose `min_ngram_size` / `expected_ngram_count` / `fp_rate` tuning. |

---

## 8. Timeline + budget

**Total estimate**: 4-6 weeks of focused engineering by the implementer (~110-150 hours).

| Phase | Effort | Wall time (1 eng full-time) |
|---|---|---|
| Phase 0+ (pre-Rust Python wins) | 1-2 days | 1-2 days |
| Phase 0 (Rust scaffolding) | 1-2 days | 1-2 days |
| Phase 1 (MinHash Rust) | 3-4 days | 3-4 days |
| Phase 2 (Decontam Rust) | 2-3 days | 2-3 days |
| Phase 3 (Filter Rust) | 3-4 days | 3-4 days |
| Phase 4 (Integration + E2E bench) | 2-3 days | 2-3 days |
| Phase 5 (Dolma global dedup, Stage 3) | ~5 days | ~5 days |
| Phase 6a (DCLM fastText, Rust) | 3 days | 3 days |
| Phase 6b (FineWeb-Edu ONNX) | ~5 days | ~5 days |
| Phase 7 (docs + handoff) | 1-2 days | 1-2 days |
| **Phases 0+ through 4 (MVP for Stage 2)** | **12-18 days** | **~2.5-3 weeks** |
| **Phases 0+ through 7 (full migration for Stage 3)** | **25-35 days** | **~5-7 weeks** |

**Compute budget**: minimal — all work runs on the existing 128-core dev box. CI runs need GitHub Actions standard runners (free for public repos; small monthly cost for private). No GPU needed.

**Reviewer involvement**: cross-verify this plan (1 hr); review Phase 1 PR (1 hr); review Phase 5 design doc separately (1 hr); review final integration PR (2 hr). Total ~5 hr of reviewer time across the project.

---

## 9. Acceptance criteria for the whole migration

A successful migration is one where, by the end of Phase 7:

1. **Performance**: 5B build wall ≤ 20 min (vs 2.5 hr today). 1T build projected ≤ 5 days total = ~4 days local per-source build (128 cores) + ~1 day rented-box global dedup pass (vs 70-100 days baseline).
2. **Correctness**: bit-for-bit output equivalence on Phases 1-3; documented algorithmic-equivalence on Phase 5; full test suite passes.
3. **Maintainability**: Rust crate is buildable from a fresh clone in <10 min with `pip install -e .` + `pip install -e crates/myllm_dataprep/python/`. Documentation exists at `docs/data_prep_architecture.md`.
4. **Reversibility**: `_USE_NATIVE=False` shim still works for any operator who can't or won't build Rust.
5. **Reviewed**: reviewer signed off on the algorithm choices (especially Phase 5 BFF).

---

## 10. Coordination with Session A (main branch)

Session A continues working on the main branch (Stage 1 pilot, Stage 2 prep, reviewer's P0 items). Implementer works in feature branches in the same repo (D10). Coordination protocol:

- **Branching**: per-phase feature branch `rust/phase-N-<topic>` off `main`. Open the PR early as a draft so Session A can see what's coming.
- **Sync point**: rebase against `main` daily during active development; whenever Session A reports a touched file in CLAUDE_COLLAB that overlaps the implementer's branch, rebase immediately.
- **Conflict-likely files** Session A may touch: `src/myllm/data/build.py`, `dedupe.py`, `decontamination.py`, `filters.py`, `scripts/build_packed_corpus.py`. Session A should AVOID changing these on main during the migration unless absolutely necessary; if they must, document the change in `docs/archive/CLAUDE_COLLAB.md` so the implementer sees it on the next sync.
- **Conflict-unlikely files** Session A is likely to touch: `src/myllm/training/*`, `scripts/run_pretrain.py`, `configs/*`. Implementer should AVOID these.
- **Merge**: at the end of each phase. Session A or user reviews + merges. The `_USE_NATIVE` shim means each phase landing is non-breaking (Python fallback still works).
- **Live doc**: `docs/archive/CLAUDE_COLLAB.md` "Rust migration" subsection should reflect current phase + ETA + blockers.

---

## 11. Open questions (v0.2.1 status)

| # | Question | v0.2.1 status |
|---|---|---|
| 1 | Phase order: BFF (Phase 5) before fastText (Phase 6)? | **Still open** — implementer's call once Phase 0+ through 4 land. BFF more impactful for Stage 3; fastText gives quality wins even at per-source dedup. |
| 2 | Skip Phase 5 entirely if Stage 2 per-source MinHash is enough? | **RESOLVED (v0.2.1)** — Phase 5 GO trigger is concrete: per-source val_loss spread > 0.15 nats OR memorization-probe extraction > 1%. Otherwise HOLD. See Phase 5. |
| 3 | Adopt Datatrove? | **Still open** — v0.2 keeps "no" per v0.1's D7. Implementer to flag if their hands-on view changes this. |
| 4 | gxhash + AES-NI hardware target? | **Still open** — v0.2 keeps xxh64 for index compatibility (D3). gxhash could come in Phase 1b after corpus rebuild if useful. User to confirm hardware AES-NI support for future-proofing. |
| 5 | FineWeb-Edu classifier license? | **Partially resolved** — DCLM is MIT (verified). FineWeb-Edu's HF model page license needs explicit check in Phase 6b kickoff before bundling the ONNX export. |
| 6 | Wheel matrix scope? | **RESOLVED (v0.2)**: ship single `abi3-py310` wheel for linux x86_64 only. Add other platforms post-Phase-4 if there's actual demand. |
| 7 | Which BFF spec? | **RESOLVED (v0.2)**: Dolma's `by_ngram` (Python library) primary; `mlfoundations/dclm/dedup/bff` (MIT, active) as backup if Dolma's Python API is too coarse. |
| 8 | Fork merge model? | **RESOLVED (v0.2.1)**: feature-branches-in-same-repo with PR-per-phase. See D10 + §10. |

---

## 12. References

Sourced from 3 parallel research agents on 2026-05-13 + 5-agent v0.2 cross-verification pass.

### Frameworks + frontier patterns (single-node corpus prep)
- [HuggingFace datatrove](https://github.com/huggingface/datatrove) — *not adopted, see D7*; useful for algorithm reference only (their dedup module is pure Python)
- [AI2 Dolma](https://github.com/allenai/dolma) — Apache-2.0; **Phase 5 primary path** as Python library (`pip install dolma`); see [dedup docs](https://github.com/allenai/dolma/blob/main/docs/deduplication.md), [bloom_filter.rs](https://github.com/allenai/dolma/blob/main/src/bloom_filter.rs), [deduper.rs](https://github.com/allenai/dolma/blob/main/src/deduper.rs)
- [DCLM toolkit](https://github.com/mlfoundations/dclm) — MIT; active 2025 commits
- [DCLM/dedup/bff](https://github.com/mlfoundations/dclm/tree/main/dedup/bff) — **Phase 5 backup path** (vendor if Dolma's Python API doesn't expose tuning knobs)
- [DCLM issue #71 — 98% data loss with bad params](https://github.com/mlfoundations/dclm/issues/71) — Phase 5 tuning cautionary tale
- [DCLM issue #99 — 836GB box at 90% RAM with 2T ngrams](https://github.com/mlfoundations/dclm/issues/99) — informs Option C rented-box decision
- [The FineWeb Datasets (2406.17557)](https://arxiv.org/html/2406.17557v1)
- [Llama 3 paper (2407.21783)](https://ar5iv.labs.arxiv.org/html/2407.21783)
- [DeepSeek-V3 Tech Report (2412.19437)](https://arxiv.org/abs/2412.19437)
- [OLMo 2 paper (2501.00656)](https://arxiv.org/pdf/2501.00656)
- [DataComp-LM benchmark (2406.11794)](https://arxiv.org/abs/2406.11794)

### Algorithms
- [Lee et al. 2021 — Deduplicating Training Data (2107.06499)](https://arxiv.org/abs/2107.06499) — suffix-array dedup; *rejected* (24 TB RAM needed at 1T)
- [google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)
- [LSHBloom paper (2411.04257)](https://arxiv.org/abs/2411.04257) — no public implementation in 2026-05; tracked for future
- [SlimPajama blog](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)
- [revbucket/bff](https://github.com/revbucket/bff) — *rejected, see D6*; frozen 2024-04. Successor work at [revbucket/minhash-rs](https://github.com/revbucket/minhash-rs).
- [Fraunhofer 2024 dedup eval](https://link.springer.com/chapter/10.1007/978-3-031-82481-4_27)

### Rust crates + interop
- [PyO3](https://github.com/PyO3/pyo3) ([0.28.3 parallelism guide](https://pyo3.rs/v0.28.3/parallelism), [0.28.3 free-threading docs](https://pyo3.rs/v0.28.3/free-threading) — for reference; *not* used per D11)
- [PyO3 0.28 migration guide](https://pyo3.rs/v0.28.3/migration) — `allow_threads` → `detach` rename
- [PyO3 CHANGELOG](https://github.com/PyO3/pyo3/blob/main/CHANGELOG.md)
- [Maturin](https://github.com/PyO3/maturin) ([layout guide](https://www.maturin.rs/project_layout.html))
- [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — reference implementation of the PyO3 pattern we're adopting
- [pola-rs/polars](https://github.com/pola-rs/polars) — reference implementation; abi3 wheel pattern
- [beowolx/rensa](https://github.com/beowolx/rensa) — *rejected, see D3* (non-xxh64 hasher; would invalidate existing R2 indexes)
- [xxhash-rust](https://crates.io/crates/xxhash-rust) — **xxh64 locked per D3**
- [hashbrown](https://github.com/rust-lang/hashbrown) — Phase 2 decontam hash set
- [BurntSushi/aho-corasick](https://github.com/BurntSushi/aho-corasick) — Phase 3 PII pattern multi-match
- [BurntSushi/memchr](https://github.com/BurntSushi/memchr) — Phase 3 byte scanning
- [rust regex docs](https://docs.rs/regex/latest/regex/)
- [PyO3/rust-numpy](https://github.com/PyO3/rust-numpy) — Phase 1 zero-copy signature arrays back to Python
- [PEP 703 (free-threading)](https://peps.python.org/pep-0703/) — context only; not in scope per D11

### Classifiers
- [DCLM fastText-oh-eli5](https://huggingface.co/mlfoundations/fasttext-oh-eli5) — MIT; **Phase 6a**
- [messense/fasttext-rs](https://github.com/messense/fasttext-rs) — MIT; Rust binding for Phase 6a
- [FineWeb-Edu classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) — transformer (Snowflake-arctic-embed + linear head); **Phase 6b via ONNX**
- [kenhktsui/fineweb-edu-fasttext-classifier](https://huggingface.co/kenhktsui/fineweb-edu-fasttext-classifier) — *rejected* (32% accuracy gap on labels 3-5)
- [HuggingFace optimum](https://github.com/huggingface/optimum) — ONNX export pipeline for Phase 6b
- [ort (Rust ONNX runtime)](https://github.com/pykeio/ort) — Phase 6b inference; Apache-2 / MIT dual
- [candle](https://github.com/huggingface/candle) — Phase 6b fallback inference engine

---

## END

Reviewer + user: §6 (Locked decisions) and §11 (Open questions) are the ack-required surface. Once Q1 / Q3 / Q4 / Q5 are answered, implementer claims Phase 0+ in `docs/archive/CLAUDE_COLLAB.md` and begins.
