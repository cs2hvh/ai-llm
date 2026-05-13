# MyLLM — ML/AI Interview Question Bank

**Use:** Pick 8-12 questions across sections / difficulty bands for a 90-minute interview. Calibrate downward if the candidate stumbles on §1-2; calibrate upward if they breeze through §3-5. Section 17 (project-specific) only fires if they've read the repo — use as a "did they prep" filter.

**Grading shorthand:**
- ★ **Strong** signal — what a hire-worthy answer mentions.
- ⚠ **Weak / red flag** — surface-level / memorized / wrong.
- 🔍 **Follow-up probe** — to dig past a memorized response.
- Difficulty: **E** (easy / undergrad), **M** (medium / 1-2 yr ML eng), **H** (hard / senior / research-track).

---

## §0 — Calibration (10 min)

**Q0.1 [E]** What's the difference between training and inference for a language model?
- ★ Backward pass + optimizer state + larger memory in training; forward-only with KV cache in inference. Mentions different precision needs (bf16/fp32 master in training, int8/int4 in deployment).
- ⚠ "Inference is just forward pass" (technically true but trivial).

**Q0.2 [E]** Why do we use the negative log of softmax probabilities as the loss instead of just `1 - p(correct_class)`?
- ★ Information-theoretic motivation (entropy / KL); gradient behavior — `-log p` has gradient `(p - target)` which is well-scaled, while `(1 - p)` saturates at the boundaries. Maximum-likelihood interpretation.
- 🔍 "Show me the gradient of softmax+CE." (Test §14.2.)

**Q0.3 [M]** You have a 1B-parameter model. Roughly how much VRAM do you need to train it in bf16 with AdamW?
- ★ ~20 GB minimum: weights (2B) + grads (2B) + fp32 master (4B) + Adam m (4B) + Adam v (4B) = 16 bytes/param × 1B = 16 GB, plus activations + scratch. **Caveats**: assumes no FSDP (sharded would be 5× smaller per device); assumes fp32 m/v (Adam in bf16 m/v is a silent risk).
- ⚠ "About 4 GB" — only counted the weights, forgot opt state.

**Q0.4 [M]** What's MFU and what's a good number for 1B-class training on H100?
- ★ Model FLOPs Utilization = achieved FLOPs / peak FLOPs. Computed as `tok/sec * 6N / peak_bf16_flops`. Healthy 1B on H100/H200 with FSDP + flash + no grad-ckpt: **30-50%**. With grad-ckpt + large vocab + no FSDP: 3-10% (project bottleneck).
- 🔍 "Why not 100%?" (Memory-bound ops, kernel launch overhead, communication for distributed.)

**Q0.5 [M]** What's the difference between data parallelism and FSDP?
- ★ DP replicates model + opt state on every device, shards only the input batch — gradient sync via all-reduce. FSDP shards everything (weights, grads, opt state) along the data axis; uses all-gather (forward) + reduce-scatter (backward). FSDP saves ~5× per-device memory at the cost of more bandwidth.
- ⚠ "FSDP is just DP" or "FSDP shards the batch differently."

---

## §1 — Fundamentals

### 1.1 Tensors and linear algebra

**Q1.1 [E]** What's the difference between a matrix-vector product and a batched matrix-matrix product, computationally?
- ★ Matmul is more arithmetic-intense per byte moved (O(N³) FLOPs / O(N²) memory). On GPUs, matmul saturates tensor cores; mat-vec is memory-bandwidth-bound. Batching matmuls amortizes weight loads. Connects to "why batch size matters for MFU."

**Q1.2 [M]** Walk me through how a matrix-multiply `A @ B` (shapes `[M, K] × [K, N]`) actually executes on a GPU. What's a "tile"?
- ★ GPU tiles the output `[M, N]` into blocks (e.g., 128×128). Each thread-block computes one tile by streaming chunks of A (`[M_tile, K_chunk]`) and B (`[K_chunk, N_tile]`) through shared memory. K-dim is accumulated. Tile size trades off occupancy vs. register pressure.
- 🔍 "Why does this make MFU sensitive to dimension sizes?" (Tiles waste compute on edge fragments when M/N/K aren't multiples of tile size.)

### 1.2 Automatic differentiation

**Q1.3 [E]** What's the difference between forward-mode and reverse-mode autodiff? Which does deep learning use, and why?
- ★ Forward mode: O(N_inputs) cost per output. Reverse: O(N_outputs) cost per input. DL has 1 output (scalar loss) and millions of params → reverse mode (= backprop) wins.

**Q1.4 [M]** What's a "stop_gradient" / "detach" and when would you use it?
- ★ Cuts gradient flow at a node — node's contribution to backward is treated as constant. Use cases: teacher-student (don't backprop through teacher), straight-through estimator for non-diff ops, RL critic-target, momentum encoders (BYOL).
- 🔍 "Could you implement stop_gradient using a custom_vjp?" (Yes — forward returns input, VJP returns zero.)

**Q1.5 [H]** Derive the VJP (vector-Jacobian product) for matrix multiplication: given `y = x @ W`, what's `dL/dx` and `dL/dW` in terms of `dL/dy`?
- ★ `dL/dx = dL/dy @ W^T`; `dL/dW = x^T @ dL/dy`. Show shape consistency. **Bonus**: explain why the two transposes mean parameter-gradient has the same memory cost as forward weight, which drives the optimizer-state memory math.

### 1.3 Probability and information theory

**Q1.6 [E]** Why is entropy in nats vs. bits relevant?
- ★ Cross-entropy loss is in nats (natural log). Perplexity = e^(nat-loss) = 2^(bit-loss). Multiplying CE by `log2(e)` converts. Most papers report nat-CE; some benchmarks report bits-per-character/word.

**Q1.7 [M]** What is KL divergence, and why is it asymmetric? When you use `KL(P || Q)` for distillation, does `P` mean teacher or student?
- ★ `KL(P || Q) = Σ P(x) log(P(x)/Q(x))`. Asymmetric because the support of P matters (zero P kills the term, zero Q blows up). In distillation `KL(teacher || student)` is standard: teacher is "truth", student is approximating distribution. **Mode-covering** ≈ forward KL; **mode-seeking** ≈ reverse KL.
- 🔍 "Why would you use reverse KL instead?" (When the student must be sharper / more decisive than teacher — common in some RL/RLHF setups.)

**Q1.8 [H]** Derive cross-entropy as a special case of KL divergence between two categorical distributions.
- ★ `CE(P, Q) = -Σ P(x) log Q(x) = H(P) + KL(P || Q)`. When P is one-hot (label), H(P)=0, so `CE = KL(P || Q) = -log Q(label)`. Connects to "why CE works without needing label smoothing for hard labels."

---

## §2 — Neural network basics

### 2.1 Layers and activations

**Q2.1 [E]** What's the difference between SwiGLU and GLU? Why has SwiGLU become standard in transformers?
- ★ GLU: `(xW1) ⊙ sigmoid(xW2)`. SwiGLU: `silu(xW1) ⊙ (xW2)` (silu = x·sigmoid(x)). SwiGLU + larger hidden_dim outperforms vanilla FFN + same params (Shazeer 2020 → adopted by Llama, PaLM, etc.). Tradeoff: SwiGLU uses ~1.5× FFN-dim params (3 matrices vs 2 in vanilla FFN), so configs typically size down ffn_dim to keep total FLOPs comparable.

**Q2.2 [M]** Why does ReLU "die"? What's the fix in practice?
- ★ Negative inputs produce zero gradient → if a neuron's pre-activation is consistently negative, its gradient is always zero → its weights never update → "dead" forever. Fixes: LeakyReLU (small slope on neg side), GELU/SiLU (smooth at 0), proper init scale to keep pre-activations balanced.

### 2.2 Normalization

**Q2.3 [E]** Compare LayerNorm and RMSNorm. Why has RMSNorm become more common in LLMs?
- ★ LayerNorm: `(x - mean) / sqrt(var + eps) * gain + bias`. RMSNorm: `x / sqrt(mean(x²) + eps) * gain` (no centering, no bias). RMSNorm saves ~7-15% compute, has fewer params, no measurable quality regression at LLM scale (Zhang & Sennrich 2019; adopted by Llama, PaLM).
- 🔍 "Show that LayerNorm is invariant to scale-and-shift of the input." (See §14.4.)

**Q2.4 [H]** What's QK-norm and why was it added to recent LLMs (e.g., Llama 3)? When is it harmful?
- ★ Apply RMSNorm to Q and K separately (per-head) before the attention dot-product. Bounds the magnitude of QK^T, preventing softmax saturation at long context. Stabilizes training at long context (8K+) and large model widths. Cost: +0.3% FLOPs. Risk: with muP, must be applied consistently across the HP-transfer chain (wind-tunnel → pilot → base) — flipping it mid-chain breaks LR transferability by 10-30%.

### 2.3 Initialization

**Q2.5 [M]** What's the intuition behind Xavier vs. He initialization? When does scaled-init-for-residuals come in?
- ★ Init must keep activations and gradients at unit variance across layers — diverge or vanish otherwise. Xavier: `Var(W) = 2/(fan_in + fan_out)` for tanh/sigmoid. He: `Var(W) = 2/fan_in` for ReLU (compensates for the half-cut). Residual scaling: deep residual networks accumulate variance through residual streams; scaled-init reduces the FFN/attention output projection variance by `1/sqrt(2*num_layers)` to keep residual-stream variance unit at init.

**Q2.6 [H]** What is muP (Maximal Update Parameterization)? Why does it enable HP transfer?
- ★ Yang & Hu 2020: at standard init, an N-width model needs Adam-LR scaled by `1/N` to keep update-magnitude-per-step constant. muP fixes the parameterization (init scale, multipliers per layer type) so that the optimal LR is **width-invariant**. Practical use: tune HPs at a small "wind-tunnel" width (e.g., 384), transfer zero-shot to a larger target (e.g., 2048). Embed and LM-head get specific multipliers; attention output, FFN output, and residual paths each get specific scalings.
- 🔍 "What breaks muP transfer?" (Inconsistent QK-norm, scaled-init mismatch, different optimizer, different vocab.)

### 2.4 Loss functions

**Q2.7 [E]** Why use cross-entropy instead of MSE for classification?
- ★ MSE on softmax probabilities has shallow gradients near "right answer" — saturates. CE's gradient `(softmax_out - one_hot)` is well-scaled regardless of how confident the model already is. Information-theoretic interpretation: minimizing CE = maximizing likelihood = matching teacher distribution.

**Q2.8 [M]** What is label smoothing? When does it help, when does it hurt?
- ★ Replace one-hot label with `(1-α) * one_hot + α / V`. Discourages overconfidence; improves calibration on small models. **Hurts** distillation (the teacher logit is already a soft target — double-smoothing collapses it). **Hurts** at large scale where overfitting isn't the issue.

**Q2.9 [H]** Derive the gradient of softmax-cross-entropy w.r.t. logits.
- ★ See §14.2. Show that for a one-hot target, `dL/dz_i = softmax(z)_i - y_i`. Beautifully clean. Key insight: this is why the standard "logits → softmax → CE" path doesn't need backward through the softmax explicitly — the gradient simplifies.

---

## §3 — Transformer architecture

### 3.1 Self-attention

**Q3.1 [E]** Write out the scaled dot-product attention formula. Why scale by `sqrt(d_head)`?
- ★ `Attention(Q, K, V) = softmax(QK^T / sqrt(d_head)) @ V`. Without scaling: as `d_head` grows, dot-products grow as ~`sqrt(d_head)`, pushing softmax into saturation (one peak, ~0 elsewhere) → vanishing gradient. The `sqrt(d_head)` scaling keeps variance-preserved.

**Q3.2 [M]** Walk me through the memory cost of standard attention at sequence length S. Why is it `O(S²)` and what does that cost at S=8192?
- ★ Attention matrix is `[B, H_heads, S, S]`. At B=1, H=32, S=8192, bf16: `32 * 8192² * 2 = 4.3 GB per layer`. With 16 layers + backward, you'd need ~140 GB just for attention scores. Flash attention fixes this by tile-streaming softmax + matmul (`O(S)` memory).
- 🔍 "How does flash attention preserve the math while avoiding materialization?" (Online softmax + recompute-during-backward.)

### 3.2 Multi-head / GQA / MQA

**Q3.3 [E]** Why do we have multiple heads instead of one big attention?
- ★ Different heads attend to different patterns (syntactic, positional, semantic). Heads can be interpreted as a soft mixture-of-experts over relationships. Width gives you capacity; multi-head gives you specialization.

**Q3.4 [M]** What's GQA and why is it everywhere now?
- ★ Grouped-Query Attention: keep N query heads but use only N/G key-value heads (groups of G queries share a KV head). Cuts KV cache by G× with minimal quality loss. Llama 3: 32 query / 8 KV (4:1). At inference time, KV cache is the dominant memory cost for large context — GQA makes serving cheaper without retraining-time compromise.
- ⚠ "GQA is faster to train" — not really; the training-time win is small. Inference is the win.

### 3.3 Position encodings

**Q3.5 [M]** Compare sinusoidal, learned, ALiBi, and RoPE position encodings. Which would you pick for a new 1B model and why?
- ★ Sinusoidal: position-additive in input embedding; doesn't extrapolate well. Learned: same problem, plus a hard limit at max-trained length. ALiBi: linear distance penalty in attention logits — extrapolates well but limited expressive range. RoPE: rotation in feature space — extrapolates moderately, plays well with attention's dot-product, current standard (Llama, Mistral, Gemma). Pick: RoPE with a high base (130000+) for long-context-friendliness.

**Q3.6 [H]** RoPE applies a rotation `R(θ_pos)` to Q and K. Why does `<Q, K> = <R(m)q, R(n)k> = <q, R(n-m)k>` — i.e., why is the dot-product translation-invariant?
- ★ Rotation matrices are orthogonal: `R^T R = I`. So `<R(m)q, R(n)k> = q^T R(m)^T R(n) k = q^T R(n-m) k = <q, R(n-m)k>`. This is the magic: the absolute positions m and n disappear, only the relative offset `n-m` shows up. Lets the model generalize across absolute position shifts.

### 3.4 Masking

**Q3.7 [E]** What's a causal mask? Show the matrix.
- ★ Lower-triangular mask added to attention logits (zeros below diagonal, `-inf` above). Causal because position `t` can only attend to positions `0..t`. Without it, training/inference distribution mismatch — model would cheat at train time.

**Q3.8 [H]** You're packing multiple short documents into one sequence to improve throughput. How do you mask the attention?
- ★ Need a "segment mask" in addition to causal: each position can only attend to positions within the same document AND ≤ its own. Combined mask: `(same_segment) & (j ≤ i)`. Without segment masking, doc A "leaks" context into doc B's predictions — silent training corruption. **Bonus**: also mask the LOSS so cross-document boundary predictions don't contribute.

### 3.5 Flash attention

**Q3.9 [H]** What does flash attention compute that's mathematically equivalent to standard attention, and how does it save memory?
- ★ Same `softmax(QK^T/√d) V` semantically. Trick: stream over Q-blocks; for each Q-block, iterate over K/V blocks computing partial softmax with running max + running denominator (online softmax). Never materialize `[S, S]`. Backward recomputes `[S, S]` per Q-block tile rather than storing it. Net memory: O(S * d) instead of O(S²). FLOPs: same. Wall-clock: faster due to less HBM traffic (compute-bound vs memory-bound).
- 🔍 "Why does flash attention show up as faster even though FLOPs are identical?" (Memory-bandwidth-bound attention is now compute-bound; better tensor-core utilization.)

### 3.6 KV cache

**Q3.10 [M]** Why is KV caching valuable at inference?
- ★ During autoregressive generation, each new token's Q only needs to attend to past K, V — which were already computed. Storing K, V across the generated tokens avoids recomputing the entire prefix. Cost: `B * S_gen * 2 * H_kv * d_head * dtype_size` per layer. Dominates memory for large context inference.

---

## §4 — Tokenization

**Q4.1 [E]** BPE vs Unigram vs WordPiece — what's the difference?
- ★ BPE: greedy merge-most-frequent-pair, deterministic encoding. Unigram: probabilistic — multiple ways to tokenize a string, pick highest-prob path. WordPiece: like BPE but uses likelihood instead of frequency for merging. Practical: BPE is GPT/Llama default; Unigram is SentencePiece default; WordPiece is BERT.

**Q4.2 [M]** What is "byte fallback" and why does it matter for multilingual tokenizers?
- ★ Every byte (0-255) is in the vocab. Any UTF-8 string can always be encoded (no OOV). Critical for non-Latin scripts: avoids the unicode-character-becomes-UNK failure mode of naive vocab. Cost: rare-script tokens are byte-fragmented (high fertility) — partially mitigated by adding script-aware merges during training.

**Q4.3 [M]** What's "fertility" and how does it affect training cost?
- ★ Average tokens per "intended unit" (character / word). For English Llama-tokenizer: ~1.3 tokens/word. For Hindi-Devanagari: 3-5 tokens/word (much worse). Higher fertility = more compute per text unit = less effective training. Multilingual models trade some English fertility for less-bad non-Latin fertility via larger vocab.

**Q4.4 [H]** When would you pick a 131K vocab over a 32K vocab? When would you pick 32K?
- ★ 131K: serious multilingual coverage (Llama 3.2 is 128K, Qwen 2.5 is 151K). Helps non-English fertility. Cost: ~100M embedding params (40% of a 250M model). Hurts MFU when LM head dominates compute. 32K: pure English/code; smaller models where embed overhead matters; faster sampling. Decision is downstream-deployment-driven: who uses this model?

---

## §5 — Pretraining objectives

**Q5.1 [E]** Why is next-token prediction the canonical pretraining objective for generative LMs?
- ★ Forces the model to learn the joint distribution of language by chain-rule factorization: `P(x₁...xₙ) = Π P(xᵢ | x_{<i})`. Trivially evaluable (no special heads). Generates coherent continuations zero-shot. Compare to BERT-style MLM: gives strong embeddings but harder to use generatively, no autoregressive generation at inference.

**Q5.2 [M]** Show me the chunked cross-entropy idea. Why is it needed for 131K vocab + seq=8192?
- ★ Standard CE materializes `[B, S, V]` logits and `[B, S, V]` one-hot (or log_softmax). At B=8, S=4097, V=131072 in bf16: 8.6 GB per tensor. Chunked CE: split V into chunks, run matmul `hidden @ embed_chunk^T → [B, S, V/N]`. Accumulate global log-sum-exp via online stable trick. Gather target logit from the chunk containing the label index. Peak memory drops by N×. Need `jax.checkpoint` wrap to also save backward memory.

**Q5.3 [H]** What is z-loss and why did PaLM add it?
- ★ Z-loss = `λ * (log-sum-exp(logits))²` — penalizes the logit-norm. Without it, on long-pretrain runs at scale, the log-partition function drifts toward `±∞`, training loss looks fine but bf16 numerics fail. PaLM 2022 reports +1-2% downstream gain from adding z-loss with `λ=1e-4`. Now standard in production runs.

---

## §6 — Distillation

**Q6.1 [E]** What's the difference between hard distillation and soft distillation?
- ★ Hard: teacher's argmax prediction becomes a label for student CE. Soft: teacher's full softmax (or top-K) gives a target distribution for student KL. Soft transfers more information per sample because it conveys the teacher's uncertainty.

**Q6.2 [M]** What's the role of temperature `T` in distillation? What changes for student loss?
- ★ Both teacher and student logits are divided by T before softmax. Higher T → softer (more uniform) distributions → emphasizes the relative rankings of less-likely classes. The student-side gradient through `softmax(z/T)` is scaled by `1/T²`, so the distillation loss term is typically multiplied by `T²` to keep magnitude calibrated. T=1 most common; T>1 used when teacher is over-confident.

**Q6.3 [M]** Why use top-K logit truncation instead of full vocab in distillation? What's lost?
- ★ Storage + bandwidth: at V=131K, full teacher logits are ~256KB per token. Top-K (e.g., K=64) is ~256 bytes per token — 1000× less. Loss: the "mass outside top-K" is dropped, which is fine if teacher concentrates >99% mass on top-K. For code/math (sharply peaked), K=8-64 works. For free-form text (heavy tail), K=64-256.

**Q6.4 [H]** "Decay-phase-only distillation" — what does it mean and why use it?
- ★ Train the student normally (pure CE) for the first 85% of tokens; in the last 15% (the WSD decay phase), inject teacher logits and train on a mixture of CE + KL. Rationale: (a) saves teacher inference cost during the easy phase, (b) decay-phase distillation aligns the student's *converged* distribution with the teacher, (c) early-training distillation can over-constrain a student that hasn't yet learned representations. Effective-token multiplier: ~1.15-1.2× (vs ~2-3× for full-throughout distillation like Llama 3.2).

---

## §7 — Optimization

**Q7.1 [E]** SGD vs Adam — when would you pick each?
- ★ SGD: simpler, less memory, better generalization on classification (CV literature). Adam: adaptive per-param LR; handles wildly varied gradient magnitudes; standard for LLMs / RNNs / sparse problems. AdamW = Adam with decoupled weight decay (the standard now). At LLM scale, SGD just doesn't work — Adam's per-param scaling is essential when params have hugely different gradient scales.

**Q7.2 [M]** What's the difference between Adam and AdamW?
- ★ Adam: weight decay is added to the gradient, which gets normalized by `sqrt(v)` — so the effective decay is gradient-magnitude-dependent. AdamW: weight decay is applied directly to params (decoupled from the optimizer update). Better generalization (Loshchilov & Hutter 2017). All modern LLMs use AdamW.

**Q7.3 [M]** Why do we need fp32 for AdamW's `m` and `v` moments? What goes wrong if they're bf16?
- ★ `v = β₂ v + (1-β₂) g²`. At β₂=0.95, with small grads, the `(1-β₂)g²` term can be 10⁻⁸ or smaller. bf16 has only ~3 decimal digits of precision — small-g² updates underflow to zero, breaking the second moment. Production: pin `mu_dtype=fp32` (and ideally `nu_dtype=fp32` too).

**Q7.4 [E]** What does gradient clipping do? Global-norm vs per-param.
- ★ Global-norm: scale all gradients by `min(1, max_norm / ||g||)` based on total grad norm. Per-param: clip each param's grad independently. Global-norm preserves gradient direction (just shortens magnitude); per-param distorts direction. **Global-norm is the standard** for transformers (typical max_norm = 1.0).

**Q7.5 [M]** Explain WSD (Warmup-Stable-Decay). Why has it replaced cosine for production LLMs?
- ★ WSD: linear warmup → constant peak LR → cosine/linear decay in the last N% (typically 10-20%). Advantages over cosine: (a) you can stop at any point in the stable phase and run a partial decay → flexible budget; (b) WSM (checkpoint averaging at end of stable) gives a free quality bump; (c) easier to extend training without resetting the LR curve. Adopted by SmolLM2, MiniCPM, OLMo 2.

**Q7.6 [H]** A "loss spike" happens at step 50K. Walk me through your spike-recovery protocol.
- ★ Detect: monitor running loss + variance; trigger on `loss > μ + Kσ` with hysteresis. Recover: (a) save a marker checkpoint at spike step for forensics, (b) restore from most-recent good checkpoint (typically 1K-5K steps back), (c) halve the LR multiplier (compounds if it happens multiple times), (d) skip K batches of training data to avoid replaying the toxic data, (e) reset the spike detector to avoid phantom re-fires. Production: cap recovery count (e.g., 3); if exhausted, fail loud.

---

## §8 — Mixed precision

**Q8.1 [E]** fp32, bf16, fp16 — what's the bit layout difference?
- ★ fp32: 1 sign + 8 exponent + 23 mantissa. bf16: 1+8+7 (same exponent range as fp32, less precision). fp16: 1+5+10 (less range, more precision). bf16 is dynamic-range-stable for DL (no loss scaling); fp16 needs loss scaling to avoid underflow.

**Q8.2 [M]** What's the "master weights" pattern in mixed precision?
- ★ Forward + backward in bf16/fp16 for compute speed and memory. **Master weights kept in fp32**. Optimizer reads master, applies update in fp32, then casts back to bf16 for next forward. Without master weights, small accumulated updates round to zero in bf16. Cost: +4 bytes per param (4× weight memory). Pretty much mandatory at LLM scale.

**Q8.3 [M]** When does loss scaling matter? Why isn't it needed for bf16?
- ★ fp16 has limited dynamic range (~6e-8 to 65504). Gradients near 0 underflow to 0 — silent zero updates. Loss scaling: multiply loss by 2^N before backward (gradients get scaled too), then divide back at optimizer step. Adapts based on overflow detection. bf16 has fp32's exponent range — underflow at typical training gradient magnitudes is rare → no scaling needed. **Why bf16 dominates LLM training**: simpler than fp16, less brittle.

---

## §9 — Distributed training

**Q9.1 [E]** What are the four standard distributed-training parallelism strategies?
- ★ Data parallel (DP): same model on every device, different batch. Tensor parallel (TP): split a single matmul across devices. Pipeline parallel (PP): different model layers on different devices. Sequence parallel (SP): split sequence dim across devices. Practical large-LM stacks combine 2-4: e.g., DP × FSDP × TP (Llama 3 uses 4D).

**Q9.2 [M]** ZeRO-1, ZeRO-2, ZeRO-3 — what does each shard?
- ★ Z1: only optimizer state sharded; weights + grads replicated. Z2: + gradients sharded. Z3: + weights sharded (full FSDP). Memory savings: Z3 cuts per-device state to `1/N`. Communication cost: Z3 needs all-gather at forward + reduce-scatter at backward (extra bandwidth) but it's reduce-scatter not all-reduce (similar volume to DDP).

**Q9.3 [M]** Define the four standard collective ops: all-reduce, reduce-scatter, all-gather, all-to-all.
- ★ All-reduce: every device gets the SUM of all devices' tensor. Reduce-scatter: each device gets a 1/N SHARD of the sum. All-gather: every device collects all other devices' shards into the full tensor. All-to-all: each device sends a different chunk to each other device. Math: `all-reduce = reduce-scatter + all-gather` (cost-wise, all-reduce ≈ 2× either of its constituents).

**Q9.4 [H]** Why does FSDP use reduce-scatter on backward, while DDP uses all-reduce?
- ★ DDP: all devices have the full gradient → need to sum → all-reduce makes everyone agree. FSDP: each device only OWNS a shard of params → only needs its shard of gradients → reduce-scatter produces exactly the per-device shard. Bandwidth: same total bytes moved, but reduce-scatter leaves the result already-sharded (no separate scatter needed). **Subtle bug**: if `with_sharding_constraint` is missing on grads, XLA emits all-reduce → "DDP-shape collectives with FSDP-shape memory" → silently 2× slower.

**Q9.5 [M]** What's MFU and how do you compute it?
- ★ MFU = achieved-FLOPs / peak-FLOPs. Achieved-FLOPs ≈ `6 * N_params * tokens/sec` (forward + backward + update = 6 FLOPs/param/token, Chinchilla approximation). Peak: from hardware spec (H100 SXM = 989 TFLOPS BF16 dense; H200 same compute, more HBM). Healthy production: 30-50%. Below 10%: structural problem (replicated state, large vocab, grad-ckpt + tiny batch).

**Q9.6 [H]** Inter-pod bandwidth is ~10 Gbps (default RunPod); intra-pod NVLink is ~600 GB/s. What does that mean for your parallelism strategy choice?
- ★ ~500× bandwidth gap between intra-node and inter-node. Pure DP across pods: gradient all-reduce volume scales with model size; at 1B that's ~8 GB/step → 6.4 s/step just for sync at 10 Gbps. Same model intra-pod: ~10 ms. **Cross-pod DP only works** with InfiniBand (100+ Gbps); otherwise use pipeline parallel (only activations cross pods, much smaller) or local-SGD-style infrequent sync.

---

## §10 — Scaling laws

**Q10.1 [E]** Kaplan 2020 vs Chinchilla 2022 — what changed?
- ★ Kaplan: data-not-the-bottleneck, scale params aggressively. Chinchilla: re-run the experiments with proper LR tuning → optimal is ~20 tokens per param (much more data, smaller model than Kaplan implied). Set the modern "Chinchilla-optimal" frame. Most production models now over-train past Chinchilla (Llama 3 is ~225× over-Chinchilla) because inference cost > training cost in production.

**Q10.2 [M]** Why over-train a 1B model on 9T tokens (Llama 3.2) when Chinchilla says 20B is optimal?
- ★ Chinchilla optimizes training-FLOPs-vs-loss. Production cares about loss-per-inference-cost. Over-training keeps the small model size (fast inference) while pushing loss curve to where a Chinchilla-optimal LARGER model would be. Tradeoff: 100B → 300B tokens nets ~3 MMLU points; 300B → 1T nets ~4 more. Returns diminish but are real.

**Q10.3 [M]** When does muP work and when does it break?
- ★ Works: zero-shot LR/init transfer from a small width to a large width, IF the architecture is held constant (same depth, same QK-norm, same activation, same vocab, same scaled-init policy). Breaks: changing depth (muP only covers width), changing optimizer family (e.g., SGD ↔ Adam), changing precision/normalization mid-chain. Practical: validate at one intermediate width before launching base.

---

## §11 — Data

**Q11.1 [E]** Why do production pretraining pipelines apply filters before tokenization?
- ★ Cheap filters (length, language ID, symbol ratio) remove junk first → less tokenization work. Order: stream → text-level filter → tokenize → pack. Decontamination is text-level (before tokenization) so n-gram match works on words, not tokens.

**Q11.2 [M]** Explain MinHash + LSH for deduplication. What are the false-positive vs false-negative tradeoffs?
- ★ MinHash: hash document into a fixed-length signature (typically 128 perms) such that Jaccard similarity is approximated by hash agreement rate. LSH bands: split signature into B bands of R rows; documents collide if any band matches → finds near-duplicates without all-pairs comparison. Tradeoffs: higher Jaccard threshold (e.g., 0.75) → fewer false positives, more near-duplicates pass through. More bands → higher recall, more false positives.

**Q11.3 [M]** Why is deduplication critical for pretraining quality?
- ★ Repeated documents: (a) inflate likelihood of duplicated content via memorization, (b) waste tokens on already-learned info, (c) bias the model toward whatever's frequently duplicated (often spam / boilerplate). CommonCrawl has 40-80% near-duplicates raw; quality pipelines (FineWeb, DCLM) dedupe aggressively.

**Q11.4 [H]** Compare 8-gram vs 13-gram for benchmark decontamination. Which would you pick?
- ★ 8-gram: high recall (catches more paraphrases) but false positives (common n-grams trigger). Llama 3 uses 8-gram with normalization. 13-gram: high precision (rarely a coincidence) but misses paraphrased benchmark content. Best practice (per reviewer): dual-mode reporting — 8-gram as primary removal, 13-gram as secondary precision-check.

**Q11.5 [M]** What's "packed sequence training" and why is segment-aware masking needed?
- ★ Concatenate multiple short documents into a single training sequence (length S) separated by EOS tokens. Without segment masking: doc B's first tokens attend to doc A's content → cross-document leak. Fix: track segment IDs per token; attention mask combines causal + same-segment. Also: mask the loss at document boundaries — predicting "doc B token 1 from doc A token N" is meaningless.

---

## §12 — Evaluation

**Q12.1 [E]** What does perplexity actually measure?
- ★ `PPL = exp(CE)` — exponential of the average cross-entropy. Interpretable as "the model is as uncertain as if it had this many equally-likely options at each step." Lower is better. Tied directly to log-likelihood; the model's training objective IS minimizing per-token CE.

**Q12.2 [M]** Why does PPL on held-out test correlate poorly with downstream benchmark scores at 1B+ scale?
- ★ At large scale, two models with the same PPL on a fixed test set can have vastly different MMLU. PPL is averaged over all tokens equally; benchmark performance lives in a narrow distribution (correct-answer tokens). A model can have great average PPL but be terrible at the tokens that benchmark scoring cares about. Direct benchmark eval is required for production decisions.

**Q12.3 [M]** What's "memorization probe" and why run it during a long pretrain?
- ★ Sample text snippets from the training corpus; prompt the model with the first half and check whether it regurgitates the second half verbatim. As tokens-per-param grows, memorization grows. Production: run probes at ~25% checkpoints; if verbatim recall is rising sharply, you're over-training into rote memorization (and risk PII leak / copyright issues).

---

## §13 — Engineering / production

**Q13.1 [E]** How does atomic checkpoint save prevent partial-write bugs?
- ★ Write to `tmp/step-NNNN/`, then atomic rename to `step-NNNN/`. Manifest.json is written LAST as the completion marker. Reader checks manifest exists → trusts the directory. Crash mid-write leaves a `tmp/` that gets cleaned up next run, no broken-checkpoint corruption.

**Q13.2 [M]** What state MUST be saved to restore bitwise-exact training? Why "must"?
- ★ Trainable params, non-trainable buffers, optimizer state (m, v, count), step counter, LR-recovery multiplier (if using watchdog), data-cursor / position. Each one matters: missing optimizer state restarts from zero-momentum → first few steps diverge. Missing step → bias correction uses wrong denominator. Missing data position → re-seeing already-trained data → drift.

**Q13.3 [H]** Walk me through the BLOOM-tr11 bug class. How would you guard against it?
- ★ BLOOM-tr11: tied input/output embedding gradients weren't being summed across data-parallel devices, so each replica updated only its local view of the embedding. Result: silent divergence between replicas → training collapsed. Guard: dedicated canary that runs the same step on 1 device vs N devices with the same seed/data and asserts identical loss curves. This is L2 parity — the do-not-pass-go check before any production run.

**Q13.4 [M]** What's atomic NaN-revert and why is just-zeroing-the-gradient not enough?
- ★ When loss/grad is NaN, zeroing the grad lets the optimizer step still run — but AdamW's decoupled weight decay applies `lr * wd * params` regardless of grad, AND the optimizer's step counter (used for bias correction) advances. So a "skipped" batch still drifts params + advances bias-correction. Fix: compute candidate-new-state, then `jnp.where(loss_is_finite, candidate, old)` on every leaf. No half-update gets through.

---

## §14 — Math derivations (whiteboard)

### 14.1 Softmax gradient

**Q14.1 [M]** Derive `dsoftmax_i / dz_j` for softmax `p_i = exp(z_i) / Σ exp(z_k)`.
- ★ Case i=j: `p_i (1 - p_i)`. Case i≠j: `-p_i p_j`. In matrix form: `J = diag(p) - p p^T`. Show the derivation: write `p_i = e_i / S` where `S = Σ e_k`; apply quotient rule.

### 14.2 Softmax cross-entropy gradient

**Q14.2 [M]** Given loss `L = -log(p_y)` where `p = softmax(z)`, derive `dL/dz_i`.
- ★ `dL/dz_i = p_i - 1{i=y}` = softmax minus one-hot label. Beautifully simple. Show how this collapses two chain-rule steps into one (CE's `-1/p_y` cancels softmax's `p_y(1-p_y)` for i=y).

### 14.3 Log-sum-exp stability

**Q14.3 [E]** Why is `logsumexp(z) = max(z) + log(Σ exp(z - max(z)))` more numerically stable than the direct formula?
- ★ Direct `log(Σ exp(z))` overflows when z is large (~1000 → exp overflows fp32 max 1e38). Trick: subtract max(z) → exponents are all ≤ 0 → no overflow → add max back. Cost: one extra pass + subtract.

### 14.4 LayerNorm invariance

**Q14.4 [H]** Show that LayerNorm `LN(x) = (x - μ) / σ * γ + β` is invariant to a scale-and-shift `x → a*x + b` of the input (when γ, β are held fixed).
- ★ New μ' = a*μ + b. New σ' = a*σ. So `(a*x + b - a*μ - b) / (a*σ) = a(x-μ) / (a*σ) = (x-μ)/σ`. The `a` and `b` cancel out. **Consequence**: LayerNorm makes activations scale-shift-invariant. Any "amplitude" info upstream is destroyed by LayerNorm; the network can only use the residual direction.

### 14.5 Tied embedding gradient

**Q14.5 [H]** If input and output embeddings are tied (`E_in = E_out = E`), and the forward uses E in two places, show that the gradient must be summed from both paths.
- ★ Chain rule: `dL/dE = dL/dE|input + dL/dE|output`. If the implementation only counts one path, the embedding gets the wrong gradient. The BLOOM-tr11 bug was exactly this: the data-parallel framework treated `E_in` and `E_out` as separate params, only summed `E_in`'s gradient cross-device, and the `E_out` gradient was per-replica → silent corruption. Test: gradcheck on the tied path.

### 14.6 KL divergence

**Q14.6 [M]** Derive `KL(P || Q)` for two categorical distributions over the same vocab. Show it's non-negative.
- ★ `KL = Σ P(x) log(P(x)/Q(x))`. Non-negativity via Jensen's inequality on `-log` (convex): `KL = E_P[log(P/Q)] ≥ -log(E_P[Q/P]) = -log(Σ Q(x)) = 0`.

### 14.7 Adam bias correction

**Q14.7 [H]** Show that Adam's bias correction `m_hat = m / (1 - β₁^t)` is needed at small t.
- ★ At t=1 with m_0 = 0: `m_1 = (1-β₁) g_1`. So m_1 is biased toward zero by factor `(1-β₁)`. Bias-correction divides by `(1-β₁)` → recovers `m_hat_1 = g_1`. At large t, `β₁^t → 0` so the correction becomes negligible. Critical: at warmup steps 1-100, missing bias correction makes the first updates tiny → drag on early loss.

---

## §15 — Debugging scenarios (oral case studies)

**Q15.1 [H]** Your loss went from 4.2 to 17 over 50 steps and then NaN'd. Walk me through your debug protocol.
- ★ Strong answer mentions:
  1. Check the last good checkpoint's loss curve — was the spike sudden or gradual?
  2. Inspect quarantined batches (NaN-revert should have logged the offending data) for poisoned content
  3. Check optimizer state: did `v` (Adam second moment) blow up?
  4. Check gradient norm history — sustained `||g|| → ∞` indicates the model is in a saturation regime
  5. Verify LR schedule: a sudden LR jump?
  6. Check for data corruption — was the corpus rebuilt between runs?
  7. Verify mixed-precision: any param NaN in fp32 master? bf16 working copy?
  8. If all checks are clean, the model architecture may have a latent instability (missing QK-norm, missing z-loss, etc.)
- ⚠ "Restart from earlier checkpoint and hope it doesn't happen again." (Doesn't address root cause.)

**Q15.2 [H]** Same training script, same data, same seed — first run converges, second diverges. What's different?
- ★ Mostly determinism gaps:
  - JAX/XLA non-determinism in reductions (set `jax_default_prng_impl` + `XLA_FLAGS=--xla_gpu_deterministic_ops`)
  - cuDNN heuristic selection (different convolution algos per run)
  - Atomic adds in distributed reductions (order varies)
  - Data loader shuffle / num_workers
  - Optimizer state load mismatched between runs (one has resumed state, one fresh)
- 🔍 "How would you bisect to find the cause?" (Diff first N step losses; the first divergence step tells you where determinism broke.)

**Q15.3 [H]** Your 1B model on 8×H100 hits OOM at micro_batch=4 but Llama 3 1B fits at mb=16. What's wrong?
- ★ Likely causes:
  - No FSDP (replicated state eats 5× more memory)
  - Vocab is much larger (131K vs 32K) → embedding/LM-head balloon
  - No gradient checkpointing
  - No flash attention (seq=8192 attention scores are huge)
  - bf16 master weights instead of fp32 might paradoxically save memory but break training
  - Logits materialization (no chunked CE)
- 🔍 "If you had to pick the top-1 fix, which one?" (Probably FSDP — biggest leverage.)

**Q15.4 [M]** Eval score on MMLU went up by 4 points but downstream code-gen got worse. Hypotheses?
- ★ (1) Training mix shifted toward general web text, away from code. (2) Decontamination removed code benchmarks but the training data was unaltered, so improvements are real on MMLU but code corpus was thinner. (3) Tokenizer change increased code fertility. (4) Distillation teachers don't include code-strong models. (5) Hyperparam change (LR / WD) affects code specifically (longer-tail distribution).

**Q15.5 [H]** FSDP claims to be running but throughput is the same as DDP. How do you check?
- ★ Compile train_step, dump HLO via `jit_fn.lower(*args).compile().as_text()`. Grep for `reduce-scatter` (FSDP-correct) vs `all-reduce` (DDP-shape). Also check `jax_log_donation=True` to see if `donate_argnums` is being silently disabled. Confirm per-device memory is `1/N` of replicated — if same as DDP, params aren't actually sharded.

**Q15.6 [H]** Checkpoint resume produces a different loss curve than the uninterrupted run. Where do you look?
- ★ (1) Optimizer state restore: namedtuple flattened to dict? (2) Step counter restored or reset to 0? (3) Data cursor: `data_position` saved? (4) LR schedule: re-computed from step or restored? (5) Watchdog state: `lr_recovery_multiplier` restored? (6) RNG state: PRNG keys for synthetic data / dropout / etc. (7) Mixed-precision: master weights restored from bf16 only? **Project canary**: L3 catches this — bitwise-exact resume test on 1B real-corpus path.

---

## §16 — Open-ended judgment questions

**Q16.1 [M]** $50K, 30 days, 5×H200 pod. What 1B model would you commit to training, on how many tokens?
- ★ Honest answer accounting for: (a) measured MFU (likely 5-15% at first), (b) FSDP overhead (need to implement + validate), (c) staged plan (pilot → rehearsal → base) eats 20-30% of budget, (d) buffer for setbacks. Realistic: 1B at seq=4096, ~300-500B tokens. Trade context length down to fit, not model size.
- ⚠ "1T tokens, 8192 context, full base run." (Ignored measured throughput.)

**Q16.2 [M]** When would you NOT use distillation?
- ★ (a) No teacher exists or teacher quality < student target. (b) Inference-side latency budget excludes the larger teacher's outputs at scale (e.g., on-device). (c) Distillation cache pipeline is more expensive than just more pretraining tokens. (d) Student has different architecture/vocab — distillation can transfer biases. (e) Adversarial / safety-fine-tuning where teacher's distribution would dilute the desired student behavior.

**Q16.3 [M]** Make the case for 131K vocab. When would you regret it?
- ★ Case for: serious multilingual coverage (Hindi, Arabic, Chinese fertility), avoids byte-fallback fragmentation on rare scripts, future-proofs for tool tokens + special markers. Regret: 100M+ embedding params dominate a 250M model (40%); LM-head compute dominates at small batch (low MFU); inference quantization (int4) is heavier per token. Switch to 64K if model size < 500M and multilingual isn't core.

**Q16.4 [H]** Mixed precision: bf16 vs fp8 — when does each make sense?
- ★ bf16: pretrain default; same exponent range as fp32 (no loss scaling); production-mature on H100+. fp8: ~2× faster matmul throughput on H100/H200 (Hopper introduced E4M3 + E5M2 formats); requires careful scaling per-tensor; quality results competitive with bf16 if scaling is right. Use fp8 if: hardware supports it natively (H100+), training infra has scaling-recipe maturity (Nvidia TransformerEngine), willing to invest in scaling QA. Otherwise bf16.

**Q16.5 [M]** Add support for a new low-resource language. Minimal change?
- ★ Strong answer: (1) measure tokenizer fertility on the new language; if >3-4 tokens/word, retrain tokenizer or merge in new tokens; (2) audit training data — if <1% of total corpus, model won't generalize there; (3) add 1-5% of high-quality target-language data via mixture upsampling; (4) add per-language val splits to track loss; (5) eval on a held-out target-language benchmark, not just MMLU.
- ⚠ "Just add more data" (no analysis of fertility / measurement plan).

---

## §17 — Project-specific (MyLLM; tests prep)

These ONLY make sense if the candidate has read `docs/PROJECT_OVERVIEW.md` and the relevant source files.

**Q17.1 [M]** In MyLLM, what does `_PERSIST_KEYS` do and why is it important?
- ★ Defined in `src/myllm/training/loop.py:67-74`. Tuple of keys that get saved to checkpoint: trainable_variables, non_trainable_variables, opt_state, step, lr_recovery_multiplier, data_position. Important because the train_step preserves ALL state keys via `dict(state).update(...)`, but the checkpoint manager only persists this subset. Missing a key here means resume-restart from default (e.g., data_position back to 0 → silent corpus re-read).

**Q17.2 [H]** The packed-corpus L3 canary caught an off-by-one in resume seek. Explain the bug and the fix.
- ★ Bug location: `scripts/run_pretrain.py:783-786` (before fix, commit `715406c`). The trainer divided `data_position` by `packed_seq_len` (= context_length + 1) to compute `start_sequence_id`. But `data_position` is incremented by `B × context_length` per batch in `loop.py`. Off-by-one: at the tiny test scale, 128 // 33 = 3 vs correct 128 // 32 = 4. Resume re-consumed the last-trained sequence. Fix: divide by `model_input_len` (= context_length), not `packed_seq_len`. The synthetic L3 didn't catch it because synthetic data uses a different resume mechanism (start_step). Only the packed-corpus L3 (`canary_l3_resume_packed.py`) hit the bug.

**Q17.3 [M]** Gradient checkpointing is default ON in MyLLM. What's the tradeoff?
- ★ See `src/myllm/model/transformer.py`. Wraps each DecoderBlock in `jax.checkpoint`. Backward recomputes the block's forward instead of storing activations. ~33% more compute (each block runs twice), but ~4-8× lower backward-stored activation memory. Required at 1B/seq=8192 on H200 (without it: 176 GB needed > 141 GB available; with it: 61 GB fits). Off only for tiny smoke tests where recompute tax > memory savings.

**Q17.4 [M]** Training reads corpus from R2. Walk me through the path.
- ★ At training start, `PackedCorpusReader(args.packed_corpus_root)` opens a directory on local disk OR a fuse-mounted R2 prefix. The reader memory-maps `tokens.bin` per shard for random access by sequence_id. On resume, `peek_data_position_from_checkpoint` reads the latest checkpoint's manifest.extra.data_position, converts to start_sequence_id via `sequence_id_from_data_position(data_position, model_input_len)`, and seeks the reader. Pre-tokenized + R2-streamed = no HF stream / filter / tokenize cost at training time.

**Q17.5 [H]** What's a "decay-phase distillation" and how is it activated?
- ★ See `src/myllm/training/decay_phase.py`. The training loop has a `DecayPhaseActivation` object that activates in the last 15% of steps (per WSD schedule). Before activation: `decay_phase.maybe_inject(state, batch)` is a no-op; train_step sees no teacher data → pure CE. After activation: maybe_inject reads teacher top-K logits from the cache by data_position and adds them to the batch. train_step's `distillation_mixed_loss` sees teacher data → applies `α * CE + (1-α) * KL` with α annealing 0.7 → 0.3. Reading order: stable phase teaches base distribution; decay phase aligns to teacher.

**Q17.6 [H]** The FSDP HLO inspection (`MYLLM_DEBUG_HLO=1`) hard-fails on GPU but soft-logs on CPU. Why?
- ★ XLA's CPU backend lowers `reduce-scatter` semantically into `all-reduce + slice` — the string "reduce-scatter" doesn't appear in CPU HLO even when the JAX program is FSDP-correct. GPU XLA emits actual reduce-scatter (NCCL). So the assertion `reduce_scatter > 0` is valid only on GPU. The inspection always logs counts; hard-fails only when platform∈{gpu, tpu} AND reduce_scatter==0. Without this gate, every CPU smoke would false-fail.

---

## §18 — Quickfire / 1-minute each (warmup or filler)

| # | Q | Quick answer signal |
|---|---|---|
| 18.1 | What does `softmax(x + c) = softmax(x)` for any scalar c mean numerically? | LSE-stability trick |
| 18.2 | If your model has 16 layers, 32 heads, head_dim=64, what's hidden_dim? | 32*64 = 2048 |
| 18.3 | How many params in a SwiGLU FFN with hidden=H and ffn_dim=F? | 3*H*F (gate + up + down) |
| 18.4 | What's the memory cost of fp32 weights for a 1B model? | 4 GB |
| 18.5 | β₂ in Adam controls what? | Smoothing of the squared-gradient (second moment) |
| 18.6 | What's gradient noise scale? | Variance of gradient across batches / mean² — predicts batch-size sweet spot |
| 18.7 | What does τ=0 mean for nucleus sampling? | Greedy (only top token) |
| 18.8 | Why don't we use BatchNorm in transformers? | Sequence dim is batch-of-sequences, BN's batch dim doesn't correspond to anything meaningful |
| 18.9 | What's "logit lens"? | Apply LM head at intermediate layers to interpret what each layer "predicts" |
| 18.10 | How many bytes is a teacher top-K cache entry at K=64, bf16 logits + int32 indices? | 64*(2+4) = 384 bytes/token |

---

## Section-by-section grading rubric

For each section, a hire-worthy candidate should:

| § | Threshold for "strong" |
|---|---|
| 0 | Q0.4 + Q0.5 right |
| 1 | At least one math derivation cleanly |
| 2 | Reasoned discussion of why SwiGLU/RMSNorm/QK-norm/scaled-init |
| 3 | Flash attention + GQA + RoPE understood at mechanism level |
| 4 | Vocab/fertility tradeoffs not just memorized |
| 5 | Chunked-CE motivation + z-loss purpose |
| 6 | KL temperature math + decay-phase intuition |
| 7 | Adam vs AdamW + fp32 moments + spike recovery |
| 8 | Master weights pattern + bf16 vs fp16 |
| 9 | Reduce-scatter vs all-reduce for FSDP |
| 10 | Chinchilla vs over-training tradeoff |
| 11 | Decontam dual-mode + segment-aware packing |
| 12 | Memorization probe motivation |
| 13 | Atomic NaN-revert understood |
| 14 | At least 3 derivations clean |
| 15 | Methodical debug protocol (not "try random fixes") |
| 16 | Cost-aware, measurement-driven judgment |
| 17 | Read the repo, can point at file:line |

**Red flags across ALL sections:**
- Memorized buzzwords without mechanism ("attention attends to relevant tokens")
- Can't derive softmax gradient
- Doesn't distinguish all-reduce from reduce-scatter
- Believes Adam moments must be in param dtype
- "I'd just throw more compute at it" for any open-ended question
- Cannot name the difference between MFU and FLOPS/s

---

## Suggested 90-min interview plays

**Play A — "Generalist ML eng"** (mostly Easy/Medium, breadth)
1. Q0.4, Q0.5 (calibration: MFU + FSDP)
2. Q3.4 (GQA)
3. Q7.2 (Adam vs AdamW)
4. Q9.3 (collectives)
5. Q11.2 (MinHash+LSH)
6. Q15.1 (debug: loss spike)
7. Q16.1 (budget judgment)
8. Q14.2 (whiteboard: softmax CE gradient)

**Play B — "Research-track / scaling specialist"** (Hard, depth)
1. Q1.5 (matmul VJP)
2. Q2.6 (muP)
3. Q3.6 (RoPE math)
4. Q3.9 (flash attention)
5. Q9.4 (reduce-scatter)
6. Q14.5 (tied embedding gradient)
7. Q15.5 (FSDP HLO debug)
8. Q16.4 (bf16 vs fp8)

**Play C — "MyLLM project hire"** (heavy on §17, drops in §15)
1. Q17.1, Q17.2, Q17.5 (project knowledge)
2. Q15.5, Q15.6 (debug scenarios that match the codebase's bug-classes)
3. Q5.2 (chunked CE — they'll work on the loss path)
4. Q9.4 (FSDP — they'll work on the sharding path)
5. Q16.5 (low-resource lang — the project's Hindi hedge)
6. Q14.5 (BLOOM-tr11 bug class)

Total time per question: ~10-12 min including follow-up probes.
