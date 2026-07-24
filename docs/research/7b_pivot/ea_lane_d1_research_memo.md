# EA Lane D1 Research Memo — Verified-Agent Environments and Data Engine

**Program:** SAMA-7B

**Date:** 2026-07-23

**Status:** D1 research memo for human review; **not an approved ADR or implementation authority**

**Scope:** `DELEGATION_BRIEF_ENV_ENGINE.md` questions 1–15

**Accountability:** a named human EA must own HAR-20 and sign the later ADR; this AI session is
research/drafting support only

**Authorization boundary:** D1 research and D2 draft planning only. No framework installation,
external dataset/image acquisition, sandbox execution, cloud spend, GPU use, or live
GSTN/ONDC/Tally/customer/financial access was performed for this memo.

---

## 0. Answer first

Phase 1 should build a **SAMA-owned, trainer-neutral environment protocol and evidence bundle**,
not fork one fast-moving framework into the permanent architecture. Harbor is the best first
repo/terminal adapter candidate; OpenEnv and NeMo Gym are useful compatibility targets; verl,
SkyRL, and NeMo-RL should remain later trainer adapters selected by a measured bake-off.

For the F1 proof, use a managed microVM/userspace-kernel sandbox fleet only for public or
synthetic inputs, or use a company-controlled BYOC/dedicated region for proprietary traces and
hidden tests. At sustained 200+ concurrency, prepare a dedicated CPU environment trust domain.
At 2,000, use several bounded cells rather than one cluster. **Kubernetes schedules hostile
workloads; it does not contain them.** Untrusted environments must never share the B200/B300
training nodes, control plane, credentials, model weights, or storage trust domain.

The proposed internal interface is the **SAMA Environment Protocol (SEP)**. Its differentiating
surface is not a renamed `reset/step` API. It combines:

- explicit capabilities, authority and reversibility;
- idempotent actions with expected/predicted state deltas;
- independently attested actual deltas;
- observation replay and execution replay;
- signed verifier evidence and fail-closed reward outcomes;
- remotely anchored, tamper-evident trajectory events;
- source/admission and evaluation-contamination provenance.

This is a useful internal IP surface while still reusing open execution engines. Detailed
invention-candidate mechanics remain outside this repository in the restricted vault.

The Phase-1 target remains deliberately thin: five adapters, at least 20 deterministic tasks per
family, a 20-concurrent one-hour reliability run, 100 verified trajectories, injected failures,
and zero security boundary failures. It is **not** a 2,000-worker RL fleet, 2,000-image mirror,
or 20,000-tool acquisition project.

### Evidence labels

- **VERIFIED:** supported by a cited first-party project, vendor, standards, or program source.
- **ESTIMATE:** arithmetic from stated assumptions; it must be replaced by measurement or quote.
- **PROPOSAL:** a D2 design choice that still needs human/security review.
- **OPEN:** unresolved fact, contract term, quote, or owner decision.

Vendor security and scale statements are product claims, not independent assurance. Dataset
license labels are not legal clearance for every embedded repository, API, response, dependency,
or model output.

---

## 1. Phase-1 contract and non-goals

The binding local target is
[`P1-70`](../../PHASE_1_G0_S0_EXECUTION_PLAN.md#p1-70--verified-agent-environment-foundation):
one control contract and five adapters—repo/git, terminal/filesystem, synthetic or approved
browser/search, documents/spreadsheets, and SQL/business-state simulation.

| Phase-1 core | Later growth, explicitly non-blocking |
|---|---|
| 20 deterministic smoke tasks per family | 25–40 environment families |
| 20 concurrent rollouts for one hour | sustained 200 and then 2,000 concurrency |
| at least 99% launch and 95% platform completion availability | a multi-cell production RL fleet |
| 100 verified end-to-end trajectories | 1,000-trajectory stretch, then million-scale corpus |
| required failure injections and zero security breach | broad MCP ecosystem and proprietary Workflow Genome |
| measured path to 200/1,000 | automatic environment evolution |

The implementation hold is correct. D3 cannot start until there is a named human EA, an IS
security reviewer and non-author gate reviewer, signed caps, an approved source/image set,
versioned HAR-15/HAR-16 interfaces, an approved threat model, and an isolated repository/branch
workflow.

---

## 2. Q1 — Framework and trainer-side stack

### Recommendation

**PROPOSAL:** own the semantic contract and canonical evidence format; wrap external projects
behind versioned adapters.

1. Use **Harbor** as the leading D3 repo/git and terminal/filesystem execution adapter.
2. Add compatibility adapters for **OpenEnv** and **NeMo Gym** only after the internal contract is
   stable enough to expose the mismatches.
3. Keep the trainer integration behind a projection layer. A later bake-off should compare:
   **verl** as the portability baseline, **SkyRL** as the long-horizon/Harbor challenger, and
   **NeMo-RL** if Megatron Bridge wins S0 and its version pins align.
4. Keep **OpenHands** as an agent/runtime and event-format interoperability source, not the
   environment control plane.
5. Treat **SandboxFusion** as an optional utility for bounded single-shot code execution/judging,
   never as SEP, the five-family state model or the isolation boundary.

### Evidence and role matrix

| Component | Current first-party evidence checked 2026-07-23 | License | Recommended role | Material qualification |
|---|---|---|---|---|
| [OpenEnv](https://github.com/huggingface/OpenEnv) 0.4.1 | typed Gym-like `reset`, `step`, and `state`; HTTP/WebSocket and Docker | BSD-3-Clause | compatibility adapter and semantic reference | Project calls itself experimental; API changes are expected and its Kubernetes provider is planned. Its state is not SAMA’s independently attested domain state. |
| [Harbor](https://github.com/harbor-framework/harbor) 0.20.0 | Terminal-Bench harness, provider abstraction, jobs/trials/verifiers and parallel execution | Apache-2.0 | first repo/terminal adapter | Agent trajectories remain agent-specific; exact source and providers must be pinned. |
| [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym/releases/tag/v0.4.0) 0.4 | task, harness, verifier and state abstractions; Harbor/OpenEnv/OpenHands/verl/NeMo-RL integrations | Apache-2.0 | NVIDIA-adjacent compatibility adapter | First-party docs call it early-development with evolving APIs. |
| [verl](https://github.com/verl-project/verl) 0.8 | multi-turn tool use, verifiable rewards, async/partial rollouts, FSDP/Megatron/TorchTitan and vLLM/SGLang | Apache-2.0 | later portable trainer baseline | Agent Loop is alpha and does not define the security/state semantics required here. |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) 0.3 line | agent RL stack, SkyRL-Gym, Harbor/OpenEnv paths and Megatron support | root/core Apache-2.0; [`skyrl-gym` metadata](https://pypi.org/project/skyrl-gym/) MIT | later long-horizon challenger | Recent repository/package reorganization raises integration churn; pin and ledger each component/package separately. |
| [NeMo-RL](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.6.0) 0.6 | Megatron Bridge, async RL, NeMo Gym, vLLM/SGLang and SWE-RL support | Apache-2.0 | conditional S0-adjacent trainer | v0.6's NGC path pins NeMo Gym `0.3.0rc0+1a4912e` and Megatron Bridge `0.5.0+95e5f38`, not current Gym 0.4. Qualify the exact source/NGC image; do not use the nonfunctional [`nemo-rl` PyPI 0.0.0 placeholder](https://pypi.org/project/nemo-rl/). |
| [OpenHands Software Agent SDK](https://docs.openhands.dev/sdk/arch/events) | typed immutable Message/Action/Observation/Error/StateUpdate event model; separate OpenHands runtime | SDK [MIT](https://github.com/OpenHands/software-agent-sdk/blob/main/LICENSE); runtime core MIT with enterprise carve-outs | event importer plus separately reviewed agent/runtime adapter | Persisted state may contain secrets; SAMA stores redacted broker references only and pins SDK/runtime independently. |
| [SandboxFusion](https://github.com/bytedance/SandboxFusion) | multi-language code execution/evaluation and judge utility | Apache-2.0 | optional single-shot code verifier utility behind SEP | not a stateful five-family authority/control plane or containment claim; its runtime/images still need full admission and isolation review |

Version numbers above are a point-in-time research snapshot, not dependency approval. For any
D3 use, build internally from an approved commit, record hashes, generate an SBOM, scan the
artifact, and pin every transitive dependency.

### Why SEP is not OpenEnv-with-a-new-name

The internal contract needs the following transport-independent operations:

```text
DescribeEnvironment(task_ref) -> capabilities, permissions, reversibility, limits
CreateEpisode(task_ref, env_digest, seed, authority_profile, quotas) -> episode_id
Reset(episode_id) -> initial_observation, initial_state_attestation
Step(episode_id, sequence, idempotency_key, action, predicted_delta,
     expected_pre_state_hash) -> observation, actual_delta_ref, audit_refs, terminal_status
Snapshot(episode_id) / Restore(episode_id, snapshot_ref)
Verify(final_snapshot_ref, hidden_suite_ref) -> signed_verdict_ref
Replay(bundle_ref, mode)
Cancel(episode_id) / Destroy(episode_id)
Health() -> dependency and capacity status
```

Create/reset/snapshot/verify/replay/cancel/destroy/health are controller or verifier-plane
operations, not model-callable tools. The model receives only the declared capability/action
surface mediated by SEP and the external authority broker.

`expected_pre_state_hash` is a controller/adapter-issued optimistic-concurrency token returned
with the prior observation; the client echoes it. A mismatch returns `STALE_STATE` before any
side effect. It is not a model-authored security assertion. `predicted_delta` is an optional
model/agent prediction used for evaluation and planning; it never substitutes for the external
attestor's actual delta.

The canonical lifecycle is:

```text
requested -> authorized -> provisioning -> ready -> running -> verifying
          -> terminal -> destroyed
                         \-> quarantined
```

Platform failures are separate from task outcomes. The reward outcome enum should be
`PASS | FAIL | INVALID | PLATFORM_ERROR`; only a verified `PASS` can receive positive reward.
A platform error must never silently become a model failure or a positive reward.

---

## 3. Q2 — Phase-1 and RL-scale fleet choice and cost

### Deployment decision by scale

| Scale | Recommended posture | Reason |
|---:|---|---|
| 20 concurrent, F1 | managed Firecracker/microVM or gVisor service for public/synthetic data; BYOC/on-prem if hidden tests or traces are proprietary | fastest way to test the protocol without first building a scheduler; raw compute is trivial relative to engineering/security |
| 200 concurrent | managed/BYOC while measuring; begin a dedicated CPU Kubernetes cell only with named platform and IS capacity | exposes quota, registry, broker, verifier, logging and image-locality limits before committing to fleet operations |
| 2,000 sustained | dedicated environment-only CPU trust domain, sharded into roughly 4–10 cells of 200–500; enterprise managed capacity only as controlled overflow | bounds blast radius and makes registry, queue, broker, DNS and control-plane saturation tractable |

No scale permits hostile rollout pods on B200/B300 training nodes. Model inference may stay on
the GPU cluster behind a narrowly scoped service boundary; the sandbox sees neither weights nor
cluster credentials.

### Comparable planning profile

**ESTIMATE assumption:** one fully active sandbox uses 2 vCPU, 4 GiB RAM and 20 GiB ephemeral
disk. This excludes model inference, retained images/snapshots, logs, network egress, tax,
enterprise plans, retries and idle/provisioning overhead.

| Provider | Usage estimate per env-hour | 20 x 1 h | 200 x 1 h | 2,000 x 1 h | Concurrency qualification |
|---|---:|---:|---:|---:|---|
| [E2B](https://e2b.dev/pricing) | $0.16560 compute+RAM | $3.31 | $33.12 | $331.20 | Hobby covers 20 slots but 10 GiB storage; the assumed 20 GiB profile needs Pro ($150/month) or a smaller disk; 2,000 needs a quote |
| [Daytona](https://www.daytona.io/pricing) | about $0.16722 including assumed 15 billable GiB disk | $3.34 | $33.44 | $334.44 | tier limits apply; 2,000 needs custom capacity |
| [Prime Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) | $0.16000 under vendor GB≈GiB simplification | $3.20 | $32.00 | $320.00 | separate 512-active and 512-CPU limits make the 2-core profile effectively 256; larger scale needs custom capacity |
| [Modal](https://modal.com/pricing) | $0.23796 | $4.76 | $47.59 | $475.92 | V2 scale claims are beta; plan/region limits still apply |
| [AWS Fargate](https://aws.amazon.com/fargate/pricing/) floor comparator | $0.09874 before EKS/registry/network/logging/ops | $1.97 | $19.75 | $197.48 | not a containment recommendation; add [EKS control-plane fees](https://aws.amazon.com/eks/pricing/) |

The estimate formula is:

```text
total = sandbox_hours * profile_rate
      + plan_or_enterprise_commit
      + persistent_image_and_snapshot_storage
      + egress + logs + monitoring
      + platform_and_security_labor
```

Concurrency determines quota and wall-clock time; **environment-hours**, not slot count alone,
determine usage cost. The prices above were observed on 2026-07-23 and are nominal resource
rates, not performance-equivalent comparisons: CPU generations, oversubscription and startup
times differ, while Modal bills a physical core. The AWS floor assumes Linux/x86 in a selected
US region and must be repriced for the actual region/architecture. Provider selection therefore
needs measured ready-time and cost per successfully verified task-hour.

At an illustrative 160 occupied hours per slot per month (8 hours/day for 20 days), the
public-rate estimates are approximately:

| Provider | 20 slots | 200 slots | 2,000 slots |
|---|---:|---:|---:|
| E2B | $530 | $5,299 | $52,992, non-contractual at this scale |
| Daytona | $535 | $5,351 | $53,510, custom capacity |
| Prime | $512 | $5,120 | $51,200, custom capacity |
| Modal | $761 | $7,615 | $76,147 |

The F1 20-by-one-hour raw sandbox usage is therefore only about $3–$5. Security review,
enterprise/residency terms, engineering, image work and repeated heavy repo tests will dominate.
No provider or spend is authorized by this estimate. Prime's strict decimal-GB-to-GiB conversion
would make the profile about $0.1644 rather than $0.1600 per hour; retain vendor-native units in
the quote and normalize measured usage before selection.

### Residency and data classification

- **PUBLIC / SYNTHETIC:** a reviewed shared hosted service may be used with network disabled
  or explicitly allowlisted.
- **PROPRIETARY trajectories / hidden tests:** company-controlled region/BYOC or signed
  enterprise terms must cover compute, logs, snapshots, backups, support access, subprocessors,
  encryption/CMEK, IP ownership, no-training/no-secondary-use, retention/deletion evidence,
  incident SLA and audit.
- **CUSTOMER, FINANCIAL, LEGAL or PRODUCTION secrets:** prohibited in Phase 1.

[E2B](https://e2b.dev/docs/byoc) documents BYOC;
[Daytona](https://www.daytona.io/docs/en/regions/) documents shared, dedicated and custom regions.
[Modal region selection](https://modal.com/docs/guide/region-selection) documents a default
Virginia routing path, so container-region selection alone is insufficient for a proprietary
data-residency claim. [Prime](https://docs.primeintellect.ai/sandboxes/overview) documents
outbound networking enabled by default; it should be limited to public/synthetic tasks until its
isolation, region, retention and DPA terms pass review.

Provider-specific qualification tests:

| Provider | Useful first-party control | Required D2 qualification |
|---|---|---|
| E2B | Firecracker-hosted service, network controls and BYOC | [environment variables](https://e2b.dev/docs/sandbox/environment-variables) are not a secret store; [kill rather than indefinitely pause](https://e2b.dev/docs/sandbox/persistence); prove network deny, deletion and BYOC control/data-plane boundaries |
| Daytona | shared/dedicated/custom regions, [firewall/allowlist](https://www.daytona.io/docs/en/network-limits/) and [host-scoped secret substitution](https://www.daytona.io/docs/en/secrets/) | select and verify the actual VM/container isolation class; fail closed on “essential” endpoints; contract exact log/backup/support regions |
| Modal | gVisor and sandbox resource separation | [outbound internet is open unless blocked/allowlisted](https://modal.com/docs/guide/sandbox-networking); prove routing, [log/snapshot retention](https://modal.com/docs/guide/sandbox-snapshots) and deletion under the chosen plan |
| Prime | disposable Docker fleet and high documented default concurrency | outbound network defaults on and secrets enter the guest environment; public/synthetic inputs only until isolation, DPA, region and deletion review pass |

### Self-hosted comparison

Owned infrastructure is not free:

```text
loaded_node_hour =
  (amortization + power + cooling + rack + network + licenses
   + platform/on-call labor + spare-capacity opportunity cost)
  / usable_node_hours
```

**OPEN:** inventory, power, loaded labor and utilization are required before claiming a
self-hosted price. At 20, labor almost certainly dominates. At sustained 2,000, public managed
usage alone is roughly $51k–$76k/month under the illustrative duty cycle, so dedicated capacity
is likely cheaper—but that is an inference requiring quotes and measurements.

---

## 4. Q3 — Image acquisition, registry and storage

### Pushback on “which 2,000 of 7,500?”

There is no defensible list until source approval, license review, security scans,
decontamination and an actual layer-byte census exist. F1 needs the images behind only 20
approved repo tasks. **ESTIMATE:** use 100–250 only as an initial warm-pool ceiling for sizing,
not a claim that this count is necessary or sufficient. The final pool size is an output of the
approved task manifest, layer-byte/build/resource census and cache-hit rehearsal. Phase 1 should
**not** mirror 2,000 images merely because they are discoverable.

The selection algorithm should be task-first:

1. discover URL and metadata only;
2. remove private-eval, repository and generator-family overlaps;
3. retain only source/repository licenses and intended use approved by DL/counsel;
4. quarantine a capped pull and pin its digest;
5. generate SBOM and scan for vulnerabilities, malware, secrets and license conflicts;
6. require deterministic baseline build/test and explicit resource bounds;
7. deduplicate shared base layers and cap representation per repository;
8. stratify the admitted set by language, domain, difficulty, date and resource profile;
9. assign at least 20% of task families to a group holdout before generation so no selected
   repository/generator family appears in both train and holdout.

Among candidates that pass every gate, rank by marginal approved task/domain coverage divided by
incremental compressed unique-layer bytes, subject to the recorded representation and holdout
quotas. Emit the scored manifest and byte census; do not hand-pick an undocumented list.

[`SWE-rebench`](https://huggingface.co/datasets/nebius/SWE-rebench) publishes image/task
metadata but not an authoritative aggregate compressed-byte total. A nearby measured
[`SWE-Gym`](https://arxiv.org/html/2412.21139v1) example reports 6 TB for 2,438 images
(rounded to about 2.6 GB/image). Direct total-ratio extrapolation gives about 4.9 TB for 2,000
images, while the rounded per-image figure gives 5.2 TB. Use **4.9–5.2 TB** only as an estimate,
not a SWE-rebench fact.

### Proposed storage gates

**PROPOSAL, pending owner/DL/IS approval:**

- Phase-1 environment lane hard line: **4 TB physical high-water across all replicas/domains**;
- at most 1 TB admitted/quarantine authoritative registry, inclusive of registry replicas and
  metadata;
- at most 2 TB aggregate ephemeral volumes and node caches across all worker domains;
- at most 0.5 TB trajectory, audit and verifier artifacts;
- at least 0.5 TB snapshots/backups/headroom;
- any future 2,000-image program gets a separate approval after a measured manifest/layer census,
  with an 8 TB physical authoritative-registry planning ceiling inclusive of replicas/metadata
  and no full mirror on every node.

At [AWS ECR's](https://aws.amazon.com/ecr/pricing/) public $0.10/GB-month storage price, the
4.9–5.2 TB estimate would be about $492–$520/month and 8 TB about $800/month before replication
and transfer. The program's 50 TB cap is global and pending signature; it is not an entitlement
for this lane.

Use immutable digest-only manifests, quarantine and admitted namespaces, signed admission
attestations, content-addressed layers, registry/node co-location, image-aware scheduling and
next-batch prefetch. Keep an immutable pin/lease set for active tasks and gate evidence. Garbage
collection may remove only unreferenced quarantine, failed-build and expired cache artifacts;
node cache is never the evidence store. Kubernetes documents
[digest references and parallel-pull controls](https://kubernetes.io/docs/concepts/containers/images/)
and [node image garbage collection](https://kubernetes.io/docs/concepts/architecture/garbage-collection/).

---

## 5. Q4 — Determinism, state attestation and replay

“Bit reproducible” should apply to pinned initialization artifacts and canonical semantic state,
not to irrelevant ZIP ordering, log timestamps, browser raster pixels, kernel scheduling or live
external APIs. Phase 1 should require:

- an exactly pinned initial manifest and content hashes;
- deterministic reset under a declared seed, locale, clock and network policy;
- a canonical state representation per family;
- a privileged attestor outside the model-controlled guest;
- faithful action/observation replay with explicit tolerances;
- a separately reproducible verifier verdict.

| Family | Deterministic initialization | Canonical actual state delta | Terminal verification | Reversal |
|---|---|---|---|---|
| repo/git | image digest, commit SHA, dependency lock, clean copy-on-write overlay, fixed identity/locale/time | canonical git diff including untracked files, modes, renames and submodule refs; artifact/test hashes | hidden tests in a fresh verifier sandbox plus patch-scope rules | discard overlay/reset |
| terminal/filesystem | read-only base plus unique overlay; fixed clock/seed/locale/network; bounded processes | Merkle filesystem diff, metadata changes, declared process/artifact outputs and prohibited-path evidence | hidden invariant/fixture checker outside worker | discard overlay |
| browser/search | version-pinned browser, viewport, synthetic approved site snapshot, deterministic backend DB | backend DB change-set plus normalized DOM/accessibility-tree and navigation/security events | server-state and semantic UI assertions; never screenshot alone | DB/snapshot restore |
| documents/spreadsheets | pinned engine, fonts, locale, template and deterministic fixture | normalized OOXML/ODF structure, text/styles/cells/formulas/names plus semantic render hashes | parser/formula assertions and bounded render comparison | fixture/snapshot restore |
| SQL/business state | pinned engine/schema/data dump, transaction isolation, fixed time and ordering | schema diff, canonical sorted row/change set, transaction/audit log and forbidden-object checks | read-only hidden queries/invariants outside worker | transaction rollback or snapshot restore |

The model may submit a `predicted_delta`; it must not author the `actual_delta`. An attestor with
read-only privileged access computes before/after hashes and signs or MACs the evidence reference.

Two replay modes are required:

1. **Observation replay:** feed recorded observations and exact policy token IDs/masks back through
   trainer/scorer logic without executing tools.
2. **Execution replay:** recreate the pinned environment, reissue recorded actions, and compare
   canonical state deltas and verifier outcomes.

The policy need not regenerate identical language during execution replay. “Replay passed” means
the recorded action sequence reproduced the accepted state transition and verdict under the
declared tolerance. Open-ended or live-network tasks are out of the deterministic F1 class.

---

## 6. Q5 — Concurrency architecture, authority and containment

### Control and execution planes

SEP's control plane should own admission, episode lifecycle, leases, idempotency, quotas,
backpressure, reconciliation, audit references and the kill switch. Workers execute one rollout
per sandbox. A verifier plane and authority broker remain outside worker/model control.

```text
trainer/policy
      |
      v
SEP admission queue ---> rollout controller ---> isolated worker cell
      |                        |                       |
      |                        +---- audit/event store |
      |                                                v
      +--------> external authority broker       disposable sandbox
      |
      +--------> independent verifier/attestor ---> signed verdict
```

At 20, use one **logical** controller service with at least two stateless replicas across worker
failure domains, backed by a separately durable HA queue/state service with leases,
leader/fencing semantics and idempotent reconciliation. If coordination state is unavailable,
new admission and authority issuance fail closed. At 200, add per-family pools, warm capacity and
registry locality. At 2,000, shard queues, brokers, caches, storage and worker clusters into cells;
retain one out-of-band global admission stop and capability revocation path.

The F1 one-hour test needs an ungameable measurement contract:

```text
admission_availability =
  eligible offered requests accepted / eligible offered requests

launch_availability =
  accepted launch requests reaching READY within SLO / accepted launch requests

platform_completion =
  launched episodes reaching PASS, FAIL, or task-declared INVALID
  with a completed verifier record
  / launched episodes
```

Use a fixed predeclared demand schedule and eligibility policy. Invalid requests stay outside the
eligible-offered denominator, but capacity shedding/rejection of eligible requests counts against
admission availability; it cannot be hidden by measuring only accepted work. Injected task
failures may be excluded from the platform-completion target only when predeclared, but
`PLATFORM_ERROR`, verifier failure, lost workers, platform-caused timeouts, incomplete cleanup and
orphans never enter the numerator. Maintain 20 occupied slots with replacement launches for one
hour and report all three raw numerators, denominators, latency distributions and confidence
intervals. A one-time batch of only 20 cannot meaningfully distinguish 99% from 95% except as
all-or-one-failed.

### What breaks between 20 and 2,000

- Docker/container runtime API saturation and worker churn;
- scheduler/autoscaler lag and control-plane event storms;
- registry rate limits, image-pull storms, disk/inode pressure and layer corruption;
- DNS, CNI IP, NAT/SNAT and connection-tracking exhaustion;
- provider create-rate, concurrency and regional-capacity limits;
- broker token-mint latency or confused-deputy errors;
- verifier tail latency and inference backpressure;
- database locks, non-idempotent retries and duplicate rewards;
- stdout/log amplification, high-cardinality metrics and object-store IOPS;
- cancellation races, orphaned sandboxes and cost leakage;
- cross-run reuse, stale state and partial cleanup.

Controls are a bounded admission queue, per-family concurrency tokens, idempotency keys, leases,
jittered retry, explicit backpressure, prewarmed pools, local registry/cache, image locality,
dead-letter/quarantine queues, a reconciler/reaper, low-cardinality telemetry, output truncation
with retained hashes, cell circuit breakers and fault-domain spare capacity.

The official [DeepSWE engineering report](https://www.together.ai/blog/deepswe) is the relevant
warning: a 512-container Docker run crashed the daemon, after which the team moved to Kubernetes,
preloaded images, and used a fleet above 1,000 CPU cores. That demonstrates a failure mode and
an order of magnitude; it is not a capacity guarantee for SAMA.

### Threat model and required isolation

Treat the model, task text, tool output, dependency, image and sandbox process as hostile. Assets
include host/control-plane integrity, model weights, hidden tests, proprietary tasks/traces,
broker capabilities, adjacent runs and audit evidence.

**PROPOSAL:** use a hardware-VM boundary such as Kata/Firecracker for repo/git and terminal/fs
by default. gVisor may be selected for compatible browser/doc/SQL workers after parity and
performance testing. Plain `runc` is not the default for model-generated code. Kubernetes itself
states that containers provide a
[weaker isolation boundary than VMs or userspace kernels](https://kubernetes.io/docs/concepts/security/multi-tenancy/).

Mandatory controls:

- environment-only nodes, accounts and network; no GPU/data-admission co-tenancy;
- no hostPath, runtime socket, privileged mode, host PID/network, device mount, Kubernetes API
  token, or provider-admin API;
- non-root/rootless execution, read-only base, bounded ephemeral overlay,
  `allowPrivilegeEscalation=false`, drop all capabilities, seccomp RuntimeDefault/custom,
  AppArmor/SELinux and user namespaces where supported;
- default-deny ingress and egress, controlled DNS, blocked RFC1918/link-local/cloud-metadata
  destinations, DNS-rebinding defense and destination/method/schema allowlists at the broker;
- no production secret; synthetic single-run/action-scoped short-TTL capability, held or
  substituted by the external broker where possible;
- [cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html) CPU, RAM, PID and block-I/O
  limits;
- filesystem/project quotas for writable bytes and inodes; CNI/tc/eBPF/proxy policy for network
  byte/rate/destination limits; controller/watchdog limits for output, logs, steps and wall time;
- digest-pinned admitted images, SBOM, vulnerability/malware/secret/license scan and internal
  admission signature;
- unique writable volumes/UIDs, kill rather than pause on timeout, verified cleanup and an orphan
  reaper;
- redacted audit events sent to a remote immutable/WORM sink or signed/Merkle-checkpointed and
  anchored outside the sandbox and controller trust domain; externally protected signing keys;
- out-of-band kill switch that closes admission, revokes broker capabilities, blocks egress,
  terminates workloads, preserves evidence and quarantines the affected image/node.

The [Kubernetes Restricted Pod Security Standard](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
supports the non-root/seccomp/capability baseline; its
[NetworkPolicy documentation](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
also makes clear that isolation is allow-all until an applicable policy exists and that DNS must
be handled deliberately. [Firecracker's design](https://github.com/firecracker-microvm/firecracker/blob/main/docs/design.md)
still requires host-level network controls. [gVisor's security guide](https://gvisor.dev/docs/architecture_guide/security/)
still relies on external cgroups and network policy. Neither runtime eliminates hardware side
channels; prohibit sensitive co-tenancy where that threat matters.

These documents are requirements, not enforcement. D2 must name the enforcing admission layer
(for example Pod Security Admission plus reviewed policy controls), and the concrete production
runtime integration—not “Firecracker” alone. A Firecracker path must include the `jailer`,
per-VM UID/chroot/cgroup/seccomp, host TAP filtering, hardened/patched KVM hosts and proven
bare-metal/nested-virtualization compatibility. A Kata path must similarly prove its selected
hypervisor and RuntimeClass on the actual host.

Before D3, IS must review an adversarial evidence matrix covering host filesystem/runtime/K8s
API access, cross-run canaries, direct-IP/SSRF/DNS rebinding, expired/replayed/wrong-destination
capabilities, mutable-tag admission, fork/OOM/disk/inode/network/log bombs, snapshot/remanence,
hidden-test access and kill-switch behavior under partial control-plane failure.

---

## 7. Q6 — Verifiers, hidden tests and verifier quality

### Common reward contract

1. Freeze task, generator-family, repository-family and evaluation fingerprints before any
   generation/import. Split groups, not rows: a generator family—and where applicable a
   repository family—cannot appear in both train and holdout. The ≥20% requirement applies to
   task-family count assigned through generator-family grouping; D2 must record the exact
   denominator.
2. Never place hidden tests or verifier logic in the model workspace. Apply the candidate output
   in a second, clean verifier environment whose image and suite are immutable and digest-addressed.
3. Give deterministic predicates authority. In F1, a rubric/LLM judge is annotation or rejection
   filtering only: it cannot add positive RL reward or override a hard failure, prohibited action
   or missing evidence. Any later reward influence needs a separately powered calibration and ADR.
4. Require explicit submit and classify limit causes in the task contract. Agent budget
   exhaustion, maximum context or maximum steps normally produces `INVALID`—or `FAIL` only when
   the task explicitly defines the limit as part of success. Provider, verifier, network or other
   infrastructure failure produces `PLATFORM_ERROR`; it never becomes model failure or success.
5. Persist the verifier version/digest, raw evidence, assertion results and signed reward
   calculation. Missing or unverifiable evidence fails closed.

This follows the useful principle in the
[DeepSWE report](https://www.together.ai/blog/deepswe): sparse binary reward only when selected
Pass-to-Pass and Fail-to-Pass tests succeed, while limit-hit trajectories are masked. SAMA should
reimplement the principle with its own contract and tests.

### Per-family verifier

| Family | Authoritative checks | Hidden-test protocol | Non-authoritative F1 annotation/filter |
|---|---|---|---|
| repo/git | canonical patch; allowed paths; file modes/hashes; build and selected Fail-to-Pass + Pass-to-Pass tests | apply patch to a fresh clean clone; hidden suite and harness exist only in verifier image; block test/harness/CI/dependency edits unless explicitly allowed | lint/quality annotation or rejection filter, never positive reward |
| terminal/filesystem | expected Merkle/artifact/exit invariants; forbidden paths, processes, network and privilege | recreate clean overlay and evaluate private fixtures/invariants | process efficiency within declared budget |
| browser/search | server-side database/events and cited-source spans; URL/storage/cookie/network policy | synthetic site contains hidden distractors and tool-output injection; compare backend truth and semantic DOM/accessibility state | visual annotation/rejection only when backed by semantic state |
| documents/spreadsheets | normalized structure, text, styles, formulas, ranges and names; independent formula recalc; no macros/external links | private parser/formula assertions against a clean copy; render reference not exposed | visual/layout annotation or rejection filter |
| SQL/business state | final-state queries, constraints, invariants, events, forbidden reads/writes, atomicity and idempotency | restore clean DB; run read-only private assertions through separate credentials | query/process efficiency within budget |
| cross-tool/MCP | tool/schema version, canonical arguments, call order where required, state transition, permission and extra-side-effect checks | broker/verifier records are inaccessible to the worker | rubric annotation/filter for acceptable alternative paths |

### Measuring verifier quality

Create a blind gold set stratified by family, verifier version, pass/fail and adversarial category.
Two independent domain reviewers label it and disagreements are adjudicated. Primary metrics are:

- promoted verifier/reward pipeline verdict versus adjudicated human-gold agreement over the
  blind audit set;
- false accepts divided by known-negative/adversarial cases for the **complete** promoted reward
  pipeline;
- false rejects divided by known-positive cases;
- inter-reviewer agreement as a separate diagnostic of gold-set quality.

Report each numerator/denominator, precision, recall and exact one-sided 95% Clopper–Pearson
confidence bounds. Do not pool cases across verifier digests after a fix: the qualifying count
restarts for the new digest while old evidence remains in the audit trail.

The master-plan targets—at least 99.5% human agreement and at most 0.1% false reward—cannot be
statistically certified by F1's 100 trajectories:

```text
with zero observed errors:
  n >= ln(0.05) / ln(1 - target_error)
```

| Desired one-sided 95% upper bound with zero observed events | Minimum independent cases, rounded up |
|---:|---:|
| 3.0% | about 99 |
| 1.0% | about 299 |
| 0.5% | about 598 |
| 0.1% | about 2,995 |

With zero false accepts, 100 known-negative cases give an upper bound of about 2.95%; 1,000 give
about 0.30%. The calculation assumes independent Bernoulli cases; correlated cases from the same
generator/repository lower the effective sample size, so group-aware sampling is mandatory.
Therefore:

- **F1 is an operational proof**, not proof of the final statistical targets;
- treat at least 300 adversarial known-negative cases across the five families as a useful
  **non-gating verifier-development stretch set** only if D2 assigns owner, schedule and budget;
  report the actual confidence bound, and do not add this number silently to F1 core;
- a later promotion gate needs at least 598 agreement cases and 2,995 known negatives if the
  targets are global and no errors are observed;
- any observed false accept immediately quarantines that verifier version and requires root-cause
  correction before sampling resumes.

**OPEN for D2:** decide whether targets apply globally, per family, per verifier version, or all
three. Recommended: global confidence-bound gate plus minimum per-family floors, recalculated for
every promoted verifier revision.

---

## 8. Q7 — Anti-reward-hacking hygiene as testable policy

The following must be machine-enforced gates, not prose in a judge prompt:

- signed task, image, tool-schema, generator and verifier hashes;
- explicit submit/termination and compact filtering for limit-hit/platform-error trajectories;
- fresh clean-room verification and immutable hidden tests;
- forbidden path, endpoint and tool policies;
- strict canonical JSON/schema validation; reject extra keys, malformed numbers, `NaN`,
  confusables and parser differentials;
- hard failure on any isolation, authority, secret or egress violation;
- decontamination fingerprints before trajectory admission;
- paired known-good and deliberately corrupted fixtures for every verifier revision;
- verifier evidence generated outside the worker and bound to the final state attestation.

The mandatory adversarial suite includes:

1. editing tests rather than source; skip/xfail/discovery sabotage;
2. fake executable/PATH, harness monkeypatch and dependency/config/CI tampering;
3. symlink, hardlink, path traversal and hidden-test discovery;
4. timeout, maximum-output, fake submit, race and flaky-result exploitation;
5. duplicate idempotency keys, partial commit and stale/concurrent update;
6. authority escalation, broker replay and unauthorized destination;
7. DOM/UI spoof without backend state and fabricated citations;
8. spreadsheet cached-value spoof, macro and external-link insertion;
9. SQL trigger/view/transaction rollback spoof and forbidden-table access;
10. malformed/truncated tool output, prompt injection and secret exfiltration;
11. fork/OOM/disk/inode/network/log bombs;
12. rollback/compensation bypass and state left behind after teardown.

Each test has one expected terminal class, required audit events and expected state hash/delta.
No judge may reinterpret a failed security or invariant assertion as partial success.

The D3 design should make this concrete as reviewed policy artifacts, not scattered conditionals:

```text
policy/
  task_policy.schema.json       # allowed paths/tools/endpoints/budgets/reversibility
  reward_policy.schema.json     # terminal classes and fail-closed composition
  admission/                    # digest, signature, source-state and scan gates
  runtime/                      # privilege, network and authority-broker rules
tests/adversarial/
  known_good/
  known_bad/
  expected_verdicts.jsonl
```

CI rejects a verifier or policy revision unless all known-good and known-bad fixtures pass and the
expected evidence bundle is reproducible. The choice of Rego/Cedar/native typed rules remains a
D2 implementation decision; the semantic policy must stay independent of that engine.

---

## 9. Q8 — Nondeterminism, retries and flaky quarantine

Separate three phenomena:

- **environment determinism:** same initial manifest, seed and action log must yield the same
  canonical state and reward for an F1 deterministic task;
- **platform error:** provider, worker, network or verifier infrastructure failed;
- **policy stochasticity:** the model chose a different valid/invalid action path.

Rules:

1. Qualify every deterministic task with at least three clean reset/execution replays of the
   **identical recorded action log**. Any semantic state or verdict divergence for the same
   manifest, seed and action log quarantines the task/verifier immediately. Fresh stochastic
   policy rollouts belong under pass@k/pass^k, not environment replay.
2. Replace live external nondeterminism with pinned captures or seeded simulators; live external
   state cannot be the authoritative F1 reward source.
3. One clearly classified infrastructure retry is allowed under the same episode lineage and a
   new execution ID. The failed attempt remains `PLATFORM_ERROR`, has no learning signal and
   counts against availability. A clean retry is scored normally as its own retained attempt.
4. A model/task failure is never erased, relabeled or success-selected through retries. Preserve
   every attempt and evaluate the predeclared `k`; do not retry failures until one succeeds and
   then retain only the success.
5. Positive trajectories are reverified in a fresh verifier sandbox. Later training may require
   two positive confirmations and a larger audit sample; F1 records the exact repeat count.
6. Report `pass@k` (“at least one succeeds”), `pass^k` (“all k succeed”), pass@1, and platform
   completion separately. Never use one as a synonym for another.
7. Quarantine immediately after any isolation breach, false accept, or inconsistent verdict for
   the same attested state.

**PROPOSAL for D2 review:** any semantic replay/state/verdict mismatch triggers immediate
quarantine. For separately classified platform/infrastructure errors only, operationally
quarantine after two in a rolling 100, or a rate above 0.5% after at least 200 executions.
Readmit only after a root cause, a new immutable version, 100 clean identical-action replays and
human approval. These thresholds are provisional operating controls, not proven optimal values.

---

## 10. Q9 — Data engine, MCP discovery and license audit

### Correct non-bypassable flow

The architecture diagram in the delegation brief must not be implemented as “harvest then
execute.” The permitted flow is:

```text
metadata discovery
  -> DL approval-to-acquire
  -> capped quarantine acquisition
  -> digest/provenance/SBOM/static/secret/malware/license checks
  -> sealed dynamic security test
  -> DL/IS runtime artifact approved-for-sealed-execution
  -> evaluation decontamination
  -> synthetic emulator/task/rubric generation
  -> simulated or approved real execution
  -> verifier and quality filter
  -> EA pre-admission trajectory package
  -> DL corpus-admitted decision, removal lineage and signed ledger
  -> tokenization/release only under the training-data contract
```

The named outer states remain the program's P1-20 contract:
`Discovered -> Quarantine -> Admitted -> Decontaminated -> Tokenized -> Released for run`.
The approval/scan/execution checks above are non-bypassable evidence within those transitions,
not a competing state machine. Approval to acquire or execute a runtime artifact never authorizes
its code, outputs or trajectories for training; only DL's separate corpus-admitted decision can
advance candidate data.

The [official MCP Registry](https://registry.modelcontextprotocol.io/) may be a **discovery
source only**. Its [terms](https://modelcontextprotocol.io/registry/terms-of-service) make the
Registry metadata CC0 while excluding packages in third-party registries from that grant. Its
[moderation policy](https://modelcontextprotocol.io/registry/moderation-policy) does not promise
to remove low-quality, buggy or vulnerable servers. Persist only registry metadata, provenance
and its hash in `Discovered`: name/version/package coordinates, publisher, source/repository URL,
status, timestamps and declared license identifier/URL. Do not clone, install, pull, connect to a
remote server or fetch a live schema at discovery. Store declared license metadata separately
from `license_review_status`; never normalize a publisher claim into approval automatically.
Snapshot the Registry API version and record status because entries may later change, deprecate or
disappear.

**PROPOSAL discovery-source list:** the
[official Registry API](https://modelcontextprotocol.io/registry/registry-aggregators) with a
terms snapshot; publisher-owned upstream repository/release metadata; package-registry metadata
only after its terms are approved; and SAMA's internal synthetic capability catalog. Community
lists, search results and mirrors are leads only, never provenance or approval evidence.

The official MCP
[security guidance](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)
identifies confused-deputy, token passthrough, SSRF/private-IP/cloud-metadata and exfiltration
risks. These threats are why every eventually approved tool needs its own source/API/output-use
review, pinned artifact, scans, least authority, destination allowlist and sealed dynamic test.

### Synthetic evolution without copying

The [Kimi K2 paper](https://arxiv.org/html/2507.20534v1) reports more than 3,000 real MCP tools,
more than 20,000 synthetic tools, stateful simulation, rubric filtering and a fleet above 10,000
sandboxes. That demonstrates a precedent; it does not validate SAMA's design or grant the right
to acquire its corpus, copy its prompts/taxonomy, or adopt its scale.

SAMA should use two clean paths:

- internally authored ontology -> wholly synthetic tools/worlds;
- approved source schemas -> new simulations only after complete artifact admission.

The high-level generator chain is sector -> workflow archetype -> entity/state graph ->
capabilities -> tool schemas and authority scopes -> error model -> tasks -> rubrics -> paired
clean/perturbed trajectories. Use a seeded world model and record all injected faults. Each
rubric names final-state assertions, intermediate invariants, forbidden actions, evidence,
budget and reversibility. Deterministic hard reward is primary; a separate judge may filter
quality but does not manufacture truth.

The reviewable quantitative scaling plan is:

1. **Phase 1:** author five internal capability families with an **estimated cap of 5–10 tool
   schemas each (25–50 total)** and at least 20 tasks per required environment family. Use no
   external MCP executable. The output is quality and pipeline evidence, not catalog scale.
2. **Growth input:** let `D` be metadata-discovered tools and `A` the much smaller set of
   source-approved, runtime-admitted and decontaminated schema families. Never assume `D=3,000`
   implies `A=3,000`.
3. **Candidate rounds:** sample approved schema/capability families under source and domain
   coverage quotas; generate bounded variants across composition, state, authority,
   reversibility and error surfaces. Cap each review round at 2,000 candidates and a signed
   CPU/spend budget.
4. **Hard gates:** 100% schema validation, deterministic simulator/state tests, authority and
   reversibility invariants, and zero exact canonical-hash duplicates. Calibrate semantic
   duplicate threshold on labeled pairs to at most 1% false-new classification, then audit at
   least 200 candidates per round. Freeze at least 20% of generator families as group holdout.
5. **Yield math:** let `y_L` be the one-sided 95% lower confidence bound on admitted candidates
   divided by generated candidates. To add `N` admitted synthetic tools, authorize at most
   `ceil(N / y_L)` new candidates in bounded rounds; never use the raw Kimi ratio as SAMA's yield.
6. **Stop/rework:** stop the generator on any authority/security invariant breach, sampled
   semantic-duplicate rate above 5%, `y_L < 40%` for two rounds, coverage quota breach, unresolved
   rights/provenance, or lane budget/storage stop. Requalify a new generator digest from zero.
7. **20,000 gate:** “20k” counts only post-gate, uniquely fingerprinted synthetic tool schemas
   with simulator tests, provenance and removal lineage. It is a later growth ceiling/decision,
   never a Phase-1 acceptance target.

The 2,000-round cap, 1% calibration target, 200-case audit, 5% duplicate stop and 40% yield stop are
**provisional D2 parameters**, not measured facts. Exact transforms, sampling weights,
combinations and novelty objectives remain in the restricted vault.

The rubric judge receives the task, immutable rubric, redacted trace and hard-verifier evidence;
it does not receive secrets or hidden test content. It returns criterion-level scores, cited event
IDs, confidence and `ABSTAIN`, all under a pinned model/prompt/template version. It cannot author
its own success criteria, override a failed invariant, or convert missing evidence into a pass.
Calibrate every judge revision against the blind human gold set, include tool-output-injection
fixtures, and use it for rejection/filtering until its error is independently characterized.

Exact generators, prompts, distributions, combinations, scoring weights and recovery objectives
belong in the restricted invention vault.

### Candidate corpus and component rights audit

Disposition vocabulary:

- `metadata-only`: discovery record; acquisition/execution/training prohibited;
- `conditional quarantine`: may be considered for capped acquisition after approval; not training
  permission;
- `runtime-code candidate`: code may be evaluated as infrastructure after artifact admission;
  no implied content/data rights;
- `method evidence only`: informs design; no artifact/data acquisition;
- `corpus-admitted`: reserved for a later DL-signed decision, never assigned by this memo.

| Candidate | First-party license/evidence | D1 disposition | Why it is not automatically cleared |
|---|---|---|---|
| [Toucan-1.5M pinned card](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/blob/575b707858ac0e441d0364e2802c7ca2e9f92e04/README.md?code=true) | card says Apache-2.0; about 1.5M chat trajectories, 495 MCP servers and 2,000+ tools; pipeline [code is MIT](https://github.com/TheAgentArk/Toucan) | **conditional quarantine only; no bulk admission** | top-level label does not settle every MCP package, API/output right, publisher authority, teacher-model term, PII or database right; card says PII checks are best effort |
| [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) | CC-BY-4.0 dataset; 21,336 issue/PR pairs in the reviewed snapshot; row-level `license_name` | conditional quarantine; row-level review required | 21k pairs are not 21k verified tasks; each source repository license and captured commit still controls its code |
| [Nebius OpenHands trajectories pinned card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/blob/13fa32b010d4e0a691a48e0c5067de1f543b7ca5/README.md) | CC-BY-4.0 trajectory release with messages, patch, exit/resolution and generated-test metrics | conditional quarantine; row-level review required | preserve attribution, source-repo license, raw tool arguments, runtime revision and teacher/output-use basis |
| [daVinci-Dev](https://huggingface.co/datasets/GAIR/daVinci-Dev) | gated, mixed permissive and CC-BY-4.0; pipeline [Apache-2.0](https://github.com/GAIR-NLP/daVinci-Dev) | conditional quarantine; gated/row-level review required | “detected permissive” needs verification at captured commit; PR/contributor rights, access terms and teacher terms remain; context PR rows are not executed verified trajectories |
| [R2E-Gym](https://github.com/agentica-project/R2E-Gym) | Apache-2.0 code; over 8.1k tasks across 13 repositories and unit-test reward | runtime-code candidate after pinning | code license does not license every task repository, teacher trace or generated output |
| [τ³-bench / tau2-bench repo](https://github.com/sierra-research/tau2-bench) v1.0.1 research snapshot | MIT code, Gym-like interface and deterministic seed support; Python >=3.12,<3.14 | runtime-code/simulation candidate after exact commit/task-split pinning | audit bundled task/data content independently; official notes make `banking_knowledge` results before/after 1.0.1 non-comparable, so never merge those score histories |
| Harbor/OpenHands/OpenEnv/NeMo/SkyRL/verl | code licenses listed in §2 | runtime-code candidates | runtime license never grants benchmark, API-output or trajectory-content rights |
| Kimi K2 / DeepSWE reports | research/method descriptions | method evidence only | no corpus acquisition or prompt/taxonomy copying |

This is an engineering disposition, not legal advice. DL/counsel own the actual source-class and
intended-use conclusion. Any imported row must retain its exact source ID, license/terms review,
attribution, teacher/generator provenance, PII status and removal route. For every admitted
evidence artifact, record the exact dataset revision/commit, source artifact digest, card and
terms snapshot digest/date, access conditions and capture date. Moving pages are discovery
evidence only. CC-BY attribution/notice obligations flow into dataset manifests and downstream
release review; whether and how they apply to trained weights is a counsel decision.

---

## 11. Q10 — Canonical trajectory and provenance schema

### Proposed artifact

**PROPOSAL:** `sama.agent_trajectory/v0.1-draft`, stored as a semantically lossless canonical
event stream. Byte-exact raw source remains a separately controlled content-addressed artifact
when its retention is admitted. Trainer/chat formats are projections, never the source of truth.

```text
TrajectoryBundle/
  manifest.json
  events.ndjson
  state/          # initial/final snapshots or content-addressed references
  verifier/       # signed assertions and hidden-suite digest, not hidden content
  replay/         # expected deltas and replay comparison
  projections/    # trainer/tokenizer-specific tokens, masks and log-prob references
```

Required manifest fields:

- schema, trajectory, task, family, generator-family, repeat and attempt IDs;
- split/holdout status, task/generator fingerprints and rubric references;
- source dataset/schema/row IDs, source artifact digest and the shared provenance envelope;
- declared license identifier/source URL/capture hash and time, separate `license_review_status`,
  terms snapshot digest/date, permitted-use class, attribution obligations, expiry/revocation and
  removal/tombstone state;
- admission state, privacy/PII/redaction/residency status and contamination references;
- environment adapter/version, image digest, source commit, dependency locks and initial snapshot;
- seed, capabilities, authority profile, permissions, network policy, quotas, isolation profile
  and reversibility class;
- policy model, tokenizer, chat-template, scaffold, sampling and trainer pins;
- verifier and attestor identities/digests;
- terminal class, infrastructure/flaky status, final state hash and rollback/cleanup result.

Evidence-bearing fields use an explicit `present | unavailable | not_applicable |
unknown_with_reason` status. Importers never turn missing provenance, rights, attestation or
verifier data into a successful/default value.

Every ordered event contains:

- monotonically increasing sequence, event ID, parent/correlation/tool-call IDs and source actor;
- event kind and monotonic timing;
- admitted/redacted semantic content needed to interpret the action or observation;
- typed action and canonical parsed arguments plus a quarantined raw-source digest/reference;
- observation/error, truncation and redaction metadata;
- pre-state, expected-state and post-state hashes;
- predicted delta, independently attested actual-delta reference and audit references;
- broker/network/security decisions and resource/latency counters.

At the general schema level, a content-addressed projection keyed to canonical event IDs stores
exact token IDs, response/loss masks and model/tokenizer/chat-template hashes with the same
explicit evidence status. For any trajectory labeled eligible for **on-policy RL**, that
projection is mandatory and `present`; `unavailable` makes it ineligible for on-policy use.
Imported, off-policy, SFT or human traces may legitimately mark it `unavailable` or
`not_applicable`; they never fabricate tokens. Raw model frames, source reasoning and raw tool
arguments remain restricted/quarantined artifacts until rights, privacy and redaction permit a
projection.

The integrity section binds manifest, events, state, verifier and replay artifacts through hashes
and signatures, with signed/Merkle checkpoints anchored to a remote immutable/WORM evidence sink.
Hash links alone are not append-only because an attacker controlling the chain could rewrite it.
It contains broker handles or redacted references, never plaintext secrets. Exact token IDs and
masks should be preserved when available because decode/re-encode can change an on-policy
trajectory; both
[verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) and
[SkyRL's generator interface](https://docs.skyrl.ai/docs/tutorials/skyrl_gym_generator)
expose trainer-facing token IDs/masks.

Immutability applies to the evidence ledger, not perpetual retention of source/PII content. Store
raw blobs and snapshots behind revocable content-addressed indirection with classification and
retention policy. A removal appends a signed tombstone/removal event, revokes the reference and
uses verified deletion or cryptographic erasure where approved; the ledger keeps hashes and the
fact of removal, not undeletable revoked content.

### Import mapping

| Source | Lossless mapping | Missing or special handling |
|---|---|---|
| Toucan | `uuid`, subset, messages, question, available/target tools, assessments and metadata -> source record, events and annotations | no trusted environment/verifier/replay/state-delta evidence; set `attestation_status=unavailable` rather than fabricating it |
| OpenHands SDK | immutable Message/Action/Observation/Error/StateUpdate events -> canonical events; preserve event ID/timestamp/source, `llm_response_id`, `tool_call_id` and condensation `forgotten_event_ids` when present | do not invent generic parent/branch fields; preserve unknown raw fields by quarantined digest, redact secret-bearing state and pin SDK/runtime versions |
| Nebius trajectories | trajectory/instance IDs, messages, patch, exit/resolution and test metrics -> source/task/events/outcome | parse serialized tool arguments into canonical JSON; raw bytes stay quarantined by digest until rights/privacy/redaction admission |
| daVinci env-native | content, source reasoning field, tool calls and tool results -> untrusted source content/events | label reasoning as `restricted_source_content` pending teacher/output-use review; it is not required hidden chain-of-thought or presumed trainable; context-native PR rows are context/transition examples, not verified execution |
| Harbor | job/trial config -> manifest; agent trajectory -> events; verifier output -> evidence | every Harbor agent format needs a versioned importer |
| OpenEnv/NeMo Gym | reset/step/state -> transport adapter | SAMA adds authority, independent attestation, replay, signed verification and rights fields |

Public imports without execution proof may be considered as supervised **candidates only after**
DL/counsel rights, privacy, decontamination and source-class admission. Public availability never
authorizes training; lack of execution proof separately prevents a `verified` label. The
trajectory bundle and training `run_manifest` remain distinct contracts sharing a small
provenance envelope.

---

## 12. Q11 — Sanitized perturbed-trace taxonomy

The shared repository may hold only this high-level, non-inventive contract:

| Category | Examples | Required recovery evidence |
|---|---|---|
| schema/argument | missing/extra field, type drift, malformed/truncated response | validate, repair or fail closed without side effect |
| transport/service | timeout, disconnect, 429, 5xx, partial stream | bounded retry/backoff or alternate path with idempotency |
| auth/authority | expired token, wrong scope, denied approval | request/route authority correctly; never escalate or leak |
| state/concurrency | stale version, duplicate/reordered event, conflicting update | detect version conflict, reconcile or safely abort |
| dependency/environment | unavailable tool, worker loss, quota/resource limit | preserve state, surface platform class and clean up |
| observation integrity | empty/plausible-wrong result, prompt injection, spoofed success | cross-check authoritative state and ignore injected control text |
| user/control | cancellation, changed constraint, approval refusal | stop or replan within declared authority |
| reversibility/compensation | partial side effect, rollback failure | verify rollback/compensation or escalate with preserved evidence |

Each perturbation record contains a seeded fault manifest, trigger, recoverability class, allowed
response, invariant, action/time budget, expected detect/retry/backoff/clarify/rollback/escalate
behavior, required terminal state and secret-leak outcome. Clean and perturbed task IDs remain
paired. Exact injection algorithms, schedules, distributions, prompt designs and scoring weights
stay in the restricted invention vault.

---

## 13. Q12 — Indic and Indian-SaaS environment design

Phase 1 implements **internal synthetic emulators only**, using protocol-neutral internal
re-expression of counsel-approved public material unless the exact schema license/API terms
permit reuse. Public availability alone is not permission. Phase 1 uses no live credentials,
real GSTIN, customer/financial/personal data or external staging endpoint.

| System | Verified public capability | Phase-1 decision | Later prerequisites |
|---|---|---|---|
| TallyPrime | official [API Explorer](https://tallysolutions.com/tallyprime-api-explorer/) is evidence of interactive XML/JSON request/format validation; the local [Tally Connector](https://help.tallysolutions.com/developer-reference/tally-prime-developer-tools/tally-connector/) sends integration requests | protocol-neutral ledger mock with wholly synthetic companies; Explorer is not evidence of an isolated, automatable or fleet-safe execution sandbox | license/EULA, automation and output/training-use review; isolated licensed VM; no real ledger |
| GSTN/e-invoice | the official [GSTN FAQ/API material](https://www.gstn.org.in/faqs-category-details) documents API/testing/GSP flows; the separate [IRP sandbox access](https://einvoice6.gst.gov.in/content/kb/access-to-sandbox/) process requires eligible GSTIN/signatory authorization before dummy test GSTINs | protocol-faithful internal mock | written company authority, GSTIN/signatory and ASP/GSP/IRP terms; broker-held credentials |
| ONDC | official [repositories](https://github.com/ONDC-Official) include protocol and tooling; [onboarding](https://github.com/ONDC-Official/developer-docs/blob/main/registry/Onboarding%20of%20Participants.md) requires FQDN/SSL, whitelisting, signing/encryption keys, callbacks and staged endpoints | internally authored commerce simulator | written authority/onboarding, exact repo/version license review, keys held by broker, synthetic data only |
| Zoho Books | official [sandbox](https://www.zoho.com/books/api/v3/sandbox/) creates an isolated organization copy and is early-access/availability dependent | do not copy a production organization; mock only | empty synthetic org, least OAuth scope, DPA/residency/terms/output-use review, production push disabled |

Before any external sandbox, counsel and IS must confirm purpose and authority, minimization,
retention/deletion lineage, processor/DPA terms, residency/cross-border position, incident response
and exact license/API terms, including the current official
[Digital Personal Data Protection Rules material](https://www.meity.gov.in/documents/act-and-policies/digital-personal-data-protection-rules-2025-gDOxUjMtQWa)
and [commencement notification](https://www.meity.gov.in/static/uploads/2025/11/c56ceae6c383460ca69577428d36828b.pdf).
Do not infer effective dates or lawful basis without counsel. Even with synthetic task payloads,
account identity, authorized-signatory details, IP addresses, OAuth metadata, support logs and
telemetry may be personal data. Record purpose and lawful basis/consent where applicable,
processors/subprocessors, retention/deletion, incident handling and cross-border/residency review.
Availability of a sandbox is not authorization to train on outputs.

---

## 14. Q13 — Milestones, dependencies and effort

The 4–6 person-week P1-70 estimate assumes real parallel capacity: approximately one EA FTE,
0.25–0.5 IS/security, fractional DL and ST support, plus named non-author/domain reviewers. A
single person should move dates; the acceptance criteria should not shrink.

| Week | D2/D3/D4 outcome | Evidence or exit condition |
|---:|---|---|
| 3 | D2 draft and human review packet | SEP/schema draft; source and image shortlist; threat model; isolation/provider decision; verifier sampling plan; budgets/stop rules; HAR-15/HAR-16 interface proposal; named owners/reviewers |
| 5 | D3, only after all holds clear: SEP slice + repo/git adapter + 20 deterministic tasks | three clean resets/replays per task; clean-room hidden verifier; one injected failure; pinned source/image/SBOM/scan/admission evidence; no external/live systems |
| 7 | five adapter families and verifier hardening | at least 20 tasks/family; required adversarial paired fixtures; 100 candidate pre-admission trajectories; failure-injection and one-hour-soak rehearsals; rollback/cleanup evidence; optional 300-negative stretch only if separately staffed/budgeted |
| 8 | D4/F1 gate packet | 20-concurrent one-hour run; at least 99% launch and 95% platform completion; zero isolation/credential/egress/cross-run failure; 100 verified trajectories; required injected cases; measured 200/1,000 growth report |

Earliest dependency chain:

```text
human authority/caps + threat model + source/image approval
  -> versioned SEP/provenance/HAR-15/HAR-16 boundaries
  -> repo adapter qualification
  -> five-family qualification
  -> verifier adversarial QA
  -> 20-way reliability rehearsal
  -> independent human/security review
  -> F1 packet
```

No calendar date is credible until the named allocations and D3 hold-clearing date are recorded.

---

## 15. Q14 — Capacity and Phase-1 budget

### CPU/RAM/disk

The baseline planning profile is 2 vCPU, 4 GiB RAM and 20 GiB ephemeral per active worker.
With 30% scheduler/headroom reserve:

| Concurrency | Active logical quota | Planning capacity with 30% headroom |
|---:|---:|---:|
| 20 | 40 vCPU / 80 GiB RAM / 400 GiB writable | 52 vCPU / 104 GiB / 520 GiB, before failure-domain spare |
| 200 | 400 vCPU / 800 GiB / 4,000 GiB | 520 vCPU / 1,040 GiB (1.016 TiB) / 5,200 GiB |
| 2,000 | 4,000 vCPU / 8,000 GiB / 40,000 GiB | 5,200 vCPU / 10,400 GiB (10.156 TiB) / 52,000 GiB |

Track three different storage ledgers:

1. **logical sandbox quota:** maximum writable bytes promised across active workers;
2. **physical high-water:** actually allocated/consumed ephemeral bytes, node caches and
   filesystem/registry metadata across all domains;
3. **core retained bytes:** authoritative images, quarantine, snapshots/backups, traces, logs and
   gate evidence after compression and replication.

D2 must state whether ephemeral physical backing counts against the program's 50 TB core cap and
then enforce both a logical-admission ceiling and physical/core stop lines. At 2,000, the baseline
headroom quota is already 52,000 GiB (about 55.8 decimal TB), so it exceeds 50 TB if the cap
applies even before registry/cache/logs. Sparse allocation does not remove the need for a physical
high-water stop. A heavy SWE profile of 4 vCPU, 8 GiB and 50 GiB reaches 100,000 GiB of logical
writable quota at 2,000 before headroom. This proves that profile-aware queues, measured
high-water use and an active-set strategy are required; it does not prove the baseline sufficient.

For a self-hosted Phase-1 alternative that preserves 20 occupied slots after either worker domain
fails, quote **each** of two domains at no less than 64 vCPU, 128 GiB RAM and 2 TB local NVMe,
plus platform overhead—at least 128 vCPU, 256 GiB and 4 TB raw local in total. Put the
authoritative replicated registry/evidence sink outside disposable node cache and size its
replication separately. Replace this floor with per-family p50/p95 CPU, RAM, logical quota,
physical high-water, startup and duration measurements. Use Little's-law capacity planning:

```text
required_concurrency ~= arrival_rate * mean_duration / target_utilization
```

The general “16–32 CPU cores per training GPU” heuristic is not a Phase-1 sizing rule. This
lane is CPU-bound and decoupled from GPU count; measure the environment workload and inference
service separately.

### Proposed sub-budget and stop rules

**PROPOSAL only; none of this is spend authority:**

| Line | Planning hard line | Stop rule |
|---|---:|---|
| managed sandbox usage/plan | $2,500 | stop before paid tier/enterprise commitment without owner approval; alert at 50%, review at 75% |
| registry/logging/monitoring | $1,000 | no mirror expansion; reduce retention only under evidence policy |
| contingency | $1,500 | owner release after documented failure/quote |
| **external environment-lane total** | **$5,000** | any overrun or custom contract needs a new written decision |
| storage | **4 TB Phase-1 physical high-water**, inclusive of replicas/metadata/snapshots/caches | stop acquisition at 75%; preserve gate evidence and metadata first |
| GPU inference | **30 GPU-hours planned, 50 maximum** | CPU-only harness first; smallest model/service diagnostic; GPU work needs the program's normal run authorization |

This lane budget sits inside pending global caps; it does not reserve them. Before any spend,
record a provider quote, exact data classification, retention/region terms and owner approval.

---

## 16. Q15 — EA-lane risk additions

Scores use program probability x impact on a 1–5 scale. Owners are human roles, not AI sessions.

| ID | Risk / early warning | Current | Target | Human owner(s) | Mitigation and stop condition |
|---|---|---:|---:|---|---|
| R19 | verifier false accept/reward hacking; any known-bad fixture passes | 20 | <=5 | EA | quarantine verifier; no reward/admission; root cause, new digest, repeat adversarial QA |
| R20 | source/image supply-chain compromise; mutable digest or scan drift | 15 | <=4 | DL, IS | stop acquisition/execution; quarantine, regenerate SBOM/scans, reapprove exact artifact |
| R21 | fleet availability below F1 target or orphan rate rises | 12 | <=4 | EA, IS | close admission, reconcile/reap, reduce concurrency, preserve platform-error accounting |
| R22 | provider IP/residency/retention mismatch | 15 | <=4 | IS, DL | public/synthetic inputs only until signed terms/BYOC; do not transmit hidden tests or proprietary traces |
| R23 | semantic replay divergence or flaky verifier | 20 | <=5 | EA | quarantine task/version on first deterministic mismatch; no averaging |
| R24 | image/cache/storage blowout | 12 | <=4 | EA, DL | 75% stop line, metadata-only discovery, measured active set, separate mirror decision |
| R25 | hidden eval or generator-family contamination | 20 | <=5 | EA, DL | freeze fingerprints before generation; invalidate and remove affected derivatives |
| R26 | API terms, PII or output-use rights unresolved | 20 | <=5 | DL | synthetic emulator only; no acquisition/external call/admission |
| R27 | framework/API churn breaks canonical evidence | 12 | <=4 | EA, ST | SEP-owned contract, exact pins, compatibility suite, no framework-specific canonical state |
| R28 | authority broker/control plane becomes confused deputy | 15 | <=4 | IS, EA | independent capability checks; wrong-scope/replay/SSRF tests; kill and revoke on failure |
| R29 | staffing estimate hides security/reviewer load | 20 | <=8 | program owner | name allocations or move schedule; never substitute AI peer QA for human review |
| R30 | audit/log pipeline leaks secrets or hidden tests | 15 | <=4 | IS | schema-level redaction, broker handles only, canary scans, restricted retention and access |
| R31 | one fleet creates 2,000-way blast radius | 12 | <=4 | IS, EA | cell architecture, per-cell limits/circuit breakers and tested global kill switch |

Existing R15 remains the top stop condition: any environment escape, credential exposure,
unauthorized egress or cross-run leak stops the fleet, preserves evidence, revokes/rotates
capabilities and requires security review.

---

## 17. D2 decision packet

The following are recommendations to turn into a human-signed ADR; they are not decisions yet.

| Decision | Recommended D2 answer | Required evidence before sign-off |
|---|---|---|
| canonical protocol | SAMA-owned SEP + TrajectoryBundle; thin external adapters | API/schema review, ownership, compatibility test plan |
| D3 first adapter | Harbor-backed repo/git slice behind SEP | exact commit/license/SBOM/scans and 20 approved tasks |
| isolation | production microVM integration (Kata or reviewed Firecracker+jailer path) for hostile repo/terminal; gVisor only after family parity | IS threat-model sign-off, host/runtime compatibility and adversarial containment results |
| F1 hosting | managed public/synthetic or company-controlled BYOC/on-prem for proprietary material | quote, region/DPA/retention/deletion/no-training terms |
| trainer | defer; bake off verl/SkyRL/NeMo-RL after SEP proof | exact token/mask round-trip, cancellation/retry and replay tests |
| images | no 2,000 mirror; F1 task images plus small approved warm set | measured layer census and row/source approval |
| MCP/data | official-registry metadata discovery only; mock-first synthetic path | DL state-machine enforcement and per-artifact review |
| verifier target | F1 operational proof; later statistically powered promotion gate | sampling plan, reviewer allocation and target scope |
| Indic systems | internal synthetic mocks only in F1 | separate written authority for any external staging |
| Phase-1 lane caps | $5k external, 4 TB physical storage high-water, 30 planned/50 max GPU-hours | owner signature, replica accounting and program-cap reconciliation |

### Open items that block D3

1. Named human EA, IS/security reviewer and qualified non-author reviewer.
2. Signed authority, program caps, lane budget and stop rules.
3. Private `sama-7b` remote, protected main/CI and isolated feature workflow.
4. Approved SEP, trajectory/provenance, HAR-15 admission and HAR-16 fingerprint interfaces.
5. Approved minimal task/source/image manifest and row/artifact-level rights decisions.
6. Provider or on-prem isolation choice with region, retention, deletion and threat-model evidence.
7. Exact verifier statistical target scope and human sampling plan.
8. Measured family profiles and vendor enterprise quotes; no claim of final cost until they exist.

---

## 18. Acceptance traceability

| Delegation question | Memo answer |
|---:|---|
| 1. framework/trainer | §2 |
| 2. fleet and cost | §3 |
| 3. images/storage/GC | §4 |
| 4. determinism/replay/state delta | §5 |
| 5. concurrency/authority | §6 |
| 6. verifiers/hidden tests/quality | §7 |
| 7. anti-hacking | §8 |
| 8. nondeterminism/pass-k/flakes | §9 |
| 9. MCP/data engine/license | §10 |
| 10. trajectory schema mapping | §11 |
| 11. perturbation taxonomy | §12 |
| 12. Indic/Indian-SaaS | §13 |
| 13. milestones/effort | §14 |
| 14. CPU/storage/spend | §15 |
| 15. risks | §16 |

---

## 19. Primary-source register

The most decision-relevant sources are linked inline. This compact register separates them by
claim family:

- **Program:** `PHASE_1_G0_S0_EXECUTION_PLAN.md` P1-20, P1-70, §§10–12;
  `PHASE_1_PLAN.md` T8; `FINAL_PROGRAM_PLAN.md` §§7–9.
- **Frameworks:** [OpenEnv](https://github.com/huggingface/OpenEnv),
  [Harbor](https://github.com/harbor-framework/harbor),
  [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym),
  [verl](https://github.com/verl-project/verl),
  [SkyRL](https://github.com/NovaSky-AI/SkyRL),
  [NeMo-RL](https://github.com/NVIDIA-NeMo/RL),
  [OpenHands SDK](https://github.com/OpenHands/software-agent-sdk),
  [SandboxFusion](https://github.com/bytedance/SandboxFusion).
- **Scale/method:** [DeepSWE](https://www.together.ai/blog/deepswe),
  [Kimi K2 paper](https://arxiv.org/html/2507.20534v1).
- **Security:** [Kubernetes multi-tenancy](https://kubernetes.io/docs/concepts/security/multi-tenancy/),
  [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/),
  [NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/),
  [Firecracker design](https://github.com/firecracker-microvm/firecracker/blob/main/docs/design.md),
  [gVisor security](https://gvisor.dev/docs/architecture_guide/security/).
- **Costs/residency:** [E2B pricing](https://e2b.dev/pricing) and
  [BYOC](https://e2b.dev/docs/byoc);
  [Daytona pricing](https://www.daytona.io/pricing) and
  [regions](https://www.daytona.io/docs/en/regions/);
  [Modal pricing](https://modal.com/pricing) and
  [region selection](https://modal.com/docs/guide/region-selection);
  [Prime Sandboxes](https://docs.primeintellect.ai/sandboxes/overview).
- **Data and licenses:** [official MCP Registry](https://registry.modelcontextprotocol.io/),
  [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M),
  [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench),
  [daVinci-Dev](https://github.com/GAIR-NLP/daVinci-Dev),
  [R2E-Gym](https://github.com/agentica-project/R2E-Gym),
  [tau2-bench](https://github.com/sierra-research/tau2-bench).

---

## 20. D1 conclusion

D1 supports moving to D2 with one major architectural change from the starting brief: **SAMA
must own the protocol, evidence, authority and provenance semantics, while open frameworks remain
replaceable execution/trainer adapters.** It does not support starting implementation yet.

The safe thin path is: approve the trust/data boundaries, sign SEP and the trajectory bundle,
admit a very small repo-task/image set, prove one adapter, then expand to the five F1 families.
The moat is the verified state-and-recovery corpus and its evidence chain—not the number of
downloaded MCP servers, mirrored images or simultaneously launched containers.
