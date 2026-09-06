# ExpertForge Synthesis Architecture Pattern Register

**Document class:** Living architecture pattern register  
**Project:** ExpertForge Synthesis  
**Register version:** 1.0  
**Prepared:** 2026-09-06  
**Status:** Non-normative companion to implementation plans and experiment contracts.  
**Purpose:** Preserve, characterize, compare, test, adopt, defer, reject, or supersede reusable architecture patterns for extracting expertise from heterogeneous source models and consolidating it into new standalone models under constrained compute and memory.

---

# 1. Governing Rule and Planning Position

ExpertForge Synthesis maintains two deliberately separate surfaces:

1. **Authoritative implementation and experiment planning** — approved objectives, frozen acceptance criteria, resource envelopes, experiment contracts, validators, execution protocols, and implementation work.
2. **Architecture Pattern Register (APR)** — reusable architectural ideas and negative lessons that may be observed, characterized, compared, trialed, accepted, deferred, rejected, or superseded.

The APR is **not** an implementation plan, roadmap, backlog, release gate, acceptance contract, benchmark result, experiment authorization, or product-claim authority.

> **Canonical distinction:** Observation is not adoption. Adoption is not implementation. Implementation is not verification. Verification is not generalization.

## 1.1 Planning-method rule

Research ideas are registered before they are inserted into an active implementation phase unless the active authorized objective already covers them or an explicit project decision authorizes a bounded correction or trial.

A pattern may be useful, attractive, repeatedly observed, or accepted as a project principle without thereby creating implementation work.

## 1.2 Repository naming rule

Repository-facing material must not name external products or projects.

Accordingly:

- canonical pattern definitions use project-owned or vendor-neutral language;
- source provenance for external research uses opaque source references;
- the mapping from an opaque source reference to an external source identity is maintained outside this repository;
- repository artifacts may retain source class, revision/date metadata, content digests, and evidence boundaries without carrying external product/project names;
- no opaque reference implies adoption, endorsement, compatibility, or implementation.

This rule intentionally modifies the normal source-ledger presentation while preserving the separation between source evidence and project decisions.

## 1.3 Why this register exists

The register prevents recurring failure modes:

- **Research-to-roadmap collapse:** an attractive mechanism silently becomes mandatory work.
- **Vocabulary contamination:** external terminology becomes project architecture by repetition.
- **Mechanism fixation:** the project commits to one transfer or runtime mechanism before proving the relevant bottleneck.
- **Capacity/throughput confusion:** making a model executable is mistaken for making it useful or fast.
- **Evidence inflation:** architectural similarity is treated as proof of expertise transfer.
- **Negative-result amnesia:** failed mechanisms are repeatedly rediscovered.
- **Parameter-count theater:** total parameter count is treated as proof of useful capability.
- **Hardware overfitting:** architecture is keyed to one device name rather than an execution/resource surface.

## 1.4 What belongs here

Record a pattern when at least one is true:

- a research study demonstrates a reusable representation-transfer or sparse-runtime mechanism;
- a project experiment exposes a reusable design rule or failure boundary;
- a hardware limitation suggests a reusable execution, caching, paging, or scheduling pattern;
- a negative result should constrain future search;
- multiple studies converge on the same architectural invariant;
- an idea is strategically useful but should not expand the current experiment;
- a future dense or sparse synthesized-model direction should be retained without scheduling it.

## 1.5 What does not belong here

Do not use the APR for:

- ordinary bugs or task tracking;
- one-off implementation notes;
- temporary benchmark values that belong in evidence artifacts;
- secrets, credentials, or machine-specific identifiers;
- retroactive acceptance-criteria changes;
- unapproved execution commands;
- product claims;
- external product/project names;
- decisions that belong in an authoritative plan, experiment contract, or implementation decision record.

## 1.6 Promotion rule

A registered pattern may be promoted only when a concrete project forcing function exists, such as:

- a measured transfer-quality deficit;
- a measured memory, I/O, compute, or storage bottleneck;
- a source-model family that cannot be handled cleanly by the current runtime;
- a second heterogeneous source model that requires a common expertise contract;
- a student architecture that cannot consume existing expertise artifacts safely;
- evidence showing that a simpler baseline is materially inferior;
- an explicit model-capacity or active-compute objective.

“An external system does this” is never sufficient.

---

# 2. Project Goal and Research Boundary

The project hypothesis is:

> Heterogeneous source models can be used sequentially as expertise sources; source-specific Expertise Adapters can extract portable representations; those representations can be persisted, composed, and consolidated into a new standalone model while keeping accelerator residency bounded.

The intended high-level flow is:

```text
Source Model Pool
        ↓
Source Selection
        ↓
Bounded-Working-Set Expert Runtime
        ↓
Expertise Extraction
        ↓
Source-Specific Expertise Adapter
        ↓
Canonical Expertise Representation
        ↓
Expertise Artifact
        ↓
Expertise Consolidation
        ↓
Synthesized Model
```

The project does **not** assume that latent transfer is superior to simpler supervision. Text targets, token distributions, labels, and other conventional supervision remain mandatory comparators where applicable.

The project also does **not** assume that a large total sparse parameter count implies useful specialization. Useful capacity must be demonstrated.

---

# 3. Mandatory Five-Layer Traceability Structure

Every maintained APR record separates evidence from project decisions through five semantic layers.

## 3.1 Pattern definition

Describe the reusable problem and solution in vendor-neutral architectural language.

The definition should remain understandable if provenance is hidden.

## 3.2 Source provenance

Record where the idea came from without naming an external product/project in the repository.

Allowed provenance kinds:

```text
external
project-native
synthesis
methodology
```

For external sources, use:

```text
source_ref: EXT-YYYY-NNN
source_class: <paper | repository | benchmark | architecture-study | other>
version_or_revision: <known revision or null>
observed_date: YYYY-MM-DD
content_digest: <digest or null>
evidence_boundary: <what was actually inspected>
```

The identity behind `source_ref` is retained outside the repository.

## 3.3 Source observation

State what the cited source actually implements, measures, claims, or demonstrates.

Do not:

- generalize beyond the evidence;
- treat source capability as project evidence;
- attribute project adaptations back to the source;
- infer missing revisions or measurements.

## 3.4 ExpertForge Synthesis interpretation

Restate the pattern under this project's own architecture, resource constraints, evidence rules, and model-transfer objectives.

This layer owns:

- adaptation;
- must-preserve constraints;
- conflicts;
- adoption trigger;
- required evidence;
- relevant timescale;
- dense/sparse implications;
- source/runtime/student implications;
- related patterns.

## 3.5 Project decision log

Record project decisions only:

- pattern status;
- implementation status;
- planning disposition;
- adoption/trial/rejection/supersession decision;
- authority/date;
- rationale;
- implementation links.

Source maturity is evidence, not a project decision.

The information flow is:

```text
external evidence / project experiment
              ↓
       source observation
              ↓
 vendor-neutral pattern definition
              ↓
 project interpretation
              ↓
       project decision
              ↓
optional explicit plan/contract linkage
              ↓
      implementation + real proof
```

---

# 4. Status Model

Pattern status and implementation status are intentionally independent.

## 4.1 Pattern status

| Status | Meaning |
|---|---|
| `OBSERVED` | Captured but not sufficiently understood for project comparison. |
| `CHARACTERIZED` | Mechanism, forces, liabilities, failure modes, and likely project interaction are understood well enough for comparison. |
| `CANDIDATE` | Plausibly useful when a concrete forcing function exists. |
| `ACCEPTED` | Accepted as a project architectural principle or mechanism. This does not schedule implementation. |
| `TRIAL-AUTHORIZED` | A bounded experiment/spike is explicitly authorized without full adoption. |
| `DEFERRED` | Useful or potentially useful, but intentionally outside the current planning window or blocked by prerequisites. |
| `REJECTED` | Deliberately not adopted in the stated form. |
| `SUPERSEDED` | Replaced by a stronger pattern or project decision. |

## 4.2 Implementation status

| Status | Meaning |
|---|---|
| `NOT-LINKED` | No authoritative implementation unit currently references it. |
| `LINKED` | An approved plan, experiment contract, issue, or implementation mapping references it. |
| `IN-TRIAL` | A bounded implementation/qualification experiment is active. |
| `IMPLEMENTED` | A concrete mechanism embodies the pattern. |
| `VERIFIED` | Required executable evidence has passed. |
| `ROLLED-BACK` | A prior implementation was removed; the architectural record remains historical evidence. |

## 4.3 Planning disposition

Planning disposition answers **when to reconsider**, not **what to build**:

- `current-plan-authorized`
- `future-plan-candidate`
- `research-only`
- `do-not-promote`

---

# 5. Explicit Adoption / Promotion Record

A pattern becomes binding only when a separate project decision identifies its exact effect.

```yaml
pattern_id: APR-XXX
decision: accepted | trial-authorized | rejected | superseded
scope: <exact subsystem / experiment / model objective>
authority: <approved plan / experiment contract / explicit project decision>
decision_date: YYYY-MM-DD
implementation_link: <plan section / issue / change / null>
acceptance_effect: none | <explicit acceptance-contract change>
claim_effect: none | <explicit claim implications>
rationale: <project rationale>
```

Promotion must establish:

1. **Problem match** — a concrete project design pressure exists.
2. **Resource-model match** — memory, storage, compute, bandwidth, and runtime assumptions fit the target execution surface.
3. **Representation match** — the pattern declares what information is preserved, compressed, transformed, or discarded.
4. **Semantics match** — exact, target-equivalent, quality-bounded, or model-altering behavior is explicit.
5. **Scope match** — exact source model class, adapter, student, workload, runtime, and representation are bounded.
6. **Failure model** — stale artifacts, incompatible adapter versions, OOM, paging failure, partial extraction, routing error, and fallback behavior are defined where relevant.
7. **Evidence plan** — the adaptation can be measured through real composed execution.
8. **Authorization** — a separate authoritative planning surface links the pattern.

If this trace does not exist, the APR remains non-executing.

---

# 6. Normalized Pattern Record Schema

```yaml
id: APR-XXX
name: <vendor-neutral pattern name>
category: <planning | representation | extraction | runtime | resource | routing | scheduling | consolidation | sparse-model | evidence | other>

pattern_definition:
  problem: ...
  definition: ...
  benefits: []
  liabilities: []
  failure_modes: []

source_provenance:
  kind: external | project-native | synthesis | methodology
  source_ref: <opaque reference or null>
  source_class: <class or null>
  version_or_revision: <known revision or null>
  observed_date: <date or null>
  content_digest: <digest or null>
  evidence_boundary: ...

source_observation:
  summary: ...
  evidence_notes: []

project_interpretation:
  adaptation: ...
  must_preserve: []
  conflicts: []
  adoption_trigger: ...
  required_evidence: []
  relevant_timescale: <offline | batch | request | token-layer | training-step | multi>
  related_patterns: []

decision_log:
  pattern_status: OBSERVED | CHARACTERIZED | CANDIDATE | ACCEPTED | TRIAL-AUTHORIZED | DEFERRED | REJECTED | SUPERSEDED
  implementation_status: NOT-LINKED | LINKED | IN-TRIAL | IMPLEMENTED | VERIFIED | ROLLED-BACK
  planning_disposition: current-plan-authorized | future-plan-candidate | research-only | do-not-promote
  decision: none | accepted | trial-authorized | rejected | superseded
  authority: <project authority or null>
  decision_date: <date or null>
  rationale: ...
  links: []
```

---

# 7. Pattern Assessment Rubric

Do not collapse architectural fit into one weighted score. Assess material dimensions independently.

## 7.1 Expertise-transfer fit

- What information is expected to transfer?
- Is the signal lexical, semantic, relational, structural, probabilistic, or task-specific?
- Does the source representation contain information not already available through simpler supervision?
- How will the project prove that the student actually acquired the claimed capability?

## 7.2 Representation fit

- What is the native hidden width, sequence representation, tokenizer boundary, and source layer?
- Does equal dimensionality hide semantic mismatch?
- Is the adapter output canonical, versioned, and independently interpretable?
- What exact information is compressed or dropped?

## 7.3 Resource fit

- Which resource is constrained: accelerator memory, host memory, storage, bus bandwidth, compute, or wall-clock time?
- Does the mechanism improve capacity, throughput, or both?
- What resource becomes dominant after the mechanism succeeds?

## 7.4 Execution fit

- Is the source model dense or sparse?
- Does the runtime require resident execution, host offload, layer paging, or routed-expert paging?
- Can movement be overlapped with useful compute?
- What is the minimum active working set?

## 7.5 Artifact fit

- Can the expertise output be reused without loading the source again?
- Is source identity, adapter identity, extraction configuration, and data identity reconstructible?
- Can incompatible artifacts fail closed rather than silently mix representations?

## 7.6 Consolidation fit

- Is the student dense or sparse?
- Does the student learn through a canonical alignment head, output-level supervision, or both?
- What prevents catastrophic interference among capabilities?
- Can one source dominate the consolidated objective?

## 7.7 Sparse-model fit

- What are total parameters and active parameters?
- How many internal experts are selected per token?
- Are duplicated experts actually specialized?
- What are storage, routing, paging, and cache costs?
- Is total capacity useful or merely inactive copied weight?

## 7.8 Evidence fit

- What baseline is required?
- What ablation proves the adapter adds value?
- Are performance, memory, quality, and capability claims measured separately?
- Can negative results be retained without forcing a mechanism?

## 7.9 Operational fit

- Does the mechanism add persistent caches, background workers, calibration, prefetch, JIT state, or conversion steps?
- Can it fail back to a simpler baseline?
- Can the project resume or reproduce a partial extraction safely?

## 7.10 Scale fit

- Has the mechanism been demonstrated at a smaller scale before escalation?
- Does the scaling step change the bottleneck regime?
- Is the next scale justified by evidence rather than parameter-count aspiration?

---

# 8. Baseline Project Constraints

Patterns are assessed against these project constraints unless an explicit decision changes them:

- The target environment may have substantially less accelerator memory than source-model size.
- Only the minimum required source-model working set should occupy accelerator memory.
- Multiple source models do not need simultaneous residency.
- Source execution and synthesized-model training may be separate phases.
- Expertise artifacts should be reusable so source execution is not repeated unnecessarily.
- Heterogeneous hidden states are not assumed to be directly compatible.
- Exact lexical/entity information must not be silently destroyed by continuous compression.
- Equal tensor shape does not imply semantic alignment.
- Dense and sparse source models require different paging strategies.
- Total parameter count and active parameter count are reported separately for sparse models.
- Streaming/offload solves capacity constraints but does not eliminate compute or I/O cost.
- Simpler supervision remains a required comparator.
- A transfer claim requires evidence on the synthesized model, not merely source-model quality.
- Negative results remain valid project outputs.
- The project may conclude that an adapter is not justified.
- Repository-facing architecture vocabulary remains project-owned or vendor-neutral.

---

# 9. Initial Project Pattern Register

## APR-001 — Architecture Pattern Register as a Non-Binding Planning Sidecar

**Category:** planning  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `IMPLEMENTED`  
**Planning disposition:** `current-plan-authorized`

### Pattern definition

**Problem.** Research produces useful mechanisms faster than the project can safely validate them.

**Definition.** Maintain a living architecture pattern register alongside implementation plans. Research ideas enter the APR first. Only a separate explicit adoption or trial decision may change an implementation phase, experiment contract, acceptance boundary, or model objective.

**Benefits.**
- Separates learning from commitment.
- Preserves deferred ideas.
- Prevents research enthusiasm from changing scope silently.
- Retains rejected mechanisms as architectural memory.

**Liabilities.**
- Can become a shadow backlog.
- Can become stale.
- Can create process overhead if consulted for ordinary bounded implementation.

**Failure modes.**
- `CANDIDATE` is interpreted as “must build later.”
- A deep dive silently creates implementation work.
- A closed experiment is reopened because a new mechanism appears attractive.

### Source provenance

**Kind:** methodology  
**Source ref:** `METHOD-2026-001`  
**Observed:** 2026-09-06  
**Evidence boundary:** Supplied comparison architecture registers were reviewed for register mechanics, status separation, promotion rules, intake flow, anti-pattern retention, maintenance discipline, and deep-dive procedure. External identities are intentionally not retained in this repository.

### Source observation

The reviewed methodology consistently separates observation, adoption, implementation, and verification; preserves stable IDs and negative lessons; and requires explicit promotion before implementation scope changes.

### Project interpretation

Adopt the register as project architectural memory. It does not become an execution authority.

### Decision log

- **Decision:** accepted.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.
- **Acceptance effect:** none.
- **Claim effect:** none.

---

## APR-002 — Expertise Extraction and Consolidation as Separate Phases

**Category:** extraction, consolidation  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Simultaneously loading source models and a trainable synthesized model multiplies accelerator-memory pressure and tightly couples expensive source execution to every student step.

**Definition.** Separate source-model expertise extraction from synthesized-model consolidation. Source models may run, produce durable expertise artifacts, and be fully unloaded before student training begins.

**Benefits.**
- Bounds peak accelerator residency.
- Allows expensive source execution to be performed once.
- Enables repeated student experiments over the same extracted evidence.
- Decouples source runtime engineering from student-training engineering.

**Liabilities.**
- Requires artifact storage and versioning.
- Offline extraction can make stale or poor adapters expensive to correct.
- Some online co-training objectives become harder.

**Failure modes.**
- Student training silently depends on an unrecorded source state.
- Extraction artifacts cannot be reproduced.
- Source execution is repeated unnecessarily.

### Source provenance

**Kind:** project-native  
**Source ref:** `DESIGN-2026-001`  
**Observed:** 2026-09-06

### Project interpretation

This is the default topology unless a later trial demonstrates a material need for simultaneous teacher/student execution.

### Decision log

- **Decision:** accepted as a project architecture principle.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.
- **Implementation status:** `NOT-LINKED`.

---

## APR-003 — Bounded Working-Set Source Execution

**Category:** runtime, resource  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Source-model size may exceed available accelerator memory by a large factor.

**Definition.** Require only the minimum execution working set to be accelerator-resident at any instant. The complete source checkpoint may reside across slower tiers.

**Benefits.**
- Decouples source-model total size from accelerator capacity.
- Makes large source models usable for offline extraction.
- Provides one invariant across resident, offloaded, and paged execution.

**Liabilities.**
- Movement can dominate wall-clock time.
- Requires careful scheduling, staging, and workspace accounting.
- Capacity feasibility does not imply interactive performance.

**Failure modes.**
- “Fits through paging” is reported as “runs efficiently.”
- Temporary buffers exceed the planned working set.
- I/O stalls erase accelerator utilization.

### Source provenance

**Kind:** synthesis  
**Source ref:** `DESIGN-2026-002`  
**Observed:** 2026-09-06

### Project interpretation

The core runtime invariant is:

```text
resident_source_working_set << total_source_parameters
```

where practical. Throughput must be measured separately.

### Decision log

- **Decision:** accepted as a project resource principle.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-004 — Canonical Expertise Artifact with Immutable Lineage

**Category:** representation, evidence  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Extracted expertise is scientifically unusable if the exact source, adapter, extraction configuration, input, and representation cannot be reconstructed.

**Definition.** Persist extracted expertise as a versioned artifact whose identity binds the source-model revision, source representation tap, adapter revision, extraction configuration, data identity, output representation schema, and content digest.

**Benefits.**
- Reusable source output.
- Strong provenance.
- Student experiments can be reproduced without rerunning sources.
- Incompatible representations can fail closed.

**Liabilities.**
- Metadata and storage overhead.
- Schema migrations require discipline.

**Failure modes.**
- Artifact from one adapter version is consumed as another.
- Source model changes without invalidating derived expertise.
- Partial artifact is treated as complete.

### Source provenance

**Kind:** project-native  
**Source ref:** `DESIGN-2026-003`  
**Observed:** 2026-09-06

### Project interpretation

Expertise Artifact is canonical evidence for extraction output, not proof that the student acquired the capability.

### Decision log

- **Decision:** accepted as a project evidence principle.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-005 — Simpler Supervision as a First-Class Comparator

**Category:** evidence  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `current-plan-authorized`

### Pattern definition

**Problem.** A sophisticated latent adapter can appear valuable merely because it is compared against no simpler transfer method.

**Definition.** Treat simpler supervision paths as mandatory comparators where applicable: target text, labels, token distributions, or other output-level supervision.

**Benefits.**
- Establishes whether the adapter earns its complexity.
- Prevents architecture appeal from substituting for evidence.
- Creates a safe baseline if latent transfer fails.

**Liabilities.**
- Adds experiment cost.
- Some information classes may not have an exact simpler equivalent.

**Failure modes.**
- Adapter is retained because it is novel rather than useful.
- Source-model quality is mistaken for transfer quality.
- No-ablation result is promoted as evidence of latent benefit.

### Source provenance

**Kind:** synthesis  
**Source ref:** `DESIGN-2026-004`  
**Observed:** 2026-09-06

### Project interpretation

A source-side adapter remains a hypothesis until it demonstrates useful transfer beyond the relevant baseline.

### Decision log

- **Decision:** accepted as evidence methodology.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-010 — Source Model and Internal Expert Are Distinct Concepts

**Category:** sparse-model, terminology  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `current-plan-authorized`

### Pattern definition

**Problem.** The word “expert” can refer both to a complete pretrained source model and to a routed specialist module inside a sparse synthesized model.

**Definition.** Use **Source Model** for an external complete model used to provide expertise, and **Internal Expert** for a routed parameter block inside a sparse model.

**Benefits.**
- Prevents topology ambiguity.
- Clarifies runtime and artifact ownership.
- Makes total/active parameter accounting easier to reason about.

**Liabilities.**
- Requires terminology migration in older notes.

**Failure modes.**
- Source routing is confused with token-level internal routing.
- A whole-model checkpoint is treated like an internal expert shard.

### Source provenance

**Kind:** project-native  
**Source ref:** `DESIGN-2026-005`  
**Observed:** 2026-09-06

### Decision log

- **Decision:** accepted terminology.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-011 — Source-Specific Expertise Adapter into a Canonical Space

**Category:** representation, extraction  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Hidden states from heterogeneous source models differ in dimensionality, tokenization, coordinate system, layer semantics, scale, and feature organization. Equal tensor shape does not create semantic compatibility.

**Definition.** Give each source family a source-specific **Expertise Adapter** that maps selected source representations into a versioned Canonical Expertise Space.

```text
Source Representation H_i
        ↓
Expertise Adapter A_i
        ↓
Canonical Expertise Z_i
```

**Benefits.**
- Isolates source heterogeneity.
- Gives the synthesized model one target contract.
- Enables sequential multi-source acquisition.
- Supports compression at the extraction boundary.

**Liabilities.**
- Adapter training is itself a research problem.
- A poorly learned canonical space can destroy useful source information.
- Source-specific adapters add maintenance and versioning cost.

**Failure modes.**
- Linear dimension matching is mistaken for semantic alignment.
- Two adapters produce numerically compatible but semantically incompatible artifacts.
- Canonical-space drift invalidates prior artifacts silently.

### Source provenance

**Kind:** synthesis  
**Source ref:** `EXT-2026-001`  
**Source class:** architecture-study  
**Observed:** 2026-09-06  
**Evidence boundary:** Heterogeneous latent-transfer mechanisms and entity-grounding failure were studied; external source identity is intentionally maintained outside the repository.

### Project interpretation

The adapter exists to make heterogeneous internal representations portable, not to keep multiple source models resident or to create live source-to-source communication.

### Required evidence

- adapter alignment metrics;
- downstream student transfer;
- comparison against output-level baselines;
- entity/identifier fidelity tests;
- compatibility/version rejection tests.

### Decision log

- **Decision:** accepted as the core extraction abstraction.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-012 — Typed Expertise Representation

**Category:** representation  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `research-only`

### Pattern definition

**Problem.** Forcing all transferable information through one undifferentiated continuous vector may lose exact symbolic identity or hide confidence and provenance.

**Definition.** Permit the canonical expertise contract to contain typed channels rather than one latent tensor.

Candidate channels include:

```text
semantic latent state
relational / structural state
lexical or entity anchors
confidence / uncertainty
task or capability metadata
```

**Benefits.**
- Preserves exact identifiers separately from compressed semantics.
- Makes ablations possible by information type.
- Supports selective student objectives.

**Liabilities.**
- More complex artifact schema.
- Channel semantics can become vague.
- Some channels may duplicate information.

**Failure modes.**
- Channel proliferation without evidence.
- Lexical anchors become a hidden text copy of the entire input.
- Confidence values are uncalibrated.

### Source provenance

**Kind:** synthesis  
**Source ref:** `EXT-2026-001`

### Adoption trigger

A single latent channel demonstrably loses information that materially affects student transfer.

### Decision log

- **Decision:** none.
- **Rationale:** promising but requires controlled ablation.

---

## APR-013 — Student-Side Alignment Adapter

**Category:** consolidation, representation  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** A synthesized model's native hidden state does not automatically inhabit the Canonical Expertise Space.

**Definition.** Use a Student Adapter to project selected student states into the canonical space for alignment loss, while the student's main language/modeling objective remains independently measurable.

**Benefits.**
- Decouples canonical expertise dimensions from student hidden width.
- Allows multiple student architectures to reuse the same expertise artifacts.
- Makes representation loss explicit.

**Liabilities.**
- Student can learn to satisfy the adapter without internalizing useful capability.
- Adds objective-weight tuning.

**Failure modes.**
- Low latent loss with no behavioral gain.
- Adapter head absorbs the signal while the backbone does not.
- Student adapter version changes without artifact compatibility checks.

### Adoption trigger

Canonical-space supervision is selected for a student experiment.

### Required evidence

Behavioral transfer must improve, not merely representation similarity.

### Decision log

- **Decision:** none.

---

## APR-014 — Sequential Multi-Source Expertise Composition

**Category:** extraction, resource  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Multiple source models may contribute complementary information while accelerator memory cannot host them simultaneously.

**Definition.** Execute source models sequentially, persist each canonical expertise result, and compose their artifacts afterward. Multi-source expertise does not require multi-source co-residency.

```text
Source A → Adapter A → Z_A → unload
Source B → Adapter B → Z_B → unload
Source C → Adapter C → Z_C → unload
                       ↓
                  synthesis
```

**Benefits.**
- Keeps memory bounded.
- Allows arbitrarily heterogeneous source runtimes.
- Enables same-sample multi-source composition without simultaneous residency.

**Liabilities.**
- Higher wall-clock latency for multi-source samples.
- Artifact synchronization and compatibility become important.

**Failure modes.**
- Artifacts are combined from incompatible canonical-space versions.
- Duplicate source evidence is overweighted.
- Sequential order accidentally changes canonical semantics.

### Source provenance

**Kind:** project-native  
**Source ref:** `DESIGN-2026-006`

### Decision log

- **Decision:** accepted architecture principle.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-015 — Capability-Guided Source Selection

**Category:** routing  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Running every source model on every sample wastes compute and can inject irrelevant supervision.

**Definition.** Select one source model, or a small sequential set, using capability evidence, sample characteristics, and measured marginal utility to the student.

**Benefits.**
- Reduces extraction cost.
- Encourages specialization.
- Makes one-source-at-a-time execution natural.

**Liabilities.**
- Router errors can hide useful expertise.
- Capability labels may be too coarse.
- Learned routing can collapse to one dominant source.

**Failure modes.**
- Static benchmark score is treated as per-sample expertise.
- Router becomes more expensive than the extraction it saves.
- Source utilization is artificially balanced despite one source being genuinely superior.

### Adoption trigger

Two or more source models are available and extraction cost is material.

### Required evidence

Per-source marginal student utility, routing accuracy, utilization distribution, and fallback behavior.

### Decision log

- **Decision:** none.

---

## APR-016 — Runtime Strategy behind a Common Source-Execution Interface

**Category:** runtime  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Small, medium, and oversized source models require different execution strategies, but extraction logic should not depend on where weights reside.

**Definition.** Expose one source-execution contract with multiple physical strategies:

```text
resident
host-offloaded
paged-dense
paged-sparse
```

The extraction layer consumes equivalent source representations regardless of physical placement.

**Benefits.**
- Decouples adapter logic from hardware placement.
- Allows the runtime to evolve independently.
- Makes comparison across execution strategies possible.

**Liabilities.**
- Exact equivalence can be complicated by quantization or kernel differences.
- Runtime abstraction can hide performance costs if telemetry is weak.

**Failure modes.**
- Placement strategy silently changes representation precision.
- “Supported” hides unusable throughput.
- Runtime-specific behavior leaks into canonical adapter semantics.

### Adoption trigger

A second physical execution mode is required for a supported source-model class.

### Decision log

- **Decision:** none.

---

## APR-017 — Dense Layer Paging

**Category:** runtime, resource  
**Pattern status:** `CHARACTERIZED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** A dense source model may be much larger than accelerator memory.

**Definition.** Stream dense execution units through the accelerator in dependency order while retaining only the activations and working state required for the next unit.

**Benefits.**
- Removes total checkpoint size as an accelerator-residency requirement.
- Works with ordinary dense dependency structure.

**Liabilities.**
- Every layer still executes.
- I/O can dominate.
- Training/backpropagation is substantially harder than inference extraction.

**Failure modes.**
- Capacity feasibility is mistaken for compute feasibility.
- Layer transfer is serialized with compute when overlap was possible.
- Long autoregressive generation repeatedly streams the full model.

### Source provenance

**Kind:** external  
**Source ref:** `EXT-2026-002`  
**Source class:** runtime-study  
**Observed:** 2026-09-06  
**Evidence boundary:** Layerwise offload/paging architectures were studied; source identity is intentionally external to the repository.

### Project interpretation

Use primarily for source inference/extraction, not as evidence that full-model training is practical.

### Decision log

- **Decision:** none.
- **Rationale:** mechanism is well-characterized but not yet authorized for implementation.

---

## APR-018 — Routed Internal-Expert Paging for Sparse Source Models

**Category:** runtime, sparse-model  
**Pattern status:** `CHARACTERIZED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Sparse source models may contain a very large expert bank while only a subset is selected for current tokens.

**Definition.** Keep shared/trunk computation and routing metadata available, then fetch only selected internal-expert weights, grouping token work by expert and caching high-reuse experts when beneficial.

**Benefits.**
- Makes total sparse capacity much larger than resident expert capacity.
- Aligns parameter movement with actual routing.
- Enables RAM/storage to act as expert caches.

**Liabilities.**
- Token-level routing can select many unique experts across a batch.
- Cache behavior can dominate performance.
- Routing lookahead and prefetch add complexity.

**Failure modes.**
- Request-level sparsity is assumed from token-level Top-K.
- Every expert touched by a large batch is fetched separately with poor reuse.
- Cache hit rate is reported without end-to-end latency.

### Source provenance

**Kind:** external  
**Source ref:** `EXT-2026-003`  
**Source class:** sparse-runtime-study  
**Observed:** 2026-09-06

### Decision log

- **Decision:** none.

---

## APR-019 — Prefill-First Expertise Extraction

**Category:** extraction, runtime  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Oversized paged source models become prohibitively slow when the full weight path is traversed repeatedly for long autoregressive output.

**Definition.** Prefer one or a small number of source forward/prefill passes to extract internal representations directly. Use source text generation only when the experiment requires output-level supervision.

**Benefits.**
- Avoids repeated full-weight streaming.
- Aligns the runtime with latent expertise extraction.
- Reduces persistent decode-cache requirements.

**Liabilities.**
- Some expertise may be expressed only through generation behavior.
- Prefill-only extraction can miss iterative reasoning dynamics.

**Failure modes.**
- One-pass latent extraction is assumed equivalent to generated reasoning.
- Source decoding is reintroduced implicitly through another stage.
- The runtime optimizes prefill while the experiment actually needs decode.

### Adoption trigger

Paged source execution is used and output-level generation is not itself required.

### Required evidence

Compare one-pass latent extraction against generated/logit supervision.

### Decision log

- **Decision:** none.

---

## APR-020 — Immediate Expertise Compression at the Extraction Boundary

**Category:** representation, storage  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `research-only`

### Pattern definition

**Problem.** Persisting full source hidden-state sequences at scale can move the bottleneck from accelerator memory to storage.

**Definition.** Apply the Expertise Adapter or a dedicated compressor before persistence, storing a bounded canonical representation rather than raw full hidden states where evidence permits.

**Benefits.**
- Reduces artifact size.
- Makes large extraction corpora feasible.
- Decouples student experiments from source hidden width.

**Liabilities.**
- Compression can destroy transferable information.
- Compression training may require its own objective and calibration.

**Failure modes.**
- Bytes saved are treated as proof of transfer quality.
- Compression ratio is optimized without downstream student evaluation.
- Exact lexical information disappears silently.

### Adoption trigger

Raw hidden-state artifact cost becomes material.

### Required evidence

Storage reduction, reconstruction/alignment metrics, downstream transfer, and identifier/entity fidelity.

### Decision log

- **Decision:** none.

---

## APR-021 — Expertise Artifact Reuse across Student Experiments

**Category:** evidence, consolidation  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Re-running expensive source extraction for every student architecture wastes compute and prevents controlled student comparisons.

**Definition.** Treat compatible Expertise Artifacts as immutable reusable training inputs for multiple student experiments.

**Benefits.**
- Amortizes source extraction.
- Enables fair student comparisons.
- Separates source-model changes from student changes.

**Liabilities.**
- Artifact schema can ossify too early.
- New adapter versions require explicit re-extraction or migration.

**Failure modes.**
- Artifacts are silently regenerated between student comparisons.
- A student consumes an artifact outside its declared compatibility envelope.

### Decision log

- **Decision:** accepted principle.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-022 — Same-Sample Multi-Source Acquisition without Co-Residency

**Category:** extraction, routing  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `research-only`

### Pattern definition

**Problem.** Some samples may benefit from complementary expertise from more than one source model.

**Definition.** Allow a sample to accumulate multiple source artifacts sequentially, then synthesize them only after each source has been unloaded.

**Benefits.**
- Preserves the one-source-at-a-time memory invariant.
- Enables complementary supervision.
- Makes source contribution ablation straightforward.

**Liabilities.**
- Increases extraction time and storage.
- Requires duplicate/redundancy handling.

**Failure modes.**
- More sources are assumed to be better.
- Conflicting expertise is averaged without confidence or evidence.
- Source-order effects enter the canonical representation accidentally.

### Adoption trigger

Single-source transfer plateaus and complementary sources show measurable marginal utility.

### Decision log

- **Decision:** none.

---

## APR-023 — Versioned Canonical Expertise Space

**Category:** representation, evidence  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** A canonical latent contract can drift as adapters, dimensions, channel semantics, or normalization change.

**Definition.** Give each canonical expertise schema an immutable version identity and compatibility rules. Artifacts produced under incompatible schemas must fail closed.

**Benefits.**
- Prevents silent representation mixing.
- Makes migration explicit.
- Supports parallel experimentation with several canonical spaces.

**Liabilities.**
- More schema/version management.
- Cross-version migration may be expensive or impossible.

**Failure modes.**
- “Same dimension” is treated as compatible.
- New normalization silently reinterprets old vectors.
- Student checkpoints omit the canonical-space version they learned against.

### Adoption trigger

The first persisted latent expertise format is defined.

### Decision log

- **Decision:** none.

---

## APR-024 — Total and Active Parameters Are Independent Sparse-Model Metrics

**Category:** sparse-model, evidence  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `current-plan-authorized`

### Pattern definition

**Problem.** Sparse models can have large total parameter counts while executing only a small routed subset. Reporting only total parameters or only active parameters hides important resource properties.

**Definition.** Every sparse synthesized-model claim reports at least:

```text
total parameters
active parameters per token/path
number of internal experts
Top-K routing
shared parameters
expert storage bytes
peak resident bytes
```

**Benefits.**
- Separates capacity from compute.
- Makes hardware feasibility interpretable.
- Prevents misleading comparisons.

**Liabilities.**
- Active parameters alone still do not capture I/O, routing, or communication.

**Failure modes.**
- Total parameters are marketed as active compute.
- Active parameters are used to imply the inactive bank is free.
- Duplicated un-specialized experts inflate total count.

### Decision log

- **Decision:** accepted evidence rule.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-025 — Fine-Grained Internal Expert Banks

**Category:** sparse-model  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `research-only`

### Pattern definition

**Problem.** A sparse model built from a small number of very large internal experts can make routing coarse, paging expensive, and cache granularity poor.

**Definition.** Prefer evaluating many smaller routed internal experts with small Top-K activation as an alternative to a few giant experts.

**Benefits.**
- Finer specialization.
- Smaller paging units.
- Better cache granularity.
- Potentially higher total/active parameter ratio.

**Liabilities.**
- Larger routing problem.
- Token batches may touch many experts.
- Small experts can be underutilized.

**Failure modes.**
- Expert count increases without useful specialization.
- Router overhead dominates.
- Fine-grained experts fragment storage and I/O excessively.

### Source provenance

**Kind:** external  
**Source ref:** `EXT-2026-004`  
**Source class:** architecture-study  
**Observed:** 2026-09-06

### Adoption trigger

Sparse student architecture is authorized after dense transfer has been validated.

### Decision log

- **Decision:** none.

---

## APR-026 — Sequential Internal-Expert Specialization

**Category:** sparse-model, training  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `research-only`

### Pattern definition

**Problem.** Jointly training a large sparse expert bank can exceed optimizer-state and accelerator-memory budgets.

**Definition.** Train or adapt internal experts in bounded groups while shared parameters are frozen or otherwise controlled, then assemble the sparse model and separately train/validate routing and interactions.

**Benefits.**
- Bounds training working set.
- Allows large total sparse capacity to be constructed incrementally.
- Makes per-expert specialization measurable.

**Liabilities.**
- Joint behavior may differ after assembly.
- Sequential specialization can create incompatible experts.
- Router/expert co-adaptation is weakened.

**Failure modes.**
- Independently good experts interfere after composition.
- Frozen shared representation prevents necessary specialization.
- Assembled total parameter count is reported before joint validation.

### Adoption trigger

A sparse synthesized-model experiment exceeds the feasible jointly trainable working set.

### Decision log

- **Decision:** none.

---

## APR-027 — Shared Trunk plus Sparse Specialist Capacity

**Category:** sparse-model, architecture  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `research-only`

### Pattern definition

**Problem.** A fully independent model per capability duplicates general language and representation capacity.

**Definition.** Share dense trunk computation while allocating sparse routed internal-expert capacity for specialization.

**Benefits.**
- Reuses general computation.
- Allows total model capacity to grow faster than active compute.
- Creates a natural target for sequential expert specialization.

**Liabilities.**
- Shared trunk can become a bottleneck.
- Specialist capabilities may require changes outside expert blocks.
- Router quality becomes central.

**Failure modes.**
- All capability diversity is forced into one module type.
- Shared trunk is frozen despite evidence that it must adapt.
- Internal experts collapse to near-identical behavior.

### Adoption trigger

Dense synthesized-model transfer is validated and sparse scaling is authorized.

### Decision log

- **Decision:** none.

---

## APR-028 — Multi-Tier Weight Storage as a Cache Hierarchy

**Category:** runtime, resource  
**Pattern status:** `CHARACTERIZED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Treating host memory as a requirement that the complete source model must fit wastes the possibility of deeper storage-backed execution.

**Definition.** Treat accelerator memory, host memory, and persistent storage as a hierarchy of working-set caches rather than a single model-fit test.

**Benefits.**
- Enables source models larger than host memory.
- Allows hot shared weights and experts to stay close to compute.
- Makes cache policy explicit.

**Liabilities.**
- Persistent-storage bandwidth can dominate.
- Page/cache policy becomes workload-dependent.
- Host memory pressure can affect system stability.

**Failure modes.**
- Combined nominal capacity is treated like unified high-bandwidth memory.
- Cache hits are optimized while exposed transfer time remains dominant.
- Persistent storage wear/space is ignored.

### Source provenance

**Kind:** synthesis  
**Source ref:** `EXT-2026-002`

### Decision log

- **Decision:** none.

---

## APR-029 — Double-Buffered Weight Transfer

**Category:** scheduling, runtime  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Serial weight transfer followed by compute leaves both the transfer path and accelerator idle for part of every execution unit.

**Definition.** Prefetch the next execution unit into an alternate staging buffer while the current unit computes when dependencies and memory permit.

```text
compute current
    ∥
prefetch next
```

**Benefits.**
- Converts transfer+compute toward max(transfer, compute).
- Improves accelerator utilization.
- Applies to dense layer paging and selected sparse-expert paging.

**Liabilities.**
- Requires extra staging memory.
- Transfer and compute can contend for resources.
- Prefetching the wrong sparse expert wastes bandwidth.

**Failure modes.**
- Claimed overlap is not actually hidden on the critical path.
- Double buffering causes OOM.
- Sparse prefetch is based on unvalidated prediction.

### Adoption trigger

Profiling shows exposed transfer time and enough overlap opportunity.

### Required evidence

Timeline traces, transfer bytes, compute time, exposed vs hidden movement, and peak memory.

### Decision log

- **Decision:** none.

---

## APR-030 — Bottleneck Migration and Decomposed Physical-Cost Telemetry

**Category:** evidence, runtime  
**Pattern status:** `CHARACTERIZED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Solving memory fit can expose I/O; solving I/O can expose compute; solving compute can expose routing, conversion, or controller cost.

**Definition.** Measure the composed critical path and decompose at least:

```text
source weight movement
source compute
adapter compute
activation movement
artifact serialization
student compute
optimizer cost
routing/controller cost
storage I/O
```

Reclassify the dominant bottleneck after material interventions.

**Benefits.**
- Prevents optimizing an obsolete bottleneck.
- Makes capacity and throughput claims separate.
- Supports evidence-linked planning.

**Liabilities.**
- Instrumentation overhead and complexity.

**Failure modes.**
- Better cache hit rate is called a speedup despite higher latency.
- Lower bytes are called a throughput win despite expensive conversion.
- One microbenchmark is promoted as composed performance.

### Source provenance

**Kind:** synthesis  
**Source ref:** `METHOD-2026-002`

### Decision log

- **Decision:** none.

---

## APR-031 — Hardware Execution Surface rather than Device-Name Rules

**Category:** resource, planning  
**Pattern status:** `CANDIDATE`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `future-plan-candidate`

### Pattern definition

**Problem.** Rules tied to one hardware product name do not generalize when memory, bus, storage, precision support, or compute capabilities differ.

**Definition.** Describe runtime feasibility using measurable execution-surface properties:

```text
accelerator memory
host memory
storage throughput
host↔accelerator bandwidth
supported numeric formats
kernel availability
compute throughput
context/workspace budget
```

**Benefits.**
- Generalizable planning.
- Cleaner support matrix.
- Easier benchmarking across machines.

**Liabilities.**
- Requires calibration.
- Capability surfaces can be larger than a simple device lookup.

**Failure modes.**
- One device name implies one fixed policy.
- Nominal bandwidth is used instead of measured effective bandwidth.

### Adoption trigger

A second hardware configuration is supported or a planner chooses physical execution policy.

### Decision log

- **Decision:** none.

---

## APR-032 — Progressive Scale Validation

**Category:** planning, evidence  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `current-plan-authorized`

### Pattern definition

**Problem.** Large-model experiments consume enough time that architectural errors discovered only at target scale are disproportionately expensive.

**Definition.** Validate the same mechanism across increasing scales, promoting only when the previous scale demonstrates the intended transfer or runtime effect.

Example classes:

```text
mechanics scale
research scale
scaling-validation scale
target sparse scale
```

Exact parameter counts belong in experiment contracts, not this pattern.

**Benefits.**
- Shortens failure loops.
- Separates architectural validity from scaling behavior.
- Makes resource-regime changes visible.

**Liabilities.**
- Small-scale behavior may not extrapolate.
- Adds intermediate experiments.

**Failure modes.**
- Target-scale work begins before the adapter has beaten baseline.
- Small-scale success is treated as proof of large-scale success.

### Decision log

- **Decision:** accepted research methodology.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-033 — Capability Transfer Requires Student-Side Behavioral Evidence

**Category:** evidence  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `current-plan-authorized`

### Pattern definition

**Problem.** Representation similarity or source-model benchmark quality does not prove the synthesized model acquired the desired capability.

**Definition.** A capability-transfer claim requires behavioral evaluation on the synthesized model against appropriate baselines and controls.

**Benefits.**
- Prevents latent-space metrics from becoming capability claims.
- Keeps source quality separate from transfer quality.
- Supports negative results.

**Liabilities.**
- Requires capability-specific evaluation suites.

**Failure modes.**
- Low adapter loss is called successful expertise transfer.
- Source benchmark score is reported as student capability.
- One aggregate metric hides capability regression.

### Decision log

- **Decision:** accepted evidence rule.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-034 — Adapter Utility Gate

**Category:** evidence, extraction  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `current-plan-authorized`

### Pattern definition

**Problem.** A latent extraction mechanism can remain in the architecture despite providing no useful advantage over simpler training signals.

**Definition.** Retain an Expertise Adapter only when controlled experiments demonstrate sufficient benefit relative to simpler supervision under the relevant resource budget.

Candidate comparison families:

```text
output-only supervision
token-distribution supervision
latent-only supervision
typed latent + lexical supervision
combined output + latent supervision
```

**Benefits.**
- Keeps architecture evidence-driven.
- Provides a clean rejection path.
- Prevents complexity accumulation.

**Liabilities.**
- Requires careful objective matching and repeated runs.

**Failure modes.**
- Adapter survives because removing it feels like losing research progress.
- A tiny metric gain is accepted despite major storage/runtime cost.

### Decision log

- **Decision:** accepted methodology.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

## APR-035 — Negative Results as Search-Space Constraints

**Category:** evidence, planning  
**Pattern status:** `ACCEPTED`  
**Implementation status:** `NOT-LINKED`  
**Planning disposition:** `current-plan-authorized`

### Pattern definition

**Problem.** Failed transfer, routing, paging, compression, or sparse-training mechanisms can be repeatedly rediscovered if the failure envelope is not retained.

**Definition.** Preserve negative results with exact model class, data, adapter, hardware surface, representation, and workload boundaries, and use them to prune future experiments until a material forcing condition changes.

**Benefits.**
- Converts failure into project knowledge.
- Prevents repeated expensive dead ends.
- Encourages explicit regime changes before retry.

**Liabilities.**
- Negative results can be over-generalized.

**Failure modes.**
- One failed adapter is generalized to all latent transfer.
- One poor paging result is generalized to all storage-backed execution.
- A rejected mechanism is retried without changed conditions.

### Decision log

- **Decision:** accepted evidence practice.
- **Authority:** explicit project direction.
- **Decision date:** 2026-09-06.

---

# 10. Anti-Pattern Register

## APX-001 — Research idea inserted directly into active implementation

**Why dangerous:** collapses observation into execution authority and silently changes scope.

**Preferred alternative:** register, characterize, explicitly adopt or trial, then link to a plan.

---

## APX-002 — “Bridge” terminology for non-live source-to-source communication

**Why dangerous:** implies two live models communicate directly when the intended mechanism is source-specific extraction into a reusable canonical representation.

**Preferred alternative:** Expertise Adapter, Expertise Extraction, Canonical Expertise Representation.

---

## APX-003 — Raw hidden-state fusion across heterogeneous source models

**Why dangerous:** dimensional or numerical compatibility does not imply semantic alignment.

**Preferred alternative:** source-specific adapters plus a versioned canonical contract.

---

## APX-004 — Equal dimensions treated as equal meaning

**Why dangerous:** projection width solves shape mismatch, not latent-space alignment.

**Preferred alternative:** empirical alignment and downstream transfer evidence.

---

## APX-005 — Simultaneous source-model residency on scarce accelerator memory

**Why dangerous:** multiplies memory pressure without being required for sequential expertise extraction.

**Preferred alternative:** one source at a time; persist canonical artifacts.

---

## APX-006 — Complete source-model fit treated as an execution prerequisite

**Why dangerous:** prevents using paged or sparse working-set execution.

**Preferred alternative:** reason about the minimum resident execution working set.

---

## APX-007 — Streaming capacity treated as throughput

**Why dangerous:** moving weights from slower tiers can make an oversized model executable while still making it impractically slow.

**Preferred alternative:** measure capacity and composed throughput separately.

---

## APX-008 — Long autoregressive extraction from a paged oversized source by default

**Why dangerous:** repeats large weight traffic for every generated token.

**Preferred alternative:** prefill-first latent extraction when the experiment does not require generated output.

---

## APX-009 — Full hidden-state persistence at scale

**Why dangerous:** shifts the bottleneck from accelerator memory to storage.

**Preferred alternative:** measure whether compressed canonical expertise preserves transfer.

---

## APX-010 — Total sparse parameter count treated as useful capacity

**Why dangerous:** copied or inactive experts do not constitute independent capability.

**Preferred alternative:** report specialization, utilization, active parameters, and behavioral gain.

---

## APX-011 — Inactive sparse parameters treated as free

**Why dangerous:** inactive experts still consume storage, cache capacity, metadata, routing complexity, and movement bandwidth when selected.

**Preferred alternative:** account for storage and paging alongside active compute.

---

## APX-012 — Adapter value assumed rather than ablated

**Why dangerous:** adds training, storage, and inference complexity without evidence.

**Preferred alternative:** APR-005 and APR-034.

---

## APX-013 — Representation similarity promoted as capability transfer

**Why dangerous:** low latent loss can coexist with no behavioral improvement.

**Preferred alternative:** APR-033.

---

## APX-014 — External product/project vocabulary in repository architecture

**Why dangerous:** violates repository policy and encourages architecture-by-analogy.

**Preferred alternative:** vendor-neutral pattern language plus opaque source references.

---

## APX-015 — Duplicated sparse experts counted as independent expertise before specialization

**Why dangerous:** parameter count increases without information diversity.

**Preferred alternative:** measure expert divergence, routed utilization, and capability contribution after specialization.

---

## APX-016 — One runtime policy for dense and sparse source models

**Why dangerous:** dense models require sequential layer dependency execution while sparse models permit selected internal-expert paging.

**Preferred alternative:** common source-execution interface with architecture-specific physical strategies.

---

## APX-017 — Token-level sparse routing treated as request-level source selection

**Why dangerous:** a sequence can collectively touch many internal experts even when each token activates only a small Top-K.

**Preferred alternative:** measure unique-expert demand at the actual batch/sequence geometry.

---

## APX-018 — Artifact consumed without immutable source and adapter identity

**Why dangerous:** destroys reproducibility and can silently mix incompatible expertise spaces.

**Preferred alternative:** APR-004 and APR-023.

---

## APX-019 — Parameter target chosen before mechanism proof

**Why dangerous:** target scale can dominate research decisions before transfer and runtime hypotheses are validated.

**Preferred alternative:** APR-032.

---

## APX-020 — Baseline omitted because the project “must synthesize”

**Why dangerous:** forces a complex mechanism even when a simpler path is better.

**Preferred alternative:** baseline/no-adapter outcomes remain valid.

---

# 11. Planning Usage Procedure

When preparing or revising an implementation plan or experiment contract:

1. Define the authorized objective and frozen resource/quality constraints first.
2. Search the APR for patterns addressing the measured design pressure.
3. Read each relevant pattern's source observation, project interpretation, `must preserve`, `adoption trigger`, and `required evidence`.
4. Select only patterns necessary or materially beneficial to the objective.
5. Add a short `Architecture pattern references` section to the candidate plan.
6. Record deliberate non-adoption when it materially constrains scope.
7. If adopting a pattern, create the explicit adoption record with `acceptance_effect` and `claim_effect`.
8. Freeze the plan independently.
9. Once execution begins, the APR does not silently strengthen the acceptance boundary.

Recommended notation:

```text
Architecture pattern references
- APR-011 — trial-authorized for source representation alignment.
- APR-019 — adopted for oversized-source extraction; output generation remains a comparator.
- APR-025 — considered and deferred; dense transfer has not yet been validated.
- APR-029 — not adopted; transfer overlap has not been shown to be a bottleneck.
```

---

# 12. Research / Deep-Dive Usage Procedure

For every future research deep dive:

1. **Pin the source** where practical using an off-repository source ledger.
2. Assign or reuse an opaque repository-safe source reference.
3. **Describe the source first** in the off-repository research record.
4. **Extract reusable patterns** in vendor-neutral terms.
5. **Search existing APR IDs before adding a new one.**
6. If the invariant already exists, add provenance or refine the existing APR rather than creating a duplicate.
7. Record what **must not be copied** when source hardware, semantics, model architecture, or deployment assumptions differ.
8. Set status conservatively: normally `OBSERVED`, `CHARACTERIZED`, `CANDIDATE`, `DEFERRED`, or `REJECTED`.
9. Do not create implementation work through the deep dive.
10. If a later plan uses the idea, create an explicit adoption/trial record.

A deep dive should end with:

```text
APR impact:
  refinements:
  new candidate patterns:
  strengthened negative lessons:
  no plan effect unless separately adopted:
```

---

# 13. Maintenance Rules

1. **Stable APR/APX IDs.** Never renumber or reuse retired identifiers.
2. **Vendor-neutral names.** Repository-facing architecture uses project-owned or vendor-neutral language.
3. **Opaque external provenance.** External identities remain outside the repository; repository entries use stable opaque source refs.
4. **Preserve provenance.** Correct wrong refs explicitly; never silently erase source history.
5. **No silent status escalation.** `CHARACTERIZED → CANDIDATE → ACCEPTED` requires project rationale/decision.
6. **Separate pattern and implementation status.** Accepted does not mean scheduled; implemented does not mean verified.
7. **Trial is bounded.** `TRIAL-AUTHORIZED` permits an explicit experiment, not general architectural adoption.
8. **No hidden backlog authority.** `CANDIDATE` means reconsider under a forcing function.
9. **No phase reopening.** New research does not reopen a closed experiment without changed evidence or explicit authorization.
10. **Prefer one pattern with multiple sources.** Do not duplicate materially identical invariants.
11. **Record negative lessons.** Rejected and superseded patterns remain useful.
12. **Source facts and project decisions never share a field.**
13. **Do not rewrite history silently.** Append correction/supersession notes.
14. **Implementation claims require executable evidence.**
15. **Architecture appeal is not proof.**
16. **Capacity and throughput claims remain separate.**
17. **Total and active sparse parameters remain separate.**
18. **Student capability claims require student evidence.**
19. **External naming policy is permanent unless explicitly revised by project authority.**

---

# 14. Opaque Source Ledger

This repository retains source classes and stable opaque references only. Actual external source identities, URLs, and names are maintained outside the repository.

## METHOD-2026-001 — Architecture-register methodology comparison set

**Type:** methodology  
**Observed:** 2026-09-06  
**Relevant patterns:** APR-001  
**Evidence boundary:** Five-layer traceability, independent status models, promotion records, anti-pattern retention, maintenance discipline, planning/deep-dive procedures.

## METHOD-2026-002 — Runtime/evidence methodology synthesis set

**Type:** methodology  
**Observed:** 2026-09-06  
**Relevant patterns:** APR-030, APR-035  
**Evidence boundary:** Bottleneck migration, decomposed critical-path evidence, negative-result retention.

## EXT-2026-001 — Heterogeneous representation-transfer study

**Type:** external architecture/research evidence  
**Observed:** 2026-09-06  
**Relevant patterns:** APR-011, APR-012  
**Evidence boundary:** Heterogeneous latent alignment, discrete identity preservation, latent enrichment, representation-transfer ablations.

## EXT-2026-002 — Oversized dense execution study

**Type:** external runtime evidence  
**Observed:** 2026-09-06  
**Relevant patterns:** APR-017, APR-028  
**Evidence boundary:** Layerwise source execution with slower-tier weight residency and bounded accelerator working set.

## EXT-2026-003 — Sparse expert-paging runtime study

**Type:** external runtime evidence  
**Observed:** 2026-09-06  
**Relevant patterns:** APR-018, APR-028, APR-029  
**Evidence boundary:** Routed internal-expert paging, hot expert caching, storage-backed sparse execution.

## EXT-2026-004 — Fine-grained sparse architecture study

**Type:** external architecture evidence  
**Observed:** 2026-09-06  
**Relevant patterns:** APR-024, APR-025, APR-027  
**Evidence boundary:** High total/active parameter ratio, many small internal experts, selective routing, hybrid attention/resource behavior.

---

# 15. Intake Template for New Patterns

```markdown
## APR-### — Vendor-Neutral Pattern Name

**Category:** ...
**Pattern status:** `OBSERVED`
**Implementation status:** `NOT-LINKED`
**Planning disposition:** `research-only`

### Pattern definition

**Problem.** ...

**Definition.** ...

**Benefits.**
- ...

**Liabilities.**
- ...

**Failure modes.**
- ...

### Source provenance

**Kind:** external | project-native | synthesis | methodology  
**Source ref:** ...  
**Source class:** ...  
**Version/revision:** ...  
**Observed:** YYYY-MM-DD  
**Content digest:** ...  
**Evidence boundary:** ...

### Source observation

What the source actually implements, claims, or demonstrates.

### Project interpretation

**Adaptation.** ...

**Must preserve.**
- ...

**Conflicts.**
- ...

**Adoption trigger.** ...

**Required evidence.**
- ...

**Relevant timescale.** ...

**Related patterns.** APR-...

### Decision log

- **Pattern status:** `OBSERVED`
- **Implementation status:** `NOT-LINKED`
- **Planning disposition:** `research-only`
- **Decision:** none
- **Authority:** null
- **Decision date:** null
- **Rationale:** ...
- **Links:** ...
```

---

# 16. Current Normalized Summary

| ID | Pattern | Status | Implementation | Disposition |
|---|---|---|---|---|
| APR-001 | Architecture Pattern Register as a Non-Binding Planning Sidecar | ACCEPTED | IMPLEMENTED | current-plan-authorized |
| APR-002 | Expertise Extraction and Consolidation as Separate Phases | ACCEPTED | NOT-LINKED | future-plan-candidate |
| APR-003 | Bounded Working-Set Source Execution | ACCEPTED | NOT-LINKED | future-plan-candidate |
| APR-004 | Canonical Expertise Artifact with Immutable Lineage | ACCEPTED | NOT-LINKED | future-plan-candidate |
| APR-005 | Simpler Supervision as a First-Class Comparator | ACCEPTED | NOT-LINKED | current-plan-authorized |
| APR-010 | Source Model and Internal Expert Are Distinct Concepts | ACCEPTED | NOT-LINKED | current-plan-authorized |
| APR-011 | Source-Specific Expertise Adapter into a Canonical Space | ACCEPTED | NOT-LINKED | future-plan-candidate |
| APR-012 | Typed Expertise Representation | CANDIDATE | NOT-LINKED | research-only |
| APR-013 | Student-Side Alignment Adapter | CANDIDATE | NOT-LINKED | future-plan-candidate |
| APR-014 | Sequential Multi-Source Expertise Composition | ACCEPTED | NOT-LINKED | future-plan-candidate |
| APR-015 | Capability-Guided Source Selection | CANDIDATE | NOT-LINKED | future-plan-candidate |
| APR-016 | Runtime Strategy behind a Common Source-Execution Interface | CANDIDATE | NOT-LINKED | future-plan-candidate |
| APR-017 | Dense Layer Paging | CHARACTERIZED | NOT-LINKED | future-plan-candidate |
| APR-018 | Routed Internal-Expert Paging for Sparse Source Models | CHARACTERIZED | NOT-LINKED | future-plan-candidate |
| APR-019 | Prefill-First Expertise Extraction | CANDIDATE | NOT-LINKED | future-plan-candidate |
| APR-020 | Immediate Expertise Compression at the Extraction Boundary | CANDIDATE | NOT-LINKED | research-only |
| APR-021 | Expertise Artifact Reuse across Student Experiments | ACCEPTED | NOT-LINKED | future-plan-candidate |
| APR-022 | Same-Sample Multi-Source Acquisition without Co-Residency | CANDIDATE | NOT-LINKED | research-only |
| APR-023 | Versioned Canonical Expertise Space | CANDIDATE | NOT-LINKED | future-plan-candidate |
| APR-024 | Total and Active Parameters Are Independent Sparse-Model Metrics | ACCEPTED | NOT-LINKED | current-plan-authorized |
| APR-025 | Fine-Grained Internal Expert Banks | CANDIDATE | NOT-LINKED | research-only |
| APR-026 | Sequential Internal-Expert Specialization | CANDIDATE | NOT-LINKED | research-only |
| APR-027 | Shared Trunk plus Sparse Specialist Capacity | CANDIDATE | NOT-LINKED | research-only |
| APR-028 | Multi-Tier Weight Storage as a Cache Hierarchy | CHARACTERIZED | NOT-LINKED | future-plan-candidate |
| APR-029 | Double-Buffered Weight Transfer | CANDIDATE | NOT-LINKED | future-plan-candidate |
| APR-030 | Bottleneck Migration and Decomposed Physical-Cost Telemetry | CHARACTERIZED | NOT-LINKED | future-plan-candidate |
| APR-031 | Hardware Execution Surface rather than Device-Name Rules | CANDIDATE | NOT-LINKED | future-plan-candidate |
| APR-032 | Progressive Scale Validation | ACCEPTED | NOT-LINKED | current-plan-authorized |
| APR-033 | Capability Transfer Requires Student-Side Behavioral Evidence | ACCEPTED | NOT-LINKED | current-plan-authorized |
| APR-034 | Adapter Utility Gate | ACCEPTED | NOT-LINKED | current-plan-authorized |
| APR-035 | Negative Results as Search-Space Constraints | ACCEPTED | NOT-LINKED | current-plan-authorized |

The summary is informational. It is **not** an implementation priority queue.

---

# 17. Immediate Methodological Effect

Effective with this register:

```text
research deep dive
      ↓
off-repository source record
      ↓
opaque repository-safe source reference
      ↓
source observation
      ↓
APR refinement / new candidate
      ↓
no implementation-plan effect by default
```

When planning:

```text
authorized objective
      ↓
consult relevant APRs
      ↓
explicit adopt / trial / defer / reject
      ↓
versioned implementation plan or experiment contract
      ↓
real execution + evidence
```

For the current project thesis:

```text
select source
   ↓
execute bounded working set
   ↓
extract
   ↓
adapt into canonical expertise
   ↓
persist immutable artifact
   ↓
unload source
   ↓
consolidate into student
   ↓
evaluate against simpler baselines
   ↓
retain / refine / reject
```

> **Study broadly. Name internally. Register precisely. Adopt explicitly. Execute within the measured resource envelope. Verify transfer on the synthesized model.**
