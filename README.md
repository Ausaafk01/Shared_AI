# AI Brain — Shared Persistent Memory Architecture

AI Brain is a shared persistent memory system for multiple AI agents (Claude, Codex, Gemini, Antigravity).

- **Event System**: The append-only, immutable event log recording all interactions, actions, code changes, and tool calls.
- **Memory Model**: Structured, immutable knowledge derived from events representing established facts, decisions, preferences, and context.
- **Memory Extraction**: Deterministic rule-based extraction component converting single events or correlated multi-event sequences into validated Memory objects.
- **Memory Updating**: Version-aware updating component managing evolution, supersession, deduplication, and conflict preservation across immutable memories.
- **Conflict Management**: Multi-agent conflict detection, evidence-based evaluation, and lifecycle resolution component preventing contradictory memories from masquerading as established facts.
- **Knowledge Graph**: In-memory directed graph linking memories, events, agents, projects, and concepts with deterministic BFS/DFS traversal and conflict integration.
- **Vector Memory**: Semantic retrieval subsystem using cosine similarity, deterministic offline & local embedding providers, persistent SQLite vector storage, and version-aware memory indexing.
- **Retrieval Engine**: Multi-strategy retrieval orchestrator combining Vector Memory and Knowledge Graph with Reciprocal Rank Fusion (RRF), transparent relevance scoring, and strict project isolation.
- **Context Builder**: Converts retrieved memories into a compact, prioritized, token-budget-aware context package formatted for AI agent prompts, with semantic compression, provenance preservation, and dispute warnings.
- **Brain Loop**: End-to-end orchestration layer coordinating the repeatable interaction lifecycle: agent input, retrieval, context preparation, response capture, event generation, memory extraction/updating, and knowledge synchronization.
- **Multi-Agent System**: Provider-independent agent abstraction, lifecycle registry, session continuity, agent switching, collaboration workflows, and secure provider adapters (Claude, Codex, Gemini, Antigravity) with zero credentials stored in Brain records.

---

## 1. Event System

- **Storage**: Persistent, append-only local SQLite database (`ai_brain/events/storage/sqlite.py`).
- **Engine-Level Immutability**: SQLite triggers (`prevent_event_update`, `prevent_event_delete`) prohibit `UPDATE` and `DELETE` queries.
- **Ordered & Indexed**: Chronological ordering with compound indexes for fast filtering by project, agent, session, and type.
- **Core Event Categories**: `USER_MESSAGE`, `AGENT_MESSAGE`, `TOOL_CALL`, `TOOL_RESULT`, `CODE_CHANGE`, `FILE_CHANGE`, `ERROR`, `TASK_CREATED`, `TASK_COMPLETED`, `DECISION`, `AGENT_SWITCH`, `SESSION_START`, `SESSION_END`. Custom types can be dynamically registered via `register_event_type()`.

---

## 2. Memory Model

While Events record *what happened*, Memories represent *what the brain has learned or established*.

### Memory Types (`MemoryType`)
- `FACT`: Established facts (e.g., *"The project uses PostgreSQL."*).
- `DECISION`: Architectural/design decisions (e.g., *"Redis was removed because it caused stale cache problems."*).
- `TASK`: Actionable goals or unresolved tasks (e.g., *"The authentication bug remains unresolved."*).
- `ERROR`: Pitfalls and recurring error patterns to avoid.
- `ATTEMPT`: Action attempts and observed outcomes.
- `CODE_KNOWLEDGE`: Codebase patterns, conventions, and architectural details.
- `PREFERENCE`: User, team, or agent preferences (e.g., *"The user prefers concise code."*).
- `CONTEXT`: Broad environmental or operational context.

### Lifecycle Statuses (`MemoryStatus`)
- `ACTIVE`: Currently valid, actionable knowledge.
- `SUPERSEDED`: Replaced by newer or updated knowledge while retaining historical provenance.
- `DISPUTED`: Contested or conflicting with other observations.
- `ARCHIVED`: Historical context, preserved for auditing without being active.

### Versioning Fields
- `root_memory_id`: Invariant stable identity representing the conceptual knowledge entity across all revisions.
- `memory_id`: Unique snapshot ID for that specific revision.
- `parent_memory_id`: Pointer to the immediate predecessor version.
- `version`: Monotonically increasing version integer ($1, 2, 3, \dots$).

---

## 3. Memory Extraction

The `MemoryExtractor` component analyzes events and generates immutable `Memory` objects based on deterministic extraction rules.

### Extraction Capabilities
- **Single-Event Extraction**: Extracts memory objects from individual events containing explicit knowledge.
- **Multi-Event Extraction**: Correlates related events in a session (e.g., proposal $\rightarrow$ approval $\rightarrow$ execution) to synthesize consolidated memories.
- **Noise & Speculation Filtering**: Ignores casual conversations ("hello", "thanks") and unverified assumptions ("maybe", "I think").
- **Deduplication**: Merges repeated identical information within the same extraction batch.
- **Conflict Handling**: Detects contradictory assertions and marks them `DISPUTED`.
- **Project Separation**: Events from different projects are strictly segregated.

---

## 4. Memory Updating

The `MemoryUpdater` coordinates updates between existing memories and incoming information while respecting full object immutability:

- **Rule A (New Information)**: Distinct knowledge is recorded as an independent memory.
- **Rule B (Updated Information)**: Replaced memories are superseded by creating an active Version $N+1$ and marking Version $N$ as `SUPERSEDED`, linking both in an immutable chain.
- **Rule C & D (Duplicate / Unchanged)**: Identical or unchanged information is detected and suppressed, avoiding redundant versions.
- **Rule E (Disputed Information)**: Contradictory statements that cannot be resolved with certainty are marked as `DISPUTED` on both sides without dropping provenance.

---

---

## 5. Conflict Management

The Conflict Management component detects, tracks, assesses, and resolves disagreements between memories originating from different events and AI agents:

### Conflict Types (`ConflictType`)
- `FACTUAL`: Direct contradiction between factual assertions (e.g. PostgreSQL vs MySQL).
- `TEMPORAL`: Outdated knowledge vs superseding migration statements.
- `DECISION`: Conflicting architectural directions or decisions.
- `STATUS`: Disagreements on whether a task is complete vs still active.
- `TECHNICAL`: Contradictory configuration settings (e.g., port 8000 vs 8080).
- `SOURCE`: Disagreements regarding evidence credibility, provenance, reliability, or attribution (disjoint agent sets are preserved in provenance, not classified as source conflicts).

### Conflict Lifecycle (`ConflictStatus`)
- `DETECTED`: Initial state when disagreement is recognized.
- `UNDER_REVIEW`: Actively undergoing evidence collection and evaluation.
- `RESOLVED`: Sufficient evidence found; winning memory retained and losing memory superseded via `MemoryUpdater`.
- `UNRESOLVED`: Evidence is inconclusive; claims are preserved in the `DISPUTED` state.
- `DISMISSED`: Conflict assessed and determined to be non-contradictory or invalid.

### Conflict Architecture
- **`MemoryConflict`**: Immutable record of detected disagreement, tracking involved memory IDs, evidence, detection timestamp, confidence, and resolution.
- **`ConflictDetector`**: Analyzes memories within the same project, identifying contradictions while isolating separate projects and avoiding false positives.
- **`ConflictEvaluator`**: Multi-factor evidentiary assessment checking corroboration count, verified tool/event provenance, agent authority, and temporal migration context.
- **`ConflictManager`**: Coordinates the resolution lifecycle, delegating to `MemoryUpdater` for supersession when resolved, or marking claims `DISPUTED` when inconclusive.

---

---

## 6. Knowledge Graph

The Knowledge Graph component transforms memories, events, agents, projects, and concepts into an in-memory directed graph with strict project isolation and deterministic traversal.

### Node Types (`NodeType`)
- `MEMORY`: Structured brain knowledge nodes.
- `EVENT`: Recorded foundational events.
- `AGENT`: Agents observing or contributing knowledge.
- `PROJECT`: Scoped project root nodes.
- `CONCEPT`: Conceptual or architectural themes.
- `ENTITY`: Systems, libraries, or components.

### Relationship Types (`RelationshipType`)
- `BELONGS_TO`: Node association with a project.
- `DERIVED_FROM`: Knowledge derivation from events or ancestor versions.
- `CREATED_BY`: Attribution to the producing agent.
- `SUPERSEDES` / `SUPERSEDED_BY`: Version chain progression.
- `SUPPORTS`: Corroborating knowledge relations.
- `CONTRADICTS`: Documented conflicts between opposing claims.
- `RELATED_TO`: General association.
- `DEPENDS_ON`: Dependency relationships.
- `PART_OF`: Hierarchical sub-component relations.

### Traversal Engine (`GraphTraversal`)
- **Algorithms**: Breadth-First Search (`BFS`) and Depth-First Search (`DFS`).
- **Directionality**: `OUTGOING`, `INCOMING`, or `BOTH`.
- **Filtering**: Type-based filtering for nodes and relationships.
- **Constraints**: Maximum depth cutoffs and project boundaries.
- **Cycle Detection**: Detects circular paths and prevents infinite loops.

### Memory & Conflict Integration (`GraphIntegrator`)
- **Idempotent Integration**: Batch or single-memory integration without node duplication.
- **Provenance Preservation**: Links memories to `EVENT` nodes via `DERIVED_FROM` and `AGENT` nodes via `CREATED_BY`.
- **Version Evolution**: Links predecessor and successor versions via `SUPERSEDES`, `SUPERSEDED_BY`, and `DERIVED_FROM`.
- **Conflict Mapping**: Automatically synthesizes `CONTRADICTS` relationships with resolution metadata.

---

## 7. Vector Memory

The Vector Memory subsystem enables semantic similarity search across stored memories, matching queries based on conceptual meaning rather than exact keywords.

### Architecture & Components
- **Modular Embedding Providers (`ai_brain/vectors/providers/`)**:
  - `EmbeddingProvider`: Abstract base class defining `embed_text()`, `embed_batch()`, `dimension`, and `model_id`.
  - `MockEmbeddingProvider`: Deterministic, fully offline, dependency-free embedding provider (default 384 dimensions matching MiniLM-L6-v2) utilizing semantic cluster hashing and L2 normalization for reliable testing.
  - `SentenceTransformerEmbeddingProvider`: Optional local provider using `sentence-transformers` models (e.g., `all-MiniLM-L6-v2`) when installed.
- **Persistent Vector Storage (`ai_brain/vectors/storage/`)**:
  - `VectorRecord`: Immutable, frozen vector representation with metadata and JSON/dict serialization.
  - `SQLiteVectorStore`: SQLite-backed vector storage with WAL mode, compound uniqueness constraint `(project_id, memory_id, version, model_id)`, indexed lookups, and cross-session persistence.
- **Semantic Vector Index (`ai_brain/vectors/index.py`)**:
  - `VectorIndex`: In-memory cosine similarity computation using pure Python with deterministic tie-breaking `(-score, memory_id, -version)`.
  - **Threshold & Top-K**: Returns ranked matches above `similarity_threshold` up to `top_k`.
  - **Strict Project Isolation**: Enforces tenant boundaries, rejecting cross-project searches.
  - **Default Status Filtering**: Defaults to `allowed_statuses=['ACTIVE']`, excluding `SUPERSEDED` and `ARCHIVED` memories from standard retrieval while preserving them in storage.
- **Memory Integration Manager (`ai_brain/vectors/manager.py`)**:
  - `VectorMemoryManager`: Orchestrates embedding generation, vector record creation, storage, and index synchronization.
  - **Version Supersession**: When Version $N+1$ is indexed, automatically marks Version $N$ as `SUPERSEDED` in vector storage and updates the active index.
  - **Disputed Knowledge**: Preserves vectors for disputed memories (`status='DISPUTED'`) with appropriate metadata tagging.
  - **Archival & Purging**: Supports vector status updates (`ARCHIVED`) and deletion without mutating source `Memory` objects.

---

## 8. Retrieval Engine

The Retrieval Engine coordinates query processing, project-scoped retrieval, vector search, graph traversal, Reciprocal Rank Fusion (RRF), and transparent multi-factor relevance scoring.

### Architecture & Components
- **Query Processing (`ai_brain/retrieval/query.py`)**:
  - `QueryProcessor`: Validates query integrity, normalizes casing and whitespace, removes English stopwords to extract keywords, extracts domain entities (`proj-*`, `mem-*`, `agent:*`, technical terms), and validates project scope independently of any LLM.
- **Project Isolation (`ai_brain/retrieval/exceptions.py`)**:
  - Strict boundary enforcement requiring explicit `project_id`. Prohibits cross-project leaks across vector, graph, and hybrid retrieval.
- **Graph Retrieval (`ai_brain/retrieval/graph_retriever.py`)**:
  - `GraphRetriever`: Explores the `KnowledgeGraph` using BFS traversal up to configurable `max_depth`.
  - **Honest Evidence Attribution**: Vector-selected seeds do not automatically receive graph evidence or depth-0 attribution. Graph evidence strictly requires traversing an actual domain relationship (depth $\ge 1$) or explicit graph targeting.
  - **Seed Strategies**: Uses explicit seed memory IDs if provided, otherwise matches query keywords/entities against project memory nodes, or leverages top vector search hits for outward expansion.
  - **Distance Decay**: Decays relevance score with traversal depth: $\text{score} = \frac{1.0}{1.0 + 0.5 \times \text{depth}}$.
  - **Status Filtering**: Strictly excludes `SUPERSEDED` and `ARCHIVED` memories by default before scoring across vector, graph, and hybrid modes. Explicit inclusion of historical statuses emits visible warnings.
- **Vector Retrieval (`ai_brain/retrieval/vector_retriever.py`)**:
  - `VectorRetriever`: Interfaces with `VectorIndex` to generate query embeddings, retrieve top-$k$ matches above `similarity_threshold`, and apply memory type, status, and agent filters.
- **Hybrid Fusion (`ai_brain/retrieval/fusion.py`)**:
  - `HybridFusion`: Implements Reciprocal Rank Fusion (RRF) to merge vector and graph candidate streams without double-counting:
    $$\text{RRF\_score}(d) = w_v \cdot \frac{1}{k + r_v(d)} + w_g \cdot \frac{1}{k + r_g(d)}$$
  - Preserves individual vector and graph scores, traversal metadata, and attribution sources (`("vector", "graph")`).
  - Gracefully handles partial failures via `allow_partial_results`.
- **Memory Relevance Scoring (`ai_brain/retrieval/scoring.py`)**:
  - `RelevanceScorer`: Deterministic, transparent scoring formula returning a detailed `ScoreBreakdown`:
    $$\text{final\_score} = \min\left(1.0, \max\left(0.0, (S_{\text{base}} \times F_{\text{imp}} \times F_{\text{conf}} \times F_{\text{rec}} \times M_{\text{stat}}) + B_{\text{prov}}\right)\right)$$
    - $S_{\text{base}}$: Base hybrid fused score or normalized vector/graph score.
    - $F_{\text{imp}} = 0.8 + 0.2 \times \text{importance}$
    - $F_{\text{conf}} = 0.8 + 0.2 \times \text{confidence}$
    - $F_{\text{rec}} = 0.9 + 0.1 \times e^{-\frac{\text{age\_days}}{30.0}}$
    - $M_{\text{stat}}$: `ACTIVE` ($1.0$), `DISPUTED` ($0.7$), `ARCHIVED` ($0.4$), `SUPERSEDED` ($0.2$).
    - $B_{\text{prov}}$: Corroboration bonus for multiple supporting agents and events (up to $+0.10$).
- **Public API (`ai_brain/retrieval/engine.py`)**:
  - `RetrievalEngine`: Unified retrieval orchestrator supporting `retrieve()`, `retrieve_vector()`, `retrieve_graph()`, and `retrieve_hybrid()`.

---

## 9. Context Builder

The Context Builder converts retrieved candidate memories into a compact, prioritized, token-budget-aware context package ready for direct ingestion by AI agent prompts.

### Key Capabilities
- **Relevant Memory Selection (`ai_brain/context/selector.py`)**:
  - Enforces strict project isolation.
  - Automatically deduplicates multiple revisions of the same conceptual entity across `root_memory_id`, selecting the highest-version active memory.
  - Excludes superseded and archived memories by default while allowing explicitly configured historical inclusion.
  - Preserves disputed memories as explicitly disputed rather than dropping or masking them.
- **Context Prioritization (`ai_brain/context/prioritizer.py`)**:
  - Deterministic multi-factor scoring:
    $$\text{PriorityScore} = S_{\text{relevance}} \times (0.7 + 0.3 \times \text{Importance}) \times (0.8 + 0.2 \times \text{Confidence}) \times \text{TypeBonus} \times \text{RecencyFactor} \times M_{\text{status}}$$
  - Direct relevance serves as the primary gate: if relevance is 0.0, priority is strictly 0.0 (high importance cannot force irrelevant memories into context).
  - Category bonuses for active decisions (`1.15`), active tasks (`1.10`), and core facts (`1.05`).
  - Disputed memories are discounted (`0.65`) so they do not outrank established facts.
  - Transparent, inspectable `PriorityBreakdown` explaining each factor.
- **Context Compression & Consolidation (`ai_brain/context/compressor.py`)**:
  - Semantic whitespace normalization and boilerplate removal.
  - Merges compatible, identical claims from multiple agents while consolidating source agent IDs, source event IDs, and relationships.
  - Never merges contradictory or disputed claims.
  - Zero hallucination and zero mutation of original memory records.
- **Token Budget Management (`ai_brain/context/budget.py` & `tokenizer.py`)**:
  - Strict enforcement of `max_tokens` budgets.
  - Includes offline `DeterministicTokenizer` (subword-aware, zero external dependencies) and `BaseTokenizer` abstraction for pluggable tokenizers.
  - Greedily packs candidates in priority order, accounting for markdown formatting overhead, section titles, and attribution tables.
  - Detailed accounting reporting `tokens_used`, `remaining_tokens`, and `excluded_items` with explicit reasons (`token_budget_exceeded`, `duplicate_version`, `excluded_status`, `low_relevance`).
- **Agent-Ready Generation (`ai_brain/context/formatter.py` & `models.py`)**:
  - Generates prompt-ready markdown (`to_markdown()`) and plain text (`to_text()`).
  - Distinct sections for Active Decisions, Active Tasks, Established Facts, Connected Knowledge, and Disputed Knowledge.
  - Disputed claims feature prominent warning banners instructing the agent not to assume them as established facts.
  - Full serialization round-trip (`to_dict()`, `to_json()`, `from_dict()`, `from_json()`).

---

## 10. Brain Loop

The Brain Loop subsystem orchestrates the complete, repeatable interaction lifecycle between AI agents and the shared memory architecture:

### Two-Stage Interaction Lifecycle
1. **Context Preparation (`prepare_context`)**:
   - **Input Processing (`ai_brain/loop/input_processor.py`)**: Validates `project_id`, `agent_id`, `session_id`, and `content`, rejecting empty, malformed, or unauthorized cross-project requests.
   - **Input Event Logging (`ai_brain/loop/event_handler.py`)**: Stores an immutable `INTERACTION_INPUT` event in the persistent `EventStore` with deterministic UUID v5 hashing for retry idempotency.
   - **Brain Retrieval (`ai_brain/retrieval/engine.py`)**: Executes vector, graph, or hybrid retrieval scoped strictly to the project.
   - **Context Assembly (`ai_brain/context/builder.py`)**: Fits retrieved knowledge into a token-budgeted, agent-ready context package preserving dispute warnings and omission notices.
   - **Delivery Event Logging**: Persists a linked `CONTEXT_DELIVERED` event and returns an `InteractionContextResult` to the caller without making any external LLM calls.

2. **Interaction Completion (`complete_interaction`)**:
   - **Response Capture (`ai_brain/loop/response_capture.py`)**: Validates the agent response against the originating interaction ID, project ID, agent ID, and session ID.
   - **Response Event Logging**: Persists a linked `INTERACTION_RESPONSE` event.
   - **Memory Extraction & Updating (`ai_brain/loop/memory_handler.py`)**: Runs rule-based extraction across interaction events, resolves versioning via `MemoryUpdater`, and evaluates contradictions using `ConflictDetector` and `ConflictManager`.
   - **Knowledge Synchronization**: Automatically indexes new/updated memories into the `KnowledgeGraph` via `GraphIntegrator` and vector storage via `VectorMemoryManager`.
   - **Completion Event Logging**: Records `INTERACTION_COMPLETED` and returns an `InteractionCompletionResult`.

### Error Handling, Idempotency & Partial Failure Policy
- **Deterministic Retry Idempotency**: If `prepare_context` or `complete_interaction` is retried with identical identifiers, the loop detects the existing interaction record and event IDs, returning cached results without creating duplicate events or memories.
- **Explicit Failure Attribution**:
  - Fatal failures (e.g. invalid input, cross-project injection, persistence failure) raise domain exceptions immediately (`InputValidationError`, `LoopProjectIsolationError`, `EventPersistenceError`).
  - Downstream synchronization failures (e.g. transient vector index or graph memory errors) are reported as `PARTIAL_SUCCESS` with granular warnings, avoiding silent data corruption while keeping the interaction record accurate.
- **Immutability Invariant**: Historical events in `EventStore` and original `Memory` objects are never modified in-place.
- **Provider Agnostic**: Works offline with zero dependencies on specific AI providers (Claude, Codex, Gemini, Antigravity). External AI model execution is performed by the caller between `prepare_context` and `complete_interaction`.

---

## 11. Multi-Agent System

Phase 2 introduces a decoupled, multi-agent orchestration architecture allowing multiple heterogeneous AI agents to discover, share, evaluate, and collaborate on persistent knowledge within AI Brain.

### Key Architectural Guarantees:
- **Provider Independence**: Agents interact through a unified `Agent` abstraction. The Brain never assumes or requires a specific model provider.
- **Credential & Secret Isolation**: API keys and secrets are strictly resolved dynamically from environment variables or secure credential injectors. Secret keywords (`api_key`, `secret`, `token`, `password`, `private_key`) are strictly prohibited in persistent agent configurations and logs.
- **Strict Project Isolation**: Every agent interaction, session, memory, and graph relationship is partitioned by `project_id`. Agents in Project A cannot read or pollute memories in Project B.
- **Auditability & Provenance**: Agent handovers emit `AGENT_SWITCH` audit events. All memories preserve source agent IDs, source event IDs, and version lineages.
- **Dispute Integrity**: Contradictory claims across agents are surfaced as `DISPUTED` knowledge with explicit warnings, never silently overwritten or presented as consensus truth.

### Core Multi-Agent Components:
1. **Agent Interface (`ai_brain/agents/base.py`)**: Abstract base class providing a complete interaction lifecycle: `prepare_context` $\rightarrow$ `generate_response` $\rightarrow$ `complete_interaction`. Includes timeout protection, error translation, and agent-specific interaction history retrieval.
2. **Agent Registry (`ai_brain/agents/registry/`)**: Persistent SQLite storage managing agent identities, status (`ACTIVE`, `INACTIVE`, `DECOMMISSIONED`), capabilities, and secret-prohibited configurations with duplicate registration prevention.
3. **Simulated Agents (`ai_brain/agents/simulated/`)**:
   - `ResearchAgent`: Explores architectural topics and records verified facts (`MemoryType.FACT`).
   - `CodingAgent`: Consumes research context and formulates technical decisions (`MemoryType.DECISION`).
   - `ReviewAgent`: Evaluates proposed decisions against established facts, identifying contradictions.
4. **Session Continuity (`ai_brain/agents/sessions/`)**: `SessionManager` and `SQLiteSessionStore` manage session lifecycles, surviving process restarts and tracking active and participating agents.
5. **Agent Switching (`ai_brain/agents/switching.py`)**: `AgentSwitcher` coordinates seamless handovers between agents within an active project, assembling targeted context and logging `AGENT_SWITCH` audit events.
6. **Multi-Agent Collaboration Workflows (`ai_brain/agents/collaboration/`)**: `CollaborationEngine` runs ordered multi-agent workflows (e.g. Researcher $\rightarrow$ Coder $\rightarrow$ Reviewer), tracking stage progress, outputs, and supporting resumption of interrupted workflows without re-executing completed steps.
7. **External Provider Adapters (`ai_brain/agents/adapters/`)**: Adapters translating Brain context packages into provider-native payloads for:
   - **Claude** (`ClaudeAdapter`): Actual provider API integration for Anthropic Messages API (`https://api.anthropic.com/v1/messages`).
   - **Gemini** (`GeminiAdapter`): Actual provider API integration for Google Gemini REST API (`generateContent`).
   - **Codex Agent** (`CodexAgentAdapter` / `CodexAdapter`): Actual agent integration for OpenAI's Codex coding agent, supporting both non-interactive CLI execution (`codex exec --full-auto`) and the stateful OpenAI Responses API (`/v1/responses`).
   - **OpenAI Model** (`OpenAIModelAdapter`): Generic model API adapter for OpenAI Chat Completions (`/v1/chat/completions`).
   - **Antigravity Agent** (`AntigravityAgentAdapter` / `AntigravityAdapter`): Actual agent integration for Google Antigravity Agent, supporting both Google Gemini Interactions API (`/v1beta/interactions`) and Antigravity CLI (`agy`).
   - All adapters feature deterministic mock transports for 100% offline testing, live integration hooks, secret isolation, and clean error mapping (`ProviderRateLimitError`, `AgentTimeoutError`, `ProviderAuthenticationError`, `ProviderConnectionError`).

---

## 12. Machine Learning Layer

The Machine Learning layer (`ai_brain/ml/`) provides modular, offline-capable, explainable models that enhance memory importance prediction, retrieval re-ranking, duplicate detection, conflict classification, and staleness estimation.

### Key Architectural Guarantees:
- **Zero Mandatory External APIs**: 100% offline, local execution using standard mathematical and calibrated statistical methods without requiring vendor API credentials or heavy runtime dependencies.
- **Graceful Deterministic Fallback**: Every ML component automatically falls back to deterministic rule-based logic if untrained, if confidence falls below configurable thresholds, or if errors occur.
- **Explainable Predictions & Calibrated Confidence**: Every prediction outputs numeric confidence and granular feature contributions explaining why the prediction was made.
- **Full Provenance & Leakage Protection**: Datasets strictly distinguish real, manually labeled, feedback-derived, and synthetic examples. Dataset splitters group by source memory ID to ensure zero train/validation data leakage.
- **Controlled Feedback Lifecycle**: Feedback records are durably captured in SQLite with provenance and model versions; batch preparation is decoupled from model retraining to prevent instability.
- **No Unsafe Autonomous Actions**: Potentially stale memories are flagged for review rather than deleted; candidate duplicates are never merged solely on embedding similarity; contradictory claims are preserved as disputed.

### ML Models:
1. **Memory Importance Model (`MemoryImportanceModel`)**:
   - Predicts long-term memory retention value in `[0.0, 1.0]`.
   - Evaluates memory type (decision, preference, constraint vs temporary), decision indicators, constraint markers, temporal decay, and provenance breadth.
   - Non-circular: does not treat frequently retrieved information as automatically important.
2. **Relevance Ranking Model (`RelevanceRankingModel`)**:
   - Optional secondary re-ranker across candidates retrieved by Vector, Graph, or Hybrid search.
   - Blends base retrieval score, query term overlap, agent affinity, recency decay, and status penalties.
   - Preserves deterministic retrieval filters and score breakdowns.
3. **Duplicate Detection Model (`DuplicateDetectionModel`)**:
   - Classifies memory pairs into `EXACT_DUPLICATE`, `SEMANTIC_DUPLICATE`, `RELATED_DISTINCT`, or `CONTRADICTORY`.
   - Prevents merging contradictory claims under high embedding similarity.
4. **Conflict Detection Model (`ConflictDetectionModel`)**:
   - Distinguishes genuine contradictions (`FACTUAL`, `TECHNICAL`, `TEMPORAL`, `DECISION`, `SOURCE`) from non-conflicting elaborations and chronological updates.
5. **Staleness Detection Model (`StalenessDetectionModel`)**:
   - Assesses memory obsolescence risk `[0.0, 1.0]` without deleting memories.
   - Preserves history and makes staleness probability inspectable for retrieval and context building.

### Deferred Reinforcement Learning / Contextual Bandits:
- Reinforcement learning and online contextual bandits (e.g., epsilon-greedy, Thompson sampling) are explicitly deferred. At this phase, interaction volume is discrete and sparse, and online exploration would risk serving degraded context packages to agents, violating deterministic safety guarantees. Calibrated supervised models with explicit feedback ingestion are the appropriate, verifiable architecture.

---

## Project Structure

```
├── ai_brain/
│   ├── __init__.py               # Top-level exports (v0.2.0)
│   ├── agents/                   # Multi-Agent System package (Phase 2)
│   │   ├── __init__.py
│   │   ├── base.py               # Agent ABC (execute, prepare_context, complete_interaction)
│   │   ├── exceptions.py         # Agent, Session, Adapter & Collaboration exceptions
│   │   ├── models.py             # AgentRecord, AgentStatus, AgentCapability, Execution models
│   │   ├── switching.py          # AgentSwitcher (handover & AGENT_SWITCH audit events)
│   │   ├── registry/             # Agent Registry subsystem
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # AgentRegistryStore ABC
│   │   │   ├── registry.py       # AgentRegistry (duplicate protection, secret filtering)
│   │   │   └── sqlite.py         # SQLiteAgentRegistryStore (survives restarts)
│   │   ├── simulated/            # 100% offline simulated agents
│   │   │   ├── __init__.py
│   │   │   ├── research.py       # ResearchAgent (gathers facts)
│   │   │   ├── coding.py         # CodingAgent (proposes implementation decisions)
│   │   │   └── review.py         # ReviewAgent (evaluates context & contradictions)
│   │   ├── sessions/             # Session Continuity subsystem
│   │   │   ├── __init__.py
│   │   │   ├── models.py         # SessionRecord, SessionStatus
│   │   │   ├── manager.py        # SessionManager (session events, continuity)
│   │   │   └── storage/          # Persistent session storage
│   │   │       ├── __init__.py
│   │   │       ├── base.py       # SessionStore ABC
│   │   │       └── sqlite.py     # SQLiteSessionStore (survives restarts)
│   │   ├── collaboration/        # Multi-Agent Collaboration Workflow subsystem
│   │   │   ├── __init__.py
│   │   │   ├── models.py         # CollaborationStep, CollaborationWorkflowRecord
│   │   │   └── workflow.py       # CollaborationEngine (ordered workflows & resumption)
│   │   └── adapters/             # External Provider Adapters
│   │       ├── __init__.py
│   │       ├── base.py           # ProviderAdapter ABC (secret isolation, mock transport)
│   │       ├── claude.py         # ClaudeAdapter (Anthropic Messages API)
│   │       ├── codex.py          # CodexAdapter (OpenAI Chat Completions API)
│   │       ├── gemini.py         # GeminiAdapter (Google Gemini generateContent)
│   │       └── antigravity.py    # AntigravityAdapter (Antigravity Agent protocol)
│   ├── events/                   # Event System package
│   │   ├── exceptions.py
│   │   ├── models.py
│   │   ├── types.py
│   │   └── storage/
│   ├── memories/                 # Memory System package
│   │   ├── exceptions.py         # Memory, extraction, updating, & conflict exceptions
│   │   ├── models.py             # Immutable Memory model with version fields
│   │   ├── types.py              # MemoryType, MemoryStatus & registry
│   │   ├── extraction/           # Extraction component
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py      # MemoryExtractor orchestrator
│   │   │   ├── rules.py          # Modular extraction rules
│   │   │   └── helpers.py        # Candidate memory & parsing helpers
│   │   ├── updating/             # Updating component
│   │   │   ├── __init__.py
│   │   │   ├── comparison.py     # MemoryComparator & relation classifier
│   │   │   ├── versioning.py     # VersionTracker & MemoryVersionHistory
│   │   │   └── updater.py        # MemoryUpdater orchestrator
│   │   └── conflicts/            # Conflict Management component
│   │       ├── __init__.py
│   │       ├── types.py          # ConflictType, ConflictStatus, transition validation
│   │       ├── models.py         # Immutable MemoryConflict model
│   │       ├── detector.py       # ConflictDetector
│   │       ├── evaluator.py      # ConflictEvaluator & multi-factor scoring
│   │       └── manager.py        # ConflictManager & lifecycle orchestration
│   ├── graph/                    # Knowledge Graph package
│   │   ├── __init__.py
│   │   ├── types.py              # NodeType, RelationshipType, Traversal types
│   │   ├── exceptions.py         # GraphError hierarchy & ProjectIsolationError
│   │   ├── models.py             # Immutable GraphNode and GraphRelationship
│   │   ├── graph.py              # In-memory KnowledgeGraph structure
│   │   ├── traversal.py          # GraphTraversal engine (BFS, DFS, cycle detection)
│   │   └── integration.py        # GraphIntegrator (Memory & Conflict integration)
│   ├── vectors/                  # Vector Memory subsystem
│   │   ├── __init__.py
│   │   ├── exceptions.py         # VectorError hierarchy & dimension validation
│   │   ├── models.py             # Immutable VectorRecord, VectorMatch, VectorSearchResult
│   │   ├── index.py              # VectorIndex (cosine similarity, tie-breaking, status filters)
│   │   ├── manager.py            # VectorMemoryManager (orchestrator & lifecycle)
│   │   ├── providers/            # Modular embedding providers
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # EmbeddingProvider ABC
│   │   │   ├── mock.py           # Deterministic MockEmbeddingProvider
│   │   │   └── local.py          # SentenceTransformerEmbeddingProvider
│   │   └── storage/              # Persistent vector storage
│   │       ├── __init__.py
│   │       ├── base.py           # VectorStore ABC
│   │       └── sqlite.py         # SQLiteVectorStore
│   └── retrieval/                # Retrieval Engine subsystem
│       ├── __init__.py
│       ├── types.py              # RetrievalMethod, SeedStrategy & normalizers
│       ├── exceptions.py         # RetrievalError hierarchy & ProjectIsolationError
│       ├── models.py             # Immutable QueryFilters, ScoreBreakdown, RetrievalItem, RetrievalResponse
│       ├── query.py              # QueryProcessor (normalization, stopwords, entities)
│       ├── graph_retriever.py    # GraphRetriever (BFS traversal, seed strategies, status filters)
│       ├── vector_retriever.py   # VectorRetriever (VectorIndex integration, filters)
│       ├── fusion.py             # HybridFusion (Reciprocal Rank Fusion RRF)
│       ├── scoring.py            # RelevanceScorer (multi-factor explainable scoring)
│       └── engine.py             # RetrievalEngine (public entrypoints & partial failure handling)
│   └── context/                  # Context Builder subsystem
│       ├── __init__.py
│       ├── exceptions.py         # ContextError hierarchy & isolation errors
│       ├── models.py             # ContextPackage, ContextItem, ExcludedItem, PriorityBreakdown
│       ├── tokenizer.py          # BaseTokenizer & DeterministicTokenizer
│       ├── selector.py           # MemorySelector & SelectionCriteria
│       ├── prioritizer.py        # MemoryPrioritizer & multi-factor priority scoring
│       ├── compressor.py         # ContextCompressor & compatible claim merging
│       ├── budget.py             # TokenBudgetManager & greedy knapsack packing
│       ├── formatter.py          # ContextFormatter (agent markdown & dispute warnings)
│       └── builder.py            # ContextBuilder orchestrator
│   └── loop/                     # Brain Loop subsystem
│       ├── __init__.py
│       ├── exceptions.py         # LoopError hierarchy & validation errors
│       ├── models.py             # AgentInteractionInput, AgentResponseInput, InteractionRecord, LoopStage
│       ├── input_processor.py    # InputProcessor (validation & project isolation)
│       ├── response_capture.py   # ResponseCapture (agent response validation & linkage)
│       ├── event_handler.py      # LoopEventHandler (idempotent event persistence)
│       ├── memory_handler.py     # LoopMemoryHandler (extraction, updates, conflicts & sync)
│       ├── orchestrator.py       # BrainLoop (prepare_context, complete_interaction, resumption)
│       └── storage/              # Durable interaction persistence
│           ├── __init__.py
│           ├── base.py           # InteractionStore ABC
│           └── sqlite.py         # SQLiteInteractionStore (survives restarts)
├── tests/
│   ├── test_models.py            # Event model tests
│   ├── test_storage.py           # Persistent SQLite storage tests
│   ├── test_extensibility.py     # Event extensibility tests
│   ├── test_memory_models.py     # Memory model validation & immutability tests
│   ├── test_extraction.py        # Memory extraction & rule tests
│   ├── test_updating.py          # Memory updating & version history tests
│   ├── test_conflicts.py         # Conflict detection, evaluation & resolution tests
│   ├── test_graph.py             # Knowledge Graph structure, traversal & integration tests
│   ├── test_vectors.py           # Vector Memory provider, storage, index & lifecycle tests
│   ├── test_retrieval.py         # Retrieval Engine query, graph, vector, hybrid & scoring tests
│   ├── test_context.py           # Context Builder selection, priority, budget & format tests
│   ├── test_loop.py              # Brain Loop lifecycle, idempotency & failure handling tests
│   ├── test_agents.py            # Agent models, registry, secrets filtering & simulated agents tests
│   ├── test_sessions.py          # Session lifecycle, persistence, switching & dispute tests
│   ├── test_collaboration.py     # Multi-agent collaboration workflows & failure resumption tests
│   └── test_adapters.py          # Claude, Codex, Gemini & Antigravity provider adapter tests
├── examples/
│   ├── quickstart.py             # Event system & agent logging demo
│   ├── vector_demo.py            # Vector memory semantic search & supersession demo
│   ├── retrieval_demo.py         # Retrieval engine vector, graph, hybrid & scoring demo
│   ├── context_builder_demo.py   # Context builder prioritization, budget & prompt demo
│   ├── brain_loop_demo.py        # End-to-end single-agent interaction lifecycle demo
│   └── multi_agent_demo.py       # Full multi-agent knowledge sharing, switching & collaboration demo
└── README.md
```

---

## Usage Examples

### Integrating Memories into the Knowledge Graph & Traversing

```python
from ai_brain.graph import (
    KnowledgeGraph,
    GraphIntegrator,
    GraphTraversal,
    TraversalAlgorithm,
    TraversalDirection,
    NodeType,
)
from ai_brain.memories import Memory, MemoryType

graph = KnowledgeGraph()
integrator = GraphIntegrator(graph)
traversal = GraphTraversal(graph)

# 1. Create and integrate a memory with provenance
mem = Memory.create(
    memory_id="mem-101",
    memory_type=MemoryType.FACT,
    content="The project uses PostgreSQL as its database.",
    project_id="proj-db",
    source_event_ids=["evt-001"],
    source_agent_ids=["claude"],
)
node = integrator.integrate_memory(mem)

# 2. Traverse the graph from the memory node
result = traversal.traverse(
    start_node_id=node.node_id,
    algorithm=TraversalAlgorithm.BFS,
    direction=TraversalDirection.OUTGOING,
    max_depth=2,
)

print(f"Visited {len(result.visited_nodes)} nodes:")
for n in result.visited_nodes:
    print(f"- [{n.node_type}] {n.label} (ID: {n.node_id})")
```

### Detecting and Resolving Memory Conflicts

```python
from ai_brain.memories import (
    Memory,
    MemoryType,
    ConflictDetector,
    ConflictEvaluator,
    ConflictManager,
)

detector = ConflictDetector()
evaluator = ConflictEvaluator()
manager = ConflictManager()

# 1. Conflicting memories from different agents
mem_mysql = Memory.create(
    memory_type=MemoryType.FACT,
    content="The project uses MySQL for persistence.",
    project_id="proj-web",
    source_agent_ids=["agent:gemini"],
    confidence=0.7,
)

mem_pg = Memory.create(
    memory_type=MemoryType.FACT,
    content="The project uses PostgreSQL for persistence.",
    project_id="proj-web",
    source_agent_ids=["agent:claude", "agent:codex"], # Stronger corroboration
    confidence=0.9,
)

# 2. Detect conflict
conflicts = detector.detect_conflicts([mem_mysql, mem_pg])
assert len(conflicts) == 1
conflict = conflicts[0]
print(f"Detected conflict: {conflict.description}")

# 3. Evaluate evidence
assessment = evaluator.evaluate(conflict, [mem_mysql, mem_pg])
print(f"Evaluator decision: {assessment.winning_memory_id} (confidence: {assessment.confidence})")

# 4. Resolve conflict (supersedes losing memory, preserves winning memory)
result = manager.resolve(conflict, assessment, [mem_mysql, mem_pg])
print(f"Action: {result.action_taken}")
print(f"Conflict Status: {result.conflict.status}")
```

### Semantic Memory Search & Version Supersession

```python
from ai_brain.memories import Memory, MemoryType
from ai_brain.vectors import (
    MockEmbeddingProvider,
    SQLiteVectorStore,
    VectorIndex,
    VectorMemoryManager,
)

# 1. Initialize components
provider = MockEmbeddingProvider()
store = SQLiteVectorStore(":memory:")
index = VectorIndex()
manager = VectorMemoryManager(provider=provider, store=store, index=index)

# 2. Index memory
mem1 = Memory.create(
    memory_id="mem-db-1",
    root_memory_id="root-db",
    version=1,
    memory_type=MemoryType.FACT,
    content="The application uses PostgreSQL with Prisma ORM.",
    project_id="proj-alpha",
)
manager.index_memory(mem1)

# 3. Perform semantic search (matches even with conceptual synonyms)
search_res = manager.search(
    project_id="proj-alpha",
    query="What database and relational storage do we use?",
    top_k=5,
)
for match in search_res.matches:
    print(f"Score {match.score:.4f} | Memory: {match.memory_id} | Content: {match.content}")
```

### Retrieval Engine: Hybrid Retrieval & Relevance Scoring

```python
from ai_brain.retrieval import RetrievalEngine, RetrievalMethod, QueryFilters

# 1. Initialize engine with existing vector index and knowledge graph
engine = RetrievalEngine(
    vector_index=manager.index,
    graph=graph,
    allow_partial_results=True,
)

# 2. Execute hybrid retrieval (vector semantic search + graph traversal)
response = engine.retrieve(
    "How is user authentication implemented?",
    project_id="proj-alpha",
    method=RetrievalMethod.HYBRID,
    top_k=5,
    filters=QueryFilters(memory_types=("FACT", "DECISION")),
)

# 3. Inspect results and transparent relevance score breakdown
print(f"Retrieved {len(response.results)} memories in {response.execution_time_ms:.2f}ms:")
for item in response.results:
    sb = item.score_breakdown
    print(f"[{item.score:.4f}] ({item.memory_id}) {item.content}")
    print(f"  Sources: {item.sources} | Base={sb.base_score:.2f} | Imp={sb.importance_factor:.2f} | "
          f"Conf={sb.confidence_factor:.2f} | Status={sb.status_multiplier:.2f} | Prov={sb.provenance_bonus:.2f}")
```

### Context Builder: Budget Packing & Agent Prompt Generation

```python
from ai_brain.context import ContextBuilder

# 1. Initialize Context Builder with a 600-token budget
builder = ContextBuilder(default_max_tokens=600)

# 2. Convert retrieval response into a prioritized, budgeted context package
package = builder.build_context(
    response,
    project_id="proj-alpha",
    max_tokens=600,
)

# 3. Inspect budget metrics and sections
print(f"Tokens Used: {package.tokens_used}/{package.token_budget} | Remaining: {package.remaining_tokens}")
print(f"Included Items: {len(package.items)} | Excluded: {len(package.excluded_items)}")
for sec in package.sections:
    print(f"Section '{sec.name}': {len(sec.items)} items (~{sec.section_tokens} tokens)")

# 4. Generate agent-ready markdown prompt
print("\n" + package.to_markdown())
```

### Brain Loop: End-to-End Interaction Lifecycle

```python
from ai_brain.loop import BrainLoop, AgentInteractionInput, AgentResponseInput

# 1. Initialize orchestrator (connects event store, graph, vector index, and context builder)
loop = BrainLoop(
    event_store=event_store,
    graph=graph,
    vector_manager=vector_manager,
    default_token_budget=1000,
)

# 2. Stage 1: Prepare context for incoming agent input
ctx_result = loop.prepare_context(AgentInteractionInput(
    project_id="proj-alpha",
    agent_id="claude",
    session_id="session-1",
    content="What authentication mechanism are we using?",
))
prompt_context = ctx_result.context_package.to_markdown()

# 3. Agent processes prompt_context and generates a response (caller-managed)
agent_reply = "We decided to migrate authentication from JWT to OAuth2 with PKCE."

# 4. Stage 2: Complete interaction, extract memories, update versions, and sync knowledge
completion = loop.complete_interaction(AgentResponseInput(
    interaction_id=ctx_result.interaction_id,
    project_id="proj-alpha",
    agent_id="claude",
    session_id="session-1",
    content=agent_reply,
))

print(f"Status: {completion.status.value}")
print(f"Events Logged: {len(completion.events_persisted)}")
print(f"Memories Extracted/Updated: {len(completion.memories_updated)}")
```

### Multi-Agent System: Registry, Handover, and Collaboration Workflows

```python
from ai_brain.agents import (
    AgentRegistry,
    AgentExecutionRequest,
    AgentSwitcher,
    CollaborationEngine,
    ResearchAgent,
    CodingAgent,
    ReviewAgent,
    SessionManager,
    ClaudeAdapter,
)

# 1. Initialize registry and simulated agents
registry = AgentRegistry()
researcher = ResearchAgent(agent_id="researcher", name="Research Specialist")
coder = CodingAgent(agent_id="coder", name="Lead Architect")
reviewer = ReviewAgent(agent_id="reviewer", name="Code Reviewer")

registry.register(researcher)
registry.register(coder)
registry.register(reviewer)

# 2. Start a project session and execute researcher
session_mgr = SessionManager(event_store=brain_loop.event_store)
session_mgr.create_session(project_id="proj-alpha", session_id="sess-01", initial_agent_id="researcher")

req = AgentExecutionRequest(
    project_id="proj-alpha",
    session_id="sess-01",
    content="Investigate caching requirements: The project uses Redis Sentinel for cache high availability.",
)
research_res = researcher.execute(req, brain_loop)

# 3. Seamless agent switching with audit event logging
switcher = AgentSwitcher(brain_loop=brain_loop, session_manager=session_mgr)
handover_ctx = switcher.switch(
    from_agent_id="researcher",
    to_agent=coder,
    project_id="proj-alpha",
    session_id="sess-01",
    handover_query="Redis caching architecture",
    reason="Handoff to developer for implementation planning",
)

# 4. Multi-agent collaboration workflow
collab = CollaborationEngine()
workflow = collab.create_research_coding_review_workflow(
    project_id="proj-alpha",
    session_id="sess-01",
    task_description="Implement Redis caching pipeline",
    research_agent_id="researcher",
    coding_agent_id="coder",
    review_agent_id="reviewer",
)
workflow_record = collab.execute_workflow(
    workflow,
    agents={"researcher": researcher, "coder": coder, "reviewer": reviewer},
    brain_loop=brain_loop,
)
print(f"Workflow status: {workflow_record.status.value}")

# 5. External LLM Provider Adapter (offline mock transport or live API)
claude = ClaudeAdapter(
    agent_id="claude-prod",
    model="claude-3-5-sonnet-20241022",
    mock_transport=lambda payload: "DECISION: Deploy Redis cluster with 3 sentinel nodes.",
)
claude_res = claude.execute(req, brain_loop)
print(f"Claude Output: {claude_res.content}")
```

---

## Running Automated Tests

Run all 343 collected automated tests (339 passing, 4 live provider tests skipped offline) across the Event System, Memory Model, Memory Extraction, Memory Updating, Conflict Management, Knowledge Graph, Vector Memory, Retrieval Engine, Context Builder, Brain Loop, Multi-Agent System, Machine Learning Layer, Benchmarking Framework, and Storage Lifecycle:

```bash
python -W error::ResourceWarning -m unittest discover -s tests -v
```

### Exact Test Inventory by Test File (343 Discovered Tests)
| Test Module | Discovered Tests | Status | Description |
| :--- | :---: | :---: | :--- |
| `tests/test_adapters.py` | 15 | 11 Passed, 4 Skipped | Anthropic, Codex, Gemini, Antigravity provider adapters (mock & live) |
| `tests/test_agents.py` | 14 | 14 Passed | Agent abstraction, registry persistence, secret filtering, simulated agents |
| `tests/test_benchmarks.py` | 21 | 21 Passed | Conditions A/B/C isolation, McNemar, Student's t stats, exception safety |
| `tests/test_collaboration.py` | 3 | 3 Passed | Multi-stage collaboration workflows & step execution |
| `tests/test_conflicts.py` | 28 | 28 Passed | Conflict detection, multi-factor evidence scoring & resolution lifecycle |
| `tests/test_context.py` | 34 | 34 Passed | Knapsack budget packing, markdown formatting, dispute omission notices |
| `tests/test_extensibility.py` | 4 | 4 Passed | Custom extensible event types & validation |
| `tests/test_extraction.py` | 21 | 21 Passed | Deterministic rule-based memory extraction & filtering |
| `tests/test_graph.py` | 26 | 26 Passed | KnowledgeGraph structure, BFS/DFS traversal & tenant isolation |
| `tests/test_loop.py` | 48 | 48 Passed | BrainLoop orchestration, retry idempotency, stage recovery & lifecycle |
| `tests/test_memory_models.py` | 13 | 13 Passed | Memory models, immutability & versioning pointer serialization |
| `tests/test_ml.py` | 28 | 28 Passed | Lexical/temporal/graph features, grouped CV & supervised prototypes |
| `tests/test_models.py` | 10 | 10 Passed | Event model immutability & deterministic UUID v5 generation |
| `tests/test_retrieval.py` | 27 | 27 Passed | Vector, Graph, Hybrid RRF & ML-enhanced retrieval & scoring breakdown |
| `tests/test_sessions.py` | 6 | 6 Passed | Multi-agent session continuity & SQLite storage persistence |
| `tests/test_storage.py` | 16 | 16 Passed | Append-only SQLite events, corruption recovery, context managers & GC |
| `tests/test_updating.py` | 11 | 11 Passed | Rules A-E, monotonic version chains & duplicate handling |
| `tests/test_vectors.py` | 18 | 18 Passed | VectorRecord, cosine similarity, index management & status filtering |
| **Total Discovered** | **343** | **339 Passed, 4 Skipped** | **0 Failures, 0 Errors, 0 Resource Warnings** |

### Running Demonstrations

All demonstrations are 100% self-contained and run offline without external dependencies or credentials:

```bash
# Event System quickstart
python examples/quickstart.py

# Vector Memory semantic search demo
python examples/vector_demo.py

# Retrieval Engine vector, graph, hybrid & scoring demo
python examples/retrieval_demo.py

# Context Builder prioritization, budget packing & prompt demo
python examples/context_builder_demo.py

# Brain Loop end-to-end single-agent lifecycle demo
python examples/brain_loop_demo.py

# Multi-Agent System knowledge sharing, switching & collaboration demo
python examples/multi_agent_demo.py

# Machine Learning models, feedback collection & evaluation demo
python examples/ml_demo.py

# Phase 4 Benchmarking & Evaluation live demonstration
python examples/benchmark_demo.py

# Full Phase 4 benchmark suite CLI execution
python scripts/run_benchmarks.py
```

---

## Master Architecture Checklist

### Phase 1: Core Brain Architecture (Complete)
- [x] **Event System**: Append-only SQLite event store with engine-level immutability triggers.
- [x] **Memory Model**: Structured, immutable knowledge representations with versioning pointers.
- [x] **Memory Extraction**: Deterministic rule-based extraction from single and multi-event streams.
- [x] **Memory Updating**: Monotonic version chains, supersession rules, and unchanged deduplication.
- [x] **Conflict Management**: Disagreement detection, multi-factor evidence scoring, and lifecycle resolution.
- [x] **Knowledge Graph**: In-memory directed graph with cycle-safe BFS/DFS traversal and conflict integration.
- [x] **Vector Memory**: Cosine similarity semantic search, mock/local providers, and persistent SQLite index.
- [x] **Retrieval Engine**: Hybrid RRF fusion, transparent score explainability, and status filtering.
- [x] **Context Builder**: Token-budget knapsack packing, markdown prompt generation, and dispute omission warnings.
- [x] **Brain Loop**: End-to-end orchestration, durable interaction storage, and retry idempotency.

### Phase 2: Multi-Agent System (Complete)
- [x] **Agent Abstraction & Interface**: Provider-agnostic `Agent` base class, `AgentRecord`, lifecycle execution, timeouts, and agent-specific history.
- [x] **Agent Registry & Lifecycle**: Persistent SQLite registry with duplicate detection, status transitions, and secret prohibition filtering.
- [x] **Simulated Agents**: 100% offline `ResearchAgent` (facts), `CodingAgent` (decisions), and `ReviewAgent` (contradictions).
- [x] **Shared Brain Access**: Shared project memory access via `BrainLoop`, strict tenant isolation, and conflict preservation.
- [x] **Agent Switching**: `AgentSwitcher` for clean transitions, context assembly, and immutable `AGENT_SWITCH` audit event logging.
- [x] **Session Continuity**: `SessionManager` and `SQLiteSessionStore` tracking active participants and surviving process restarts.
- [x] **Collaboration Workflows**: `CollaborationEngine` for ordered task workflows, Brain context passing, partial failure tracking, and resumption.
- [x] **External Provider Adapters**: Decoupled adapters for `ClaudeAdapter`, `CodexAdapter`, `GeminiAdapter`, and `AntigravityAdapter` with mock transports and credential isolation.
- [x] **Automated Test Suite**: 288 total unit and integration tests (250 Phase 1 tests 100% preserved + 38 Phase 2 tests [34 passing mock tests, 4 live integration tests with explicit credential skipping]).
- [x] **Comprehensive Demos & Docs**: Working `examples/multi_agent_demo.py` and detailed `README.md` documentation.

### Phase 3: Machine Learning Layer (Qualified Evaluation)
- [x] **Memory Importance Model** *(Prototype trained and evaluated)*: Continuous importance score regressor predicting long-term retention value from content, type, constraints, decisions, and provenance breadth (Holdout MAE 0.0699, 5-Fold CV MAE 0.0701 ± 0.0024 vs 0.3750 baseline).
- [x] **Relevance Ranking Model** *(Prototype trained and evaluated)*: Pointwise continuous relevance-score regressor blending base, vector, graph, and metadata scores (Holdout MAE 0.0538 vs 0.1417 baseline; ranking order NDCG@5=1.0 reflects bimodal test data, not learned ranking permutations).
- [x] **Duplicate Detection Model** *(Prototype trained and evaluated)*: Contradiction-gated pairwise classifier distinguishing exact duplicates, semantic rewrites, and distinct/contradictory assertions (100% accuracy on curated templates; unvalidated on noisy production diffs).
- [x] **Conflict Detection Model** *(Trained but insufficiently evaluated)*: One-vs-Rest multiclass classifier categorizing conflict types (`FACTUAL`, `TECHNICAL`, `TEMPORAL`, `DECISION`, `SOURCE`, `NONE`). Explicitly qualified: dataset contains 48 curated pairs (8 per class), which is too small for statistical generalization.
- [x] **Staleness Detection Model** *(Prototype trained and evaluated)*: Obsolescence probability estimator preserving historical records and status flags without memory deletion (Holdout accuracy 100%, 5-Fold CV accuracy 100% on curated age/supersession fixtures).
- [x] **Feature Extraction Pipeline**: Modular extractors for lexical/textual, temporal lifecycle, and graph/vector structural features.
- [x] **Group-Aware Cross-Validation & Zero Leakage**: Source-memory-grouped partitioning (`DatasetSplitter.split` and `DatasetSplitter.k_fold_split`) ensuring strict zero leakage across training and validation folds.
- [x] **Feedback Collection System**: SQLite-backed `FeedbackCollector` recording automated context utilization and manual feedback signals with audited batch export (no automatic retraining).
- [x] **Model Evaluation & Comparative Baselines**: Evaluates score calibration (MAE), classification metrics (Precision, Recall, F1, Brier score), and cross-validation confidence intervals against deterministic baselines.
- [x] **Brain Loop & Retrieval Integration**: Integrated execution paths in `RetrievalEngine` score blending, `LoopMemoryHandler` importance scoring, `MemoryComparator` duplicate detection, `ConflictDetector` conflict typing, and `BrainLoop` feedback logging.
- [x] **Preserved Phase 1 & Phase 2 Baseline**: All 288 existing tests preserved + 28 ML tests = 316 tests (312 passing, 4 live skipped). All 7 demos passing.

### Phase 4: Evaluation & Benchmarking (Final Terminology Cleanup Complete)
- [x] **Three Experimental Conditions**: Strictly isolated Condition A (Cold Start: prompt only, zero history), Condition B (Raw History: chronological event stream up to token budget with tail retention / FIFO discard and omitted history reporting), and Condition C (AI Brain: persistent memory, graph, vectors, and selective context compilation).
- [x] **Expanded 30-Scenario Benchmark Suite**: 30 distinct scenarios (3 per category across all 10 categories) covering Task Continuation, Architectural Decisions, Technical Constraints, Updated Requirements, Contradictory Claims, Agent Switching, Session Interruption, Repeated Explanation Avoidance, Needle in a Haystack, and Stale Knowledge Rejection with documented provenance.
- [x] **Fine-Grained Token Accounting & Formatting Overhead Audit**: Estimated subword token accounting using `DeterministicTokenizer` (~4 chars/token approximation), disclosing historical archive, supplied history, omitted history, raw memory, formatted context, agent input, output, and total tokens. Investigated 45.6 -> 337.9 token expansion (UUID provenance tables account for ~180 tokens; disabling reduces context by ~58%).
- [x] **Complete Retrieval Strategy Evaluation at K=5**: Evaluates all 4 genuine strategies—Vector (Cosine), Knowledge Graph (Traversal), Hybrid (RRF Fusion), and ML-Enhanced (Reranked)—on the identical query set and ground truth labels, reporting Macro/Micro Precision@5, Recall@5, F1, MRR, NDCG@5, and irrelevant retrieval rates without claiming unwarranted ML improvements.
- [x] **Audited Task Continuation & Partial-Credit Scoring**: Strict binary success requires meeting all 5 conditions (facts $\ge 75\%$, decisions $\ge 75\%$, 0 constraint violations, conflict handled, 0 stale memories used). Continuous partial-credit incorporates conflict handling (pure contradiction failure yields 0.0 partial score; 05B receives 0.0 rather than 1.0) and stale memory penalties. Strict pass rates: Cold Start 0.0% (0/30), Raw History 83.3% (25/30), AI Brain 70.0% (21/30).
- [x] **Corrected Conflict Confusion Matrix & Documented Precision Convention**: Explicit ground truth labels across all 30 scenarios (`GENUINE_CONTRADICTION`, `COMPATIBLE_CLAIMS`, `TEMPORAL_UPDATE`, `NO_CONFLICT`) with Scenario 05C classified as a negative conflict case (`COMPATIBLE_CLAIMS`). Precision defaults to 0.0 by documented convention when zero positive predictions are made ($TP + FP = 0$). Corrected confusion matrix: 2 TP, 0 FP, 28 TN, 0 FN (100% precision, 100% recall) with 0 stale memories leaked.
- [x] **Reconciled Avoided Explanations Formula**: Strict scenario-level arithmetic directly summing to 94 across all 10 categories ($105$ required in A vs $11$ in C = $94$ avoided, 89.52% reduction), ensuring omitted or unretrieved facts increase Condition C's needed explanations and never count as avoided.
- [x] **Cluster-Stratified Analysis & Exact Student's t Statistical Qualification**: Paired exact McNemar's tests on binary task success (A vs B: $p < 0.001$, A vs C: $p < 0.001$, B vs C: $p = 0.3877$), paired Student's t-tests, and cluster-stratified category analysis ($K=10$ clusters, $df=9$) reporting cluster standard errors ($SE = 0.104822$), cluster $t$-statistics ($t = 6.6780$), numerically evaluated two-tailed Student's t $p$-values ($p = 9.0787 \times 10^{-5}$ via composite Simpson's rule numerical integration), and full-precision 95% confidence intervals using the exact Student's t critical value ($t_{\text{crit}} = 2.262157162798205$ at $df=9$, e.g. Task Success $[+0.462876, +0.937124]$ and Token Overhead $[142.9715875613, 600.6950791054]$, presented as $[142.9716, 600.6951]$ and rounded to two decimals as $[142.97, 600.70]$ with numerically evaluated two-tailed Student's t $p$-value = $0.00511227$). Explicitly qualified as exploratory and conditional on treating categories as independent clusters; token difference ($+371.833333$ tokens, $C - A$) explicitly defined as additional token overhead rather than performance improvement. Summary statistics and regression tests verify exact mathematical agreement across all five metrics.
- [x] **Automated Reporting Framework & Full Precision Storage**: Machine-readable JSON (`benchmarks_report.json`) and Markdown (`benchmarks_report.md`) exports displaying at least four decimal places for all category differences, preferring six decimal places for repeating fractions, and storing unrounded full-precision IEEE-754 floats in JSON. Displayed vectors reproduce reported means, standard deviations, standard errors, t-statistics, and full-precision 95% confidence intervals within $< 10^{-6}$ tolerance.
- [x] **Full Regression Verification**: All 316 prior tests preserved + 21 benchmark tests + 6 storage/loop lifecycle tests = 343 tests collected: 339 passed, 4 live provider tests skipped offline. All 8 demonstrations working offline.

### Comprehensive Audit & Resource Lifecycle Verification (Phases 1–4)
- [x] **Primary Resource-Management Architecture**: Primary resource cleanup is governed strictly by explicit `.close()` calls and standard Python context management protocols (`with` statements). The `__del__` finalizer is retained strictly as a fallback / safety net for orphaned garbage collection and is never relied upon as the primary cleanup mechanism.
- [x] **Close Idempotency & Defense**: Confirmed and verified that `.close()` is fully idempotent across all 6 persistent storage classes (`SQLiteEventStore`, `SQLiteInteractionStore`, `SQLiteVectorStore`, `SQLiteAgentRegistryStore`, `SQLiteSessionStore`, `SQLiteFeedbackStore`) and `BrainLoop`. Repeated calls are safe no-ops.
- [x] **Exception Safety in Context Managers**: All storage classes and `BrainLoop` implement `__enter__` and `__exit__`. Context managers guarantee immediate resource closure on both normal exit and unhandled exception paths.
- [x] **BenchmarkRunner Exception-Safe Cleanup**: Wrapped `run_condition_c_brain` in comprehensive `try ... finally` blocks, guaranteeing that `loop`, `ev_store`, and `vec_store` are reliably closed regardless of whether execution completes successfully or raises an exception.
- [x] **Partially Initialized Instance Safety**: Finalizers and `.close()` methods guard against uninitialized attributes (e.g. `_conn`, `_lock`), preventing secondary `AttributeError` exceptions if object initialization fails partway through.
- [x] **Audited Zero-Leakage Test Execution**: The entire test suite executes with zero warnings under strict elevation (`python -W error::ResourceWarning -m unittest discover -s tests -v`).

### Accurate Phase Relationships & Component Dependency Architecture

#### Phase Relationship Model
```
  +-------------------------------------------------------------------------+
  |              Phase 1: Core Brain Architecture (Foundational)            |
  |  (Events, Memories, Updating, Conflicts, Graph, Vectors, Context, Loop) |
  +--------------------+--------------------------------+-------------------+
                       |                                |
                       v                                v
  +------------------------------------+  +---------------------------------+
  |    Phase 2: Multi-Agent System     |  | Phase 3: Machine Learning Layer |
  | (Registry, Sessions, Switcher,     |  | (Features, Calibrated Models,   |
  |  Adapters, Collaboration Engine)   |  |  Grouped CV, Feedback Store)    |
  +--------------------+---------------+  +----------------+----------------+
                       |                                   |
                       +-----------------+-----------------+
                                         |
                                         v
  +-------------------------------------------------------------------------+
  |             Phase 4: System Benchmarking & Comparative Framework        |
  |  (30 Scenarios, Conditions A/B/C, McNemar, Cluster t-Stats, Reporting)  |
  +--------------------------------------+----------------------------------+
                                         |
                                         v
  +-------------------------------------------------------------------------+
  |    Phase 5: Production Backend & Infrastructure (Planned / Evaluation)  |
  |  (Production DB, API Service Layer, Auth/RBAC, Workers, Observability)  |
  +-------------------------------------------------------------------------+
```

#### Decoupled Component Dependency Architecture
```
                         +-----------------------------+
                         |      Append-Only Events     |
                         |     (SQLiteEventStore)      |
                         +--------------+--------------+
                                        |
                   +--------------------+--------------------+
                   v                                         v
  +---------------------------------+       +---------------------------------+
  |          Memory Models          |       |        Multi-Agent System       |
  | (Versioning, Extraction, Update)|       | (Registry, Sessions, Adapters)  |
  +----------------+----------------+       +----------------+----------------+
                   |                                         |
         +---------+---------+                               |
         v                   v                               |
+-----------------+ +-----------------+                      |
| Knowledge Graph | |  Vector Memory  |                      |
|  (Traversals)   | | (Semantic Cosine)                      |
+--------+--------+ +--------+--------+                      |
         |                   |                               |
         +---------+---------+                               |
                   v                                         v
        +---------------------+                   +---------------------+
        |  Retrieval Engine   |                   |  Brain Loop Facade  |
        |  (Hybrid RRF, ML)   |<------------------| (11-Stage Lifecycle)|
        +----------+----------+                   +----------+----------+
                   |                                         |
                   v                                         |
        +---------------------+                              |
        |   Context Builder   |                              |
        | (Knapsack, Markdown)|<-----------------------------+
        +----------+----------+                              |
                   |                                         |
                   +--------------------+--------------------+
                                        v
                         +-----------------------------+
                         |    Evaluation & Benchmarks  |
                         | (Conditions A/B/C, Reporter)|
                         +-----------------------------+
```

### Phase 5: Architecture Evaluation & Trade-Off Analysis (Planned / Unimplemented)
> **Status:** Architecture Designed & Trade-Off Options Evaluated (Implementation Has Not Started).
> **Implementation Directive:** Do not assume or prescribe high-complexity tools (e.g. TimescaleDB, Redis Redlock, gRPC, Qdrant, Temporal, Kubernetes) without demonstrating technical requirements that justify their operational complexity.

The following architectural trade-offs require formal evaluation during Phase 5 design:

#### 1. PostgreSQL Native Partitioning vs. TimescaleDB
- **Option A (PostgreSQL Native Declarative Partitioning — Recommended Initial Choice)**:
  - *Requirements*: Standard PostgreSQL 16+ without specialized kernel extensions. Partition `events` table by range (`timestamp`) and sub-partition by list (`project_id`).
  - *Trade-Offs*: Zero external operational dependencies, standard backup/restore tooling (`pg_dump`/`pg_basebackup`), standard Alembic migrations. Lacks automated hypertable chunk compression and native continuous aggregates.
  - *Operational Complexity*: Low.
  - *Justification Conditions*: Sufficient for event ingestion throughput up to thousands of events/sec and standard table sizes (<50M rows).
- **Option B (TimescaleDB Extension)**:
  - *Requirements*: TimescaleDB extension compiled and enabled in PostgreSQL.
  - *Trade-Offs*: Automated chunk creation, 90%+ transparent time-series compression, high-performance continuous aggregates. Introduces cloud lock-in / portability constraints, custom extension build dependencies, and specialized upgrade procedures.
  - *Operational Complexity*: Medium to High.
  - *Justification Conditions*: Justified only if append-only event volume reaches massive scale (>10M events/day) where disk footprint and automated chunk dropping become acute constraints.

#### 2. PostgreSQL In-Database Concurrency vs. Redis Distributed Locking (Redlock)
- **Option A (PostgreSQL Transactions, Optimistic Concurrency & Advisory Locks — Recommended Initial Choice)**:
  - *Requirements*: PostgreSQL native transactions (`READ COMMITTED` / `REPEATABLE READ`), version columns on memory records (`OCC`), and session/transaction-scoped advisory locks (`pg_advisory_xact_lock`).
  - *Trade-Offs*: Zero extra infrastructure dependencies; locks release automatically upon transaction rollback or connection termination, eliminating orphaned lock deadlocks. Guarantees transactional atomicity across events, memories, and locks.
  - *Operational Complexity*: Minimal (single operational datastore to monitor and backup).
  - *Justification Conditions*: Strongly recommended for all multi-agent coordination within standard scale (up to thousands of concurrent agent sessions).
- **Option B (Redis Distributed Locking / Redlock)**:
  - *Requirements*: Independent Redis cluster (minimum 3–5 independent master nodes for true Redlock algorithm safety).
  - *Trade-Offs*: Sub-millisecond lock acquisition latency, offloads lock contention from relational database. However, introduces clock-drift vulnerabilities, network partition split-brain risks, split-state failure modes between Redis and PostgreSQL, and additional infrastructure overhead.
  - *Operational Complexity*: High.
  - *Justification Conditions*: Justified only if lock churn threatens to exhaust PostgreSQL connection pool capacity (>10,000 lock acquisitions/sec across decoupled microservices).

#### 3. REST-Only vs. REST + gRPC Dual Interface
- **Option A (REST-Only via FastAPI / OpenAPI — Recommended Initial Choice)**:
  - *Requirements*: Standard HTTP/1.1 or HTTP/2 JSON endpoints with Server-Sent Events (SSE) for streaming agent responses.
  - *Trade-Offs*: Universal interoperability across languages, API gateways, web browsers, and CLI tooling. Zero protocol compilation steps. Slightly higher payload parsing overhead than binary serialization.
  - *Operational Complexity*: Low.
  - *Justification Conditions*: Recommended default for initial service layer, fully supporting all agent interaction streams and webhooks.
- **Option B (REST + gRPC Dual Interface)**:
  - *Requirements*: Protocol Buffers (proto3) schema compiler, HTTP/2 multiplexing, dual-stack endpoint definitions.
  - *Trade-Offs*: High binary throughput, strongly-typed multi-language SDK code generation, bidirectional streaming. Requires specialized HTTP/2 trailing metadata proxying, complicates browser/curl debugging, and requires maintaining dual API contracts.
  - *Operational Complexity*: High.
  - *Justification Conditions*: Justified only if internal inter-service latency (<5ms) and high-frequency binary event ingestion become demonstrated bottlenecks across a microservice fleet.

#### 4. pgvector vs. Dedicated External Vector Database (e.g. Qdrant / Milvus)
- **Option A (pgvector in PostgreSQL — Recommended Initial Choice)**:
  - *Requirements*: `pgvector` extension enabled in PostgreSQL (HNSW indexing).
  - *Trade-Offs*: Zero data drift: vector embeddings, relational memory entities, and event provenance reside in the same ACID database. Allows transactional vector inserts in the same transaction as memory creation. Enables combined relational SQL filtering (`WHERE project_id = ... AND status = 'ACTIVE'`) with HNSW cosine search.
  - *Operational Complexity*: Low (standard database backups protect relational data and vectors simultaneously).
  - *Justification Conditions*: Recommended default for collections up to several million embeddings.
- **Option B (Dedicated Vector DB Cluster — Qdrant / Milvus)**:
  - *Requirements*: Independent vector database cluster, background synchronization pipeline, dual-write consistency management.
  - *Trade-Offs*: Advanced vector-specific indexing features (scalar/product quantization, disk-backed payloads, decoupled vector compute scaling). Introduces split-brain risk, dual-write synchronization lag (eventual consistency), separate backups, and complex rollbacks.
  - *Operational Complexity*: High.
  - *Justification Conditions*: Justified only if vector collections exceed tens of millions of embeddings or require specialized GPU-accelerated indexing.

#### 5. Lightweight Background Processing vs. Celery vs. Temporal
- **Option A (In-Process Asyncio Workers or RQ — Recommended Initial Choice)**:
  - *Requirements*: Python `asyncio.create_task` worker loops or lightweight Redis Queue (RQ).
  - *Trade-Offs*: Minimal operational footprint, easy local debugging, fast startup, low memory overhead. Lacks complex workflow choreography, multi-hour saga rollbacks, and distributed retry trees.
  - *Operational Complexity*: Low.
  - *Justification Conditions*: Recommended for asynchronous memory extraction, vector indexing, and status notification tasks.
- **Option B (Celery Task Queue)**:
  - *Requirements*: Celery worker processes, message broker (RabbitMQ/Redis), result backend.
  - *Trade-Offs*: Proven task queue with rate limiting, task retries, cron schedules (Celery Beat), and worker concurrency pools. Requires broker management, tricky serialization of custom domain models, and monitoring overhead (Flower).
  - *Operational Complexity*: Medium.
  - *Justification Conditions*: Suitable if background tasks require distributed rate limiting and isolated worker node pools.
- **Option C (Temporal Workflow Engine)**:
  - *Requirements*: Temporal server cluster, PostgreSQL/Cassandra state persistence, and Temporal SDK worker fleet.
  - *Trade-Offs*: Deterministic workflow orchestration, persistent execution state across server restarts, automatic activity retries with exponential backoff, and stateful sagas. Steep learning curve, high architectural complexity, and dedicated infrastructure.
  - *Operational Complexity*: Very High.
  - *Justification Conditions*: Justified only if long-running multi-day human-in-the-loop multi-agent workflows with stateful compensation transactions are mandated.
