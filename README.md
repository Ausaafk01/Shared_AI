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

Run all 288 automated tests across the Event System, Memory Model, Memory Extraction, Memory Updating, Conflict Management, Knowledge Graph, Vector Memory, Retrieval Engine, Context Builder, Brain Loop, and Multi-Agent System:

```bash
python -m unittest discover -s tests -v
```

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
