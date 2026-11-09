"""
Unit and integration tests for external provider adapters in AI Brain.
Separates deterministic offline mock tests from live API/CLI integration tests.
"""

import os
import unittest
from unittest.mock import patch

from ai_brain.agents.adapters.antigravity import AntigravityAgentAdapter
from ai_brain.agents.adapters.claude import ClaudeAdapter
from ai_brain.agents.adapters.codex import CodexAgentAdapter, OpenAIModelAdapter
from ai_brain.agents.adapters.gemini import GeminiAdapter
from ai_brain.agents.exceptions import (
    AgentTimeoutError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from ai_brain.agents.models import AgentExecutionRequest
from ai_brain.context import ContextBuilder, DeterministicTokenizer
from ai_brain.events import SQLiteEventStore
from ai_brain.graph import GraphIntegrator, KnowledgeGraph
from ai_brain.loop import BrainLoop, SQLiteInteractionStore
from ai_brain.memories import (
    ConflictDetector,
    ConflictManager,
    MemoryExtractor,
    MemoryUpdater,
)
from ai_brain.retrieval import RetrievalEngine
from ai_brain.vectors import (
    MockEmbeddingProvider,
    SQLiteVectorStore,
    VectorIndex,
    VectorMemoryManager,
)


class TestProviderAdaptersMock(unittest.TestCase):
    """
    100% offline mock tests verifying:
    - Payload construction for each supported surface.
    - Secret isolation (zero credentials stored in records).
    - Error translation (rate limits, timeouts, auth errors).
    - End-to-end memory persistence through Brain Loop.
    """

    def setUp(self):
        self.event_store = SQLiteEventStore(":memory:")
        self.interaction_store = SQLiteInteractionStore(":memory:")

        self.vector_provider = MockEmbeddingProvider()
        self.vector_store = SQLiteVectorStore(":memory:")
        self.vector_index = VectorIndex(storage=self.vector_store, provider=self.vector_provider)
        self.vector_manager = VectorMemoryManager(storage=self.vector_store, provider=self.vector_provider)

        self.graph = KnowledgeGraph()
        self.graph_integrator = GraphIntegrator(self.graph)

        self.retrieval_engine = RetrievalEngine(
            vector_index=self.vector_index,
            graph=self.graph,
            allow_partial_results=True,
        )
        self.context_builder = ContextBuilder(
            tokenizer=DeterministicTokenizer(),
            default_max_tokens=600,
        )
        self.brain_loop = BrainLoop(
            event_store=self.event_store,
            interaction_store=self.interaction_store,
            retrieval_engine=self.retrieval_engine,
            context_builder=self.context_builder,
            memory_extractor=MemoryExtractor(),
            memory_updater=MemoryUpdater(),
            conflict_detector=ConflictDetector(),
            conflict_manager=ConflictManager(),
            graph=self.graph,
            graph_integrator=self.graph_integrator,
            vector_manager=self.vector_manager,
        )

    def tearDown(self):
        try:
            self.event_store.close()
        except Exception:
            pass
        try:
            self.interaction_store.close()
        except Exception:
            pass
        try:
            self.vector_store.close()
        except Exception:
            pass

    def test_claude_adapter_payload_format(self):
        """ClaudeAdapter formats payload into Anthropic Messages API format."""
        adapter = ClaudeAdapter(
            agent_id="claude-test",
            name="Claude Test",
            model="claude-3-5-sonnet-20241022",
            mock_transport=lambda p: "Claude mock response",
        )
        req = AgentExecutionRequest(
            project_id="proj-adapter",
            session_id="sess-claude",
            content="Please design a cache layer",
            query="cache design",
        )
        ctx = adapter.prepare_context(req, self.brain_loop)
        payload = adapter.build_provider_payload(ctx, req)

        self.assertEqual(payload["model"], "claude-3-5-sonnet-20241022")
        self.assertIn("messages", payload)
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "Please design a cache layer")
        self.assertIn("system", payload)
        self.assertIn("Claude Test", payload["system"])
        self.assertEqual(payload["max_tokens"], 1024)

    def test_gemini_adapter_payload_format(self):
        """GeminiAdapter formats payload into Google Gemini generateContent format."""
        adapter = GeminiAdapter(
            agent_id="gemini-test",
            name="Gemini Test",
            model="gemini-1.5-pro",
            mock_transport=lambda p: "Gemini mock response",
        )
        req = AgentExecutionRequest(
            project_id="proj-adapter",
            session_id="sess-gemini",
            content="Analyze security considerations",
            query="security",
        )
        ctx = adapter.prepare_context(req, self.brain_loop)
        payload = adapter.build_provider_payload(ctx, req)

        self.assertEqual(payload["model"], "gemini-1.5-pro")
        self.assertIn("contents", payload)
        self.assertEqual(payload["contents"][0]["role"], "user")
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "Analyze security considerations")
        self.assertIn("systemInstruction", payload)

    def test_codex_agent_cli_supported_flags_and_restrictive_default(self):
        """CodexAgentAdapter (CLI mode) defaults to restrictive read-only sandbox without --full-auto."""
        adapter = CodexAgentAdapter(
            agent_id="codex-cli-test",
            name="Codex CLI Agent",
            execution_mode="cli",
            cli_binary="codex",
            mock_transport=lambda p: "Codex CLI mock response",
        )
        req = AgentExecutionRequest(
            project_id="proj-codex",
            session_id="sess-codex",
            content="Inspect repository architecture",
            query="architecture",
        )
        ctx = adapter.prepare_context(req, self.brain_loop)
        payload = adapter.build_provider_payload(ctx, req)

        self.assertEqual(payload["command"], "codex")
        self.assertEqual(payload["subcommand"], "exec")
        # Restrictive default sandbox: read-only
        self.assertEqual(payload["sandbox_mode"], "read-only")
        self.assertIn("--sandbox", payload["flags"])
        self.assertIn("read-only", payload["flags"])
        # Non-interactive approval mode
        self.assertEqual(payload["approval_mode"], "never")
        self.assertIn("--ask-for-approval", payload["flags"])
        self.assertIn("never", payload["flags"])
        # Verification that deprecated --full-auto is completely removed
        self.assertNotIn("--full-auto", payload["flags"])
        self.assertNotIn("--full-auto", payload["args"])
        self.assertEqual(adapter.record.configuration["adapter_type"], "codex_agent_integration")

    def test_codex_agent_cli_workspace_write_and_approval_modes(self):
        """CodexAgentAdapter generates workspace-write when editing is needed, and supports on-request approval."""
        adapter = CodexAgentAdapter(
            agent_id="codex-write-test",
            name="Codex Writing Agent",
            execution_mode="cli",
            sandbox_mode="workspace-write",
            approval_mode="on-request",
            cli_binary="codex",
            mock_transport=lambda p: "Codex Write mock response",
        )
        req = AgentExecutionRequest(
            project_id="proj-codex",
            session_id="sess-codex",
            content="Refactor the authentication middleware",
            query="authentication middleware",
        )
        ctx = adapter.prepare_context(req, self.brain_loop)
        payload = adapter.build_provider_payload(ctx, req)

        self.assertEqual(payload["sandbox_mode"], "workspace-write")
        self.assertIn("workspace-write", payload["flags"])
        self.assertEqual(payload["approval_mode"], "on-request")
        self.assertIn("on-request", payload["flags"])
        self.assertIn("--sandbox", payload["args"])
        self.assertIn("workspace-write", payload["args"])
        self.assertIn("--ask-for-approval", payload["args"])
        self.assertIn("on-request", payload["args"])

        # Rejection of unrestricted execution
        with self.assertRaises(ValueError) as err_ctx:
            CodexAgentAdapter(agent_id="unsafe", sandbox_mode="danger-full-access")
        self.assertIn("prohibited", str(err_ctx.exception).lower())

    def test_openai_generic_model_adapter_payload_format(self):
        """OpenAIModelAdapter clearly identifies as a generic model API adapter across endpoints."""
        # 1. Chat Completions mode
        chat_adapter = OpenAIModelAdapter(
            agent_id="openai-chat-test",
            endpoint_mode="chat_completions",
            model="gpt-4o",
            mock_transport=lambda p: "GPT-4o response",
        )
        req = AgentExecutionRequest(
            project_id="proj-openai",
            session_id="sess-openai",
            content="Summarize architectural benefits",
            query="benefits",
        )
        ctx = chat_adapter.prepare_context(req, self.brain_loop)
        chat_payload = chat_adapter.build_provider_payload(ctx, req)

        self.assertEqual(chat_payload["endpoint"], "/v1/chat/completions")
        self.assertEqual(chat_payload["model"], "gpt-4o")
        self.assertIn("messages", chat_payload)
        self.assertEqual(len(chat_payload["messages"]), 2)
        self.assertEqual(chat_adapter.record.configuration["adapter_type"], "generic_model_api")

        # 2. Responses API inference mode
        resp_adapter = OpenAIModelAdapter(
            agent_id="openai-responses-test",
            endpoint_mode="responses",
            model="gpt-4o",
            mock_transport=lambda p: "Responses API output",
        )
        resp_payload = resp_adapter.build_provider_payload(ctx, req)
        self.assertEqual(resp_payload["endpoint"], "/v1/responses")
        self.assertEqual(resp_payload["model"], "gpt-4o")
        self.assertIn("tools", resp_payload)
        self.assertTrue(any(t["type"] == "code_interpreter" for t in resp_payload["tools"]))

    def test_antigravity_agent_interactions_api_payload_format(self):
        """AntigravityAgentAdapter (Interactions API mode) matches Google Gemini Interactions API spec."""
        adapter = AntigravityAgentAdapter(
            agent_id="ag-agent-test",
            name="Antigravity Agent",
            model="antigravity-preview-09-2026",
            execution_mode="interactions_api",
            environment="remote",
            mock_transport=lambda p: "Antigravity mock output",
        )
        req = AgentExecutionRequest(
            project_id="proj-ag",
            session_id="sess-ag",
            content="Run automated security checks",
            query="security checks",
        )
        ctx = adapter.prepare_context(req, self.brain_loop)
        payload = adapter.build_provider_payload(ctx, req)

        self.assertEqual(payload["endpoint"], "/v1beta/interactions")
        self.assertEqual(payload["agent"], "antigravity-preview-09-2026")
        self.assertEqual(payload["input"], "Run automated security checks")
        self.assertEqual(payload["environment"], "remote")
        self.assertEqual(adapter.record.configuration["adapter_type"], "antigravity_agent_integration")

    def test_antigravity_agent_cli_payload_format(self):
        """AntigravityAgentAdapter (CLI mode) matches the Antigravity CLI (agy) spec."""
        adapter = AntigravityAgentAdapter(
            agent_id="ag-cli-test",
            name="Antigravity CLI Agent",
            execution_mode="cli",
            cli_binary="agy",
            mock_transport=lambda p: "Antigravity CLI output",
        )
        req = AgentExecutionRequest(
            project_id="proj-ag",
            session_id="sess-ag",
            content="Refactor database models",
            query="database models",
        )
        ctx = adapter.prepare_context(req, self.brain_loop)
        payload = adapter.build_provider_payload(ctx, req)

        self.assertEqual(payload["command"], "agy")
        self.assertEqual(payload["prompt"], "Refactor database models")
        self.assertIn("context", payload)

    def test_authentication_validation_and_secrets_isolation(self):
        """Validates auth requirements and verifies zero credentials in records or configurations."""
        # 1. No key and no mock transport raises ProviderAuthenticationError
        adapter = ClaudeAdapter(
            agent_id="claude-live",
            name="Live Claude",
            mock_transport=None,
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderAuthenticationError):
                adapter.validate_authentication()

        # 2. Key provided via env passes validation
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-mock-key"}):
            adapter.validate_authentication()
            self.assertEqual(adapter.get_api_key(), "sk-ant-mock-key")

        # 3. Explicit key passes validation
        explicit_adapter = CodexAgentAdapter(
            agent_id="codex-explicit",
            api_key="sk-test-openai-key",
        )
        explicit_adapter.validate_authentication()
        self.assertEqual(explicit_adapter.get_api_key(), "sk-test-openai-key")

        # 4. Verify secrets are never stored in AgentRecord configuration
        self.assertNotIn("api_key", adapter.record.configuration)
        self.assertNotIn("ANTHROPIC_API_KEY", adapter.record.configuration)
        self.assertNotIn("api_key", explicit_adapter.record.configuration)

    def test_mock_transports_execution(self):
        """All adapters execute offline deterministically using mock transports."""
        captured_payloads = {}

        def make_transport(name):
            def transport(payload):
                captured_payloads[name] = payload
                return f"Simulated output from {name}"
            return transport

        adapters = [
            ClaudeAdapter(agent_id="claude", mock_transport=make_transport("claude")),
            CodexAgentAdapter(agent_id="codex", mock_transport=make_transport("codex")),
            OpenAIModelAdapter(agent_id="openai-model", mock_transport=make_transport("openai-model")),
            GeminiAdapter(agent_id="gemini", mock_transport=make_transport("gemini")),
            AntigravityAgentAdapter(agent_id="antigravity", mock_transport=make_transport("antigravity")),
        ]

        for adapter in adapters:
            req = AgentExecutionRequest(
                project_id="proj-mock",
                session_id="sess-mock",
                content=f"Hello to {adapter.name}",
            )
            resp = adapter.execute(req, self.brain_loop)
            self.assertIn(f"Simulated output from {adapter.agent_id}", resp.content)
            self.assertIn(adapter.agent_id, captured_payloads)

    def test_adapter_error_translations(self):
        """Provider rate limits, timeouts, and auth errors are cleanly mapped to domain exceptions."""
        def raise_err(e):
            raise e

        # Rate limit translation (429)
        rate_limited_adapter = ClaudeAdapter(
            agent_id="claude-ratelimit",
            mock_transport=lambda p: raise_err(RuntimeError("HTTP 429: Rate limit exceeded")),
        )
        req = AgentExecutionRequest(project_id="proj-err", session_id="sess-err", content="Heavy prompt")
        with self.assertRaises(ProviderRateLimitError):
            rate_limited_adapter.execute(req, self.brain_loop)

        # Timeout translation
        timeout_adapter = CodexAgentAdapter(
            agent_id="codex-timeout",
            mock_transport=lambda p: raise_err(TimeoutError("Read timed out after 30s")),
        )
        with self.assertRaises(AgentTimeoutError):
            timeout_adapter.execute(req, self.brain_loop)

        # Auth error translation (401)
        auth_err_adapter = GeminiAdapter(
            agent_id="gemini-autherr",
            mock_transport=lambda p: raise_err(RuntimeError("401 Unauthorized: Invalid API key")),
        )
        with self.assertRaises(ProviderAuthenticationError):
            auth_err_adapter.execute(req, self.brain_loop)

        # Connection error translation
        conn_err_adapter = AntigravityAgentAdapter(
            agent_id="ag-connerr",
            mock_transport=lambda p: raise_err(RuntimeError("Network connection reset by peer")),
        )
        with self.assertRaises(ProviderConnectionError):
            conn_err_adapter.execute(req, self.brain_loop)

    def test_adapter_end_to_end_knowledge_persistence(self):
        """Provider adapters persist memories back to AI Brain via complete BrainLoop lifecycle."""
        claude_resp = (
            "DECISION: We have decided to use Redis Sentinel for high-availability session caching. "
            "The cluster must have at least 3 nodes with a quorum of 2."
        )
        adapter = ClaudeAdapter(
            agent_id="claude-e2e",
            mock_transport=lambda p: claude_resp,
        )

        req = AgentExecutionRequest(
            project_id="proj-e2e",
            session_id="sess-e2e",
            content="What session caching strategy should we adopt?",
        )
        resp = adapter.execute(req, self.brain_loop)

        self.assertEqual(resp.content, claude_resp)
        self.assertIsNotNone(resp.completion_result)
        self.assertGreater(len(resp.completion_result.extracted_memories), 0)
        self.assertTrue(any("Redis Sentinel" in m.content for m in resp.completion_result.extracted_memories))


class TestProviderAdaptersLive(unittest.TestCase):
    """
    Live provider integration tests.
    Strictly separated from mock tests; automatically skipped unless explicit
    environment variables and credentials are provided.
    """

    @unittest.skipUnless(
        os.environ.get("AI_BRAIN_RUN_LIVE_TESTS") == "1" and os.environ.get("ANTHROPIC_API_KEY"),
        "Live Claude integration tests skipped: AI_BRAIN_RUN_LIVE_TESTS=1 and ANTHROPIC_API_KEY required.",
    )
    def test_live_claude_messages_api(self):
        adapter = ClaudeAdapter(agent_id="claude-live")
        adapter.validate_authentication()
        # Live invocation only runs when credentials exist
        req = AgentExecutionRequest(
            project_id="proj-live",
            session_id="sess-live",
            content="Ping. Reply with PONG.",
        )
        payload = adapter.build_provider_payload(None, req)
        res = adapter._execute_live_call(payload)
        self.assertIn("PONG", res.upper())

    @unittest.skipUnless(
        os.environ.get("AI_BRAIN_RUN_LIVE_TESTS") == "1" and os.environ.get("GEMINI_API_KEY"),
        "Live Gemini integration tests skipped: AI_BRAIN_RUN_LIVE_TESTS=1 and GEMINI_API_KEY required.",
    )
    def test_live_gemini_api(self):
        adapter = GeminiAdapter(agent_id="gemini-live")
        adapter.validate_authentication()
        req = AgentExecutionRequest(
            project_id="proj-live",
            session_id="sess-live",
            content="Ping. Reply with PONG.",
        )
        payload = adapter.build_provider_payload(None, req)
        res = adapter._execute_live_call(payload)
        self.assertIn("PONG", res.upper())

    @unittest.skipUnless(
        os.environ.get("AI_BRAIN_RUN_LIVE_TESTS") == "1" and os.environ.get("OPENAI_API_KEY"),
        "Live Codex/OpenAI integration tests skipped: AI_BRAIN_RUN_LIVE_TESTS=1 and OPENAI_API_KEY required.",
    )
    def test_live_codex_agent(self):
        adapter = CodexAgentAdapter(agent_id="codex-live", execution_mode="responses_api")
        adapter.validate_authentication()
        req = AgentExecutionRequest(
            project_id="proj-live",
            session_id="sess-live",
            content="Ping. Reply with PONG.",
        )
        payload = adapter.build_provider_payload(None, req)
        res = adapter._execute_live_call(payload)
        self.assertTrue(len(res) > 0)

    @unittest.skipUnless(
        os.environ.get("AI_BRAIN_RUN_LIVE_TESTS") == "1" and os.environ.get("GEMINI_API_KEY"),
        "Live Antigravity integration tests skipped: AI_BRAIN_RUN_LIVE_TESTS=1 and GEMINI_API_KEY required.",
    )
    def test_live_antigravity_agent(self):
        adapter = AntigravityAgentAdapter(agent_id="ag-live", execution_mode="interactions_api")
        adapter.validate_authentication()
        req = AgentExecutionRequest(
            project_id="proj-live",
            session_id="sess-live",
            content="Ping. Reply with PONG.",
        )
        payload = adapter.build_provider_payload(None, req)
        res = adapter._execute_live_call(payload)
        self.assertTrue(len(res) > 0)


if __name__ == "__main__":
    unittest.main()
