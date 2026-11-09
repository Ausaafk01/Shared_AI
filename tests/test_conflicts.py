"""
Automated tests for AI Brain Conflict Management component.
"""

from datetime import datetime, timezone, timedelta
import unittest

from ai_brain.memories.conflicts.detector import ConflictDetector
from ai_brain.memories.conflicts.evaluator import ConflictEvaluator
from ai_brain.memories.conflicts.manager import ConflictManager
from ai_brain.memories.conflicts.models import MemoryConflict
from ai_brain.memories.conflicts.types import (
    ConflictStatus,
    ConflictType,
)
from ai_brain.memories.exceptions import (
    ConflictValidationError,
    InvalidConflictStateError,
)
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus, MemoryType


class TestConflictManagement(unittest.TestCase):
    """Tests covering conflict detection, classification, evidence evaluation, and resolution."""

    def setUp(self) -> None:
        self.detector = ConflictDetector()
        self.evaluator = ConflictEvaluator()
        self.manager = ConflictManager(evaluator=self.evaluator)
        self.now = datetime(2026, 11, 2, 14, 0, 0, tzinfo=timezone.utc)

    def test_basic_factual_conflict_detection(self) -> None:
        """Test detecting a factual conflict between contradictory claims on database."""
        m1 = Memory.create(
            memory_id="mem-pg",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-db",
        )
        m2 = Memory.create(
            memory_id="mem-mysql",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-db",
        )

        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.project_id, "proj-db")
        self.assertIn("mem-pg", conflict.memory_ids)
        self.assertIn("mem-mysql", conflict.memory_ids)
        self.assertEqual(conflict.conflict_type, ConflictType.FACTUAL.value)
        self.assertEqual(conflict.status, ConflictStatus.DETECTED.value)

    def test_technical_configuration_conflict(self) -> None:
        """Test detecting a TECHNICAL conflict over port configuration."""
        m1 = Memory.create(
            memory_id="mem-cfg-1",
            memory_type=MemoryType.FACT,
            content="PostgreSQL database port is 5432.",
            project_id="proj-cfg",
        )
        m2 = Memory.create(
            memory_id="mem-cfg-2",
            memory_type=MemoryType.FACT,
            content="PostgreSQL database port is 3306.",
            project_id="proj-cfg",
        )

        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.conflict_type, ConflictType.TECHNICAL.value)

    def test_decision_conflict(self) -> None:
        """Test detecting a DECISION conflict between incompatible architectural decisions."""
        m1 = Memory.create(
            memory_id="mem-dec-1",
            memory_type=MemoryType.DECISION,
            content="Decided to use PostgreSQL for relational storage.",
            project_id="proj-dec",
        )
        m2 = Memory.create(
            memory_id="mem-dec-2",
            memory_type=MemoryType.DECISION,
            content="Decided to use MySQL for relational storage.",
            project_id="proj-dec",
        )

        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.conflict_type, ConflictType.DECISION.value)

    def test_temporal_disagreement_detection(self) -> None:
        """Test detecting a TEMPORAL conflict when a later memory describes migration."""
        m1 = Memory.create(
            memory_id="mem-old",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-mig",
            created_at=self.now,
        )
        m2 = Memory.create(
            memory_id="mem-migrated",
            memory_type=MemoryType.FACT,
            content="The database was migrated to PostgreSQL yesterday.",
            project_id="proj-mig",
            created_at=self.now + timedelta(days=1),
        )

        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.conflict_type, ConflictType.TEMPORAL.value)

    def test_task_status_conflict(self) -> None:
        """Test detecting a STATUS conflict when memories disagree on task completion."""
        m1 = Memory.create(
            memory_id="mem-task-active",
            memory_type=MemoryType.TASK,
            content="The authentication bug remains unresolved.",
            project_id="proj-task",
            status=MemoryStatus.ACTIVE,
        )
        m2 = Memory.create(
            memory_id="mem-task-archived",
            memory_type=MemoryType.TASK,
            content="Completed task: The authentication bug remains unresolved.",
            project_id="proj-task",
            status=MemoryStatus.ARCHIVED,
        )

        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.conflict_type, ConflictType.STATUS.value)

    def test_different_projects_remaining_isolated(self) -> None:
        """Verify that memories in different projects are never compared as conflicting."""
        m1 = Memory.create(
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="project-alpha",
        )
        m2 = Memory.create(
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="project-beta",
        )

        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNone(conflict)

        # Batch detection
        conflicts = self.detector.detect_many([m1, m2])
        self.assertEqual(len(conflicts), 0)

    def test_unrelated_memories_not_classified_as_conflicts(self) -> None:
        """Verify that unrelated memories on different subjects are not flagged."""
        m1 = Memory.create(
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-clean",
        )
        m2 = Memory.create(
            memory_type=MemoryType.PREFERENCE,
            content="The user prefers concise code.",
            project_id="proj-clean",
        )

        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNone(conflict)

    def test_duplicate_conflict_detection_prevention(self) -> None:
        """Verify that detecting across the same memories does not produce duplicate conflict records."""
        m1 = Memory.create(
            memory_id="mem-dup-1",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-dup",
        )
        m2 = Memory.create(
            memory_id="mem-dup-2",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-dup",
        )

        first_run = self.detector.detect_many([m1, m2])
        self.assertEqual(len(first_run), 1)

        # Running again without clearing cache does not duplicate
        second_run = self.detector.detect_many([m1, m2])
        self.assertEqual(len(second_run), 0)

    def test_provenance_preservation_in_conflict(self) -> None:
        """Verify that detected conflicts capture all source event IDs and agent IDs."""
        m1 = Memory.create(
            memory_id="m1-prov",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-prov",
            source_event_ids=["e1", "e2"],
            source_agent_ids=["claude"],
        )
        m2 = Memory.create(
            memory_id="m2-prov",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-prov",
            source_event_ids=["e3"],
            source_agent_ids=["gemini"],
        )

        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNotNone(conflict)
        evidence = conflict.evidence
        self.assertIn("source_event_ids", evidence)
        self.assertIn("source_agent_ids", evidence)
        self.assertEqual(set(evidence["source_event_ids"]), {"e1", "e2", "e3"})
        self.assertEqual(set(evidence["source_agent_ids"]), {"claude", "gemini"})

    def test_evidence_evaluation_insufficient(self) -> None:
        """Test evaluation when evidence is balanced/inconclusive."""
        m1 = Memory.create(
            memory_id="m-inconc-1",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-eval",
            source_event_ids=["e1"],
            source_agent_ids=["claude"],
            confidence=0.9,
        )
        m2 = Memory.create(
            memory_id="m-inconc-2",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-eval",
            source_event_ids=["e2"],
            source_agent_ids=["gemini"],
            confidence=0.9,
        )
        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNotNone(conflict)

        assessment = self.evaluator.evaluate(conflict, [m1, m2])
        self.assertFalse(assessment.is_resolvable)
        self.assertEqual(assessment.recommended_status, ConflictStatus.UNRESOLVED)
        self.assertIsNone(assessment.winning_memory_id)

    def test_evidence_supported_resolution(self) -> None:
        """Test evaluation when one memory has overwhelming corroboration and verified evidence."""
        m1_strong = Memory.create(
            memory_id="m-strong",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-eval",
            source_event_ids=["e1", "e2", "e3"],
            source_agent_ids=["claude", "antigravity"],
            confidence=1.0,
            metadata={"verified": True},
        )
        m2_weak = Memory.create(
            memory_id="m-weak",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-eval",
            source_event_ids=["e4"],
            source_agent_ids=["gemini"],
            confidence=0.6,
        )
        conflict = self.detector.detect_pair(m1_strong, m2_weak)
        self.assertIsNotNone(conflict)

        assessment = self.evaluator.evaluate(conflict, [m1_strong, m2_weak])
        self.assertTrue(assessment.is_resolvable)
        self.assertEqual(assessment.recommended_status, ConflictStatus.RESOLVED)
        self.assertEqual(assessment.winning_memory_id, "m-strong")
        self.assertEqual(assessment.losing_memory_ids, ("m-weak",))

    def test_resolution_supersession_through_existing_updater(self) -> None:
        """Test resolving conflict: updates conflict status and supersedes losing memory."""
        m1_winner = Memory.create(
            memory_id="win-mem",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-res",
            source_event_ids=["e1", "e2", "e3"],
            source_agent_ids=["claude", "codex"],
            metadata={"verified": True},
        )
        m2_loser = Memory.create(
            memory_id="lose-mem",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-res",
            source_event_ids=["e4"],
            source_agent_ids=["gemini"],
        )

        conflict = self.detector.detect_pair(m1_winner, m2_loser)
        self.assertIsNotNone(conflict)

        assessment = self.evaluator.evaluate(conflict, [m1_winner, m2_loser])
        self.assertTrue(assessment.is_resolvable)

        # Resolve via ConflictManager
        result = self.manager.resolve(conflict, assessment, [m1_winner, m2_loser])
        self.assertEqual(result.status, ConflictStatus.RESOLVED)
        self.assertEqual(result.action_taken, "RESOLVED_VIA_EVIDENCE")

        # Check superseded loser
        self.assertEqual(len(result.superseded_memories), 1)
        self.assertEqual(result.superseded_memories[0].status, MemoryStatus.SUPERSEDED.value)

        # Check conflict record updated
        tracked_conflict = self.manager.get_conflict(conflict.conflict_id)
        self.assertIsNotNone(tracked_conflict)
        self.assertEqual(tracked_conflict.status, ConflictStatus.RESOLVED.value)
        self.assertEqual(tracked_conflict.resolution["winner_id"], "win-mem")

    def test_unresolved_conflict_preserves_disputed_status(self) -> None:
        """Test that unresolved conflicts preserve claims marked as DISPUTED."""
        m1 = Memory.create(
            memory_id="m1-unres",
            memory_type=MemoryType.FACT,
            content="Authentication is handled through JWT.",
            project_id="proj-unres",
        )
        m2 = Memory.create(
            memory_id="m2-unres",
            memory_type=MemoryType.FACT,
            content="Authentication is handled through Session Cookies.",
            project_id="proj-unres",
        )

        conflict = self.detector.detect_pair(m1, m2)
        assessment = self.evaluator.evaluate(conflict, [m1, m2])
        self.assertFalse(assessment.is_resolvable)

        result = self.manager.resolve(conflict, assessment, [m1, m2])
        self.assertEqual(result.status, ConflictStatus.UNRESOLVED)
        self.assertEqual(result.action_taken, "PRESERVED_AS_DISPUTED")

        # All affected memories have status DISPUTED
        for m in result.affected_memories:
            self.assertEqual(m.status, MemoryStatus.DISPUTED.value)

    def test_dismissed_conflict_handling(self) -> None:
        """Test dismissing a conflict that is not a genuine disagreement."""
        m1 = Memory.create(memory_id="m1-dis", memory_type=MemoryType.FACT, content="C1", project_id="p")
        conflict = MemoryConflict.create(
            project_id="p",
            memory_ids=["m1-dis", "m2-dis"],
            conflict_type=ConflictType.FACTUAL,
            description="False alarm",
        )

        res = self.manager.dismiss(conflict, reason="Claims describe complementary layers.")
        self.assertEqual(res.status, ConflictStatus.DISMISSED)
        self.assertEqual(res.action_taken, "DISMISSED")

        tracked = self.manager.get_conflict(conflict.conflict_id)
        self.assertEqual(tracked.status, ConflictStatus.DISMISSED.value)
        self.assertIn("complementary", tracked.resolution["dismissal_reason"])

    def test_conflict_lifecycle_transitions(self) -> None:
        """Test valid and invalid conflict lifecycle status transitions."""
        conflict = MemoryConflict.create(
            project_id="p",
            memory_ids=["m1", "m2"],
            conflict_type=ConflictType.FACTUAL,
            description="Testing transitions",
        )
        self.assertEqual(conflict.status, ConflictStatus.DETECTED.value)

        # DETECTED -> UNDER_REVIEW
        c_review = conflict.with_status(ConflictStatus.UNDER_REVIEW)
        self.assertEqual(c_review.status, ConflictStatus.UNDER_REVIEW.value)

        # UNDER_REVIEW -> RESOLVED
        c_resolved = c_review.with_status(ConflictStatus.RESOLVED)
        self.assertEqual(c_resolved.status, ConflictStatus.RESOLVED.value)

        # Illegal transition: RESOLVED directly to DISMISSED raises InvalidConflictStateError
        with self.assertRaises(InvalidConflictStateError):
            c_resolved.with_status(ConflictStatus.DISMISSED)

    def test_conflict_validation_errors(self) -> None:
        """Test that invalid fields in MemoryConflict raise ConflictValidationError."""
        # Less than 2 memory IDs
        with self.assertRaises(ConflictValidationError):
            MemoryConflict.create(
                project_id="p",
                memory_ids=["only-one"],
                conflict_type=ConflictType.FACTUAL,
                description="desc",
            )

        # Empty project_id
        with self.assertRaises(ConflictValidationError):
            MemoryConflict.create(
                project_id="",
                memory_ids=["m1", "m2"],
                conflict_type=ConflictType.FACTUAL,
                description="desc",
            )

        # Confidence out of range
        with self.assertRaises(ConflictValidationError):
            MemoryConflict.create(
                project_id="p",
                memory_ids=["m1", "m2"],
                conflict_type=ConflictType.FACTUAL,
                description="desc",
                confidence=1.5,
            )

    def test_conflict_serialization_roundtrip(self) -> None:
        """Test to_dict, from_dict, to_json, and from_json for MemoryConflict."""
        conflict = MemoryConflict.create(
            conflict_id="conf-serial-1",
            project_id="proj-serial",
            memory_ids=["mem-1", "mem-2"],
            conflict_type=ConflictType.FACTUAL,
            description="Serialization test",
            evidence={"source_events": ["e1", "e2"]},
            confidence=0.88,
            metadata={"domain": "database"},
        )

        # Dict roundtrip
        d = conflict.to_dict()
        reconstructed_dict = MemoryConflict.from_dict(d)
        self.assertEqual(reconstructed_dict.conflict_id, conflict.conflict_id)
        self.assertEqual(reconstructed_dict.project_id, conflict.project_id)
        self.assertEqual(reconstructed_dict.memory_ids, conflict.memory_ids)
        self.assertEqual(reconstructed_dict.conflict_type, conflict.conflict_type)
        self.assertEqual(reconstructed_dict.confidence, conflict.confidence)

        # JSON roundtrip
        j = conflict.to_json()
        reconstructed_json = MemoryConflict.from_json(j)
        self.assertEqual(reconstructed_json.conflict_id, conflict.conflict_id)
        self.assertEqual(reconstructed_json.description, conflict.description)

    def test_multiple_conflicts_within_a_project(self) -> None:
        """Test detecting multiple separate conflicts on different subjects in the same project."""
        # Conflict 1: Database
        m_db1 = Memory.create(memory_id="db1", memory_type=MemoryType.FACT, content="The project uses PostgreSQL as its database.", project_id="p-multi")
        m_db2 = Memory.create(memory_id="db2", memory_type=MemoryType.FACT, content="The project uses MySQL as its database.", project_id="p-multi")

        # Conflict 2: Auth
        m_auth1 = Memory.create(memory_id="auth1", memory_type=MemoryType.FACT, content="Authentication is handled through JWT.", project_id="p-multi")
        m_auth2 = Memory.create(memory_id="auth2", memory_type=MemoryType.FACT, content="Authentication is handled through Session Cookies.", project_id="p-multi")

        conflicts = self.detector.detect_many([m_db1, m_db2, m_auth1, m_auth2])
        self.assertEqual(len(conflicts), 2)
        types = {c.conflict_type for c in conflicts}
        self.assertEqual(types, {ConflictType.FACTUAL.value})

    def test_disjoint_agents_contradictory_factual_claims_classified_as_factual(self) -> None:
        """Two different agents making contradictory factual claims: classify as FACTUAL, not SOURCE."""
        m_claude = Memory.create(
            memory_id="m-claude",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-src",
            source_agent_ids=["claude"],
        )
        m_gemini = Memory.create(
            memory_id="m-gemini",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-src",
            source_agent_ids=["gemini"],
        )

        conflict = self.detector.detect_pair(m_claude, m_gemini)
        self.assertIsNotNone(conflict)
        # Disjoint agent sets must NOT independently trigger SOURCE; classified by actual subject
        self.assertEqual(conflict.conflict_type, ConflictType.FACTUAL.value)
        # Provenance of both agents is preserved in evidence
        self.assertEqual(set(conflict.evidence["source_agent_ids"]), {"claude", "gemini"})

    def test_different_agents_independently_supporting_same_claim_no_conflict(self) -> None:
        """Two different agents independently supporting the same claim: treated as corroboration, no conflict."""
        m_claude = Memory.create(
            memory_id="m-corrob-1",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-corrob",
            source_agent_ids=["claude"],
        )
        m_gemini = Memory.create(
            memory_id="m-corrob-2",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-corrob",
            source_agent_ids=["gemini"],
        )

        conflict = self.detector.detect_pair(m_claude, m_gemini)
        self.assertIsNone(conflict)

    def test_actual_provenance_disagreement_classified_as_source(self) -> None:
        """An actual provenance disagreement: classify as SOURCE."""
        m_auth1 = Memory.create(
            memory_id="m-prov-1",
            memory_type=MemoryType.FACT,
            content="The database migration script was authored by Claude.",
            project_id="proj-prov-dispute",
            source_agent_ids=["claude"],
        )
        m_auth2 = Memory.create(
            memory_id="m-prov-2",
            memory_type=MemoryType.FACT,
            content="The database migration script was authored by Codex.",
            project_id="proj-prov-dispute",
            source_agent_ids=["gemini"],
        )

        conflict = self.detector.detect_pair(m_auth1, m_auth2)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.conflict_type, ConflictType.SOURCE.value)

    def test_different_agent_sets_without_contradictory_claims_no_conflict(self) -> None:
        """Different agent sets without contradictory claims: no conflict."""
        m1 = Memory.create(
            memory_id="m-diff-1",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-diff-agents",
            source_agent_ids=["claude"],
        )
        m2 = Memory.create(
            memory_id="m-diff-2",
            memory_type=MemoryType.FACT,
            content="Authentication is handled through JWT.",
            project_id="proj-diff-agents",
            source_agent_ids=["gemini", "codex"],
        )

        conflict = self.detector.detect_pair(m1, m2)
        self.assertIsNone(conflict)

    def test_many_repeated_source_ids_cannot_defeat_better_supported_claim(self) -> None:
        """A memory with many repeated source IDs does not automatically defeat a better-supported competing claim."""
        # Better supported memory: 2 independent corroborating agents and verified metadata
        m_supported = Memory.create(
            memory_id="m-supp",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-score",
            source_agent_ids=["claude", "codex"],
            source_event_ids=["evt-1"],
            metadata={"verified": True},
            confidence=0.8,
        )
        # Quantity spam memory: 1 agent, 20 source event IDs, but unverified
        m_spam = Memory.create(
            memory_id="m-spam",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-score",
            source_agent_ids=["gemini"],
            source_event_ids=[f"evt-spam-{i}" for i in range(20)],
            confidence=0.8,
        )

        conflict = self.detector.detect_pair(m_supported, m_spam)
        self.assertIsNotNone(conflict)

        assessment = self.evaluator.evaluate(conflict, [m_supported, m_spam])
        self.assertTrue(assessment.is_resolvable)
        self.assertEqual(assessment.winning_memory_id, "m-supp")
        self.assertEqual(assessment.losing_memory_ids, ("m-spam",))

    def test_independent_corroboration_improves_assessment(self) -> None:
        """Independent corroboration by multiple distinct agents can improve an assessment to resolvable."""
        m_single = Memory.create(
            memory_id="m-single",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-corrob-eval",
            source_agent_ids=["gemini"],
            source_event_ids=["e1"],
            confidence=0.8,
        )
        # With single agent on m_candidate: balanced / inconclusive
        m_candidate_1 = Memory.create(
            memory_id="m-cand-1",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-corrob-eval",
            source_agent_ids=["claude"],
            source_event_ids=["e2"],
            confidence=0.8,
        )
        c1 = self.detector.detect_pair(m_candidate_1, m_single)
        assessment1 = self.evaluator.evaluate(c1, [m_candidate_1, m_single])
        self.assertFalse(assessment1.is_resolvable)
        self.assertEqual(assessment1.recommended_status, ConflictStatus.UNRESOLVED)

        # Now add independent corroboration from codex to m_candidate
        m_candidate_2 = Memory.create(
            memory_id="m-cand-2",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-corrob-eval",
            source_agent_ids=["claude", "codex"],
            source_event_ids=["e2"],
            confidence=0.8,
        )
        c2 = self.detector.detect_pair(m_candidate_2, m_single)
        assessment2 = self.evaluator.evaluate(c2, [m_candidate_2, m_single])
        self.assertTrue(assessment2.is_resolvable)
        self.assertEqual(assessment2.winning_memory_id, "m-cand-2")

    def test_duplicate_evidence_does_not_inflate_confidence(self) -> None:
        """Duplicate evidence does not inflate score or assessment confidence."""
        m_clean = Memory.create(
            memory_id="m-clean",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-dedup-ev",
            source_agent_ids=["claude"],
            source_event_ids=["e1"],
        )
        m_dupes = Memory.create(
            memory_id="m-dupes",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-dedup-ev",
            source_agent_ids=["claude", "claude", "claude"],
            source_event_ids=["e1", "e1", "e1", "e1", "e1"],
        )
        m_comp = Memory.create(
            memory_id="m-comp",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-dedup-ev",
            source_agent_ids=["gemini"],
            source_event_ids=["e2"],
        )

        c1 = self.detector.detect_pair(m_clean, m_comp)
        a1 = self.evaluator.evaluate(c1, [m_clean, m_comp])

        c2 = self.detector.detect_pair(m_dupes, m_comp)
        a2 = self.evaluator.evaluate(c2, [m_dupes, m_comp])

        # Scores and confidence must be identical despite repeated duplicate IDs
        self.assertEqual(a1.evidence_scores["m-clean"], a2.evidence_scores["m-dupes"])
        self.assertEqual(a1.confidence, a2.confidence)

    def test_confidence_alone_cannot_force_resolution(self) -> None:
        """High self-reported confidence alone cannot force resolution without corroboration or verification."""
        m_overconfident = Memory.create(
            memory_id="m-overconf",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-conf-eval",
            source_agent_ids=["claude"],
            source_event_ids=["e1"],
            confidence=1.0,
        )
        m_modest = Memory.create(
            memory_id="m-modest",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-conf-eval",
            source_agent_ids=["gemini"],
            source_event_ids=["e2"],
            confidence=0.5,
        )

        conflict = self.detector.detect_pair(m_overconfident, m_modest)
        assessment = self.evaluator.evaluate(conflict, [m_overconfident, m_modest])
        # Margin from confidence alone is capped at 0.20, which is below the 0.80 resolution threshold
        self.assertFalse(assessment.is_resolvable)
        self.assertEqual(assessment.recommended_status, ConflictStatus.UNRESOLVED)
        self.assertIsNone(assessment.winning_memory_id)

    def test_insufficient_evidence_still_produces_unresolved(self) -> None:
        """When evidence is balanced/insufficient, evaluator consistently produces UNRESOLVED."""
        m1 = Memory.create(
            memory_id="m-bal-1",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-bal",
            source_agent_ids=["claude"],
            source_event_ids=["e1"],
            confidence=0.8,
        )
        m2 = Memory.create(
            memory_id="m-bal-2",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-bal",
            source_agent_ids=["gemini"],
            source_event_ids=["e2"],
            confidence=0.8,
        )

        conflict = self.detector.detect_pair(m1, m2)
        assessment = self.evaluator.evaluate(conflict, [m1, m2])
        self.assertFalse(assessment.is_resolvable)
        self.assertEqual(assessment.recommended_status, ConflictStatus.UNRESOLVED)
        self.assertEqual(assessment.confidence, 0.5)

    def test_temporal_migration_resolution(self) -> None:
        """Test that temporal migration conflict resolves in favor of the later migration."""
        m_early = Memory.create(
            memory_id="m-early",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-t-res",
            created_at=self.now,
        )
        m_migrated = Memory.create(
            memory_id="m-mig",
            memory_type=MemoryType.FACT,
            content="The database was migrated to PostgreSQL yesterday.",
            project_id="proj-t-res",
            created_at=self.now + timedelta(days=1),
        )

        conflict = self.detector.detect_pair(m_early, m_migrated)
        self.assertEqual(conflict.conflict_type, ConflictType.TEMPORAL.value)

        assessment = self.evaluator.evaluate(conflict, [m_early, m_migrated])
        self.assertTrue(assessment.is_resolvable)
        self.assertEqual(assessment.winning_memory_id, "m-mig")

        result = self.manager.resolve(conflict, assessment, [m_early, m_migrated])
        self.assertEqual(result.status, ConflictStatus.RESOLVED)


if __name__ == "__main__":
    unittest.main()

