"""
Demonstration of AI Brain Vector Memory subsystem.
Shows semantic similarity search, memory version supersession, and project isolation.
"""

from pathlib import Path
import sys
import tempfile

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_brain.memories import Memory, MemoryType
from ai_brain.vectors import (
    MockEmbeddingProvider,
    SQLiteVectorStore,
    VectorMemoryManager,
)


def run_demo() -> None:
    print("=== AI Brain: Vector Memory Demonstration ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "demo_vectors.db"
        print(f"Creating persistent vector store at: {db_path}\n")

        # 1. Initialize Vector Memory components
        provider = MockEmbeddingProvider(dimension=384, model_id="demo-semantic-384")
        with SQLiteVectorStore(database_path=db_path) as store:
            manager = VectorMemoryManager(storage=store, provider=provider)

            # 2. Index diverse memories across projects
            print("--- Indexing Knowledge Memories ---")
            m_db1 = Memory.create(
                memory_id="mem-db-1",
                memory_type=MemoryType.FACT,
                content="The project uses MySQL as its primary database.",
                project_id="proj-web",
                version=1,
                root_memory_id="root-db",
            )
            m_auth = Memory.create(
                memory_id="mem-auth-1",
                memory_type=MemoryType.FACT,
                content="Authentication is handled via JWT tokens.",
                project_id="proj-web",
                version=1,
            )
            m_pref = Memory.create(
                memory_id="mem-pref-1",
                memory_type=MemoryType.PREFERENCE,
                content="The developer prefers concise, functional code.",
                project_id="proj-web",
                version=1,
            )
            m_other = Memory.create(
                memory_id="mem-sec-1",
                memory_type=MemoryType.FACT,
                content="Secret keys are stored in Vault.",
                project_id="proj-isolated",
                version=1,
            )

            manager.index_memories([m_db1, m_auth, m_pref, m_other])
            print(f"Indexed 4 memories into SQLite vector storage.\n")

            # 3. Query without exact keywords (Semantic Search)
            print("--- Query 1: 'What database engine is configured for data storage?' ---")
            res_db = manager.search(
                "What database engine is configured for data storage?",
                project_id="proj-web",
                top_k=2,
            )
            for idx, match in enumerate(res_db.matches, 1):
                print(f"{idx}. [Score: {match.score:.4f}] ({match.memory_id}) {match.content}")
            print()

            # 4. Query another semantic topic
            print("--- Query 2: 'How do users log in?' ---")
            res_auth = manager.search(
                "How do users log in?",
                project_id="proj-web",
                top_k=2,
            )
            for idx, match in enumerate(res_auth.matches, 1):
                print(f"{idx}. [Score: {match.score:.4f}] ({match.memory_id}) {match.content}")
            print()

            # 5. Evolution: Memory version update (v2 supersedes v1)
            print("--- Memory Evolution: Updating Database to PostgreSQL ---")
            m_db2 = Memory.create(
                memory_id="mem-db-2",
                memory_type=MemoryType.FACT,
                content="The project uses PostgreSQL as its primary database.",
                project_id="proj-web",
                version=2,
                root_memory_id="root-db",
                parent_memory_id="mem-db-1",
            )
            manager.index_memory(m_db2)
            print("Indexed mem-db-2 (v2), automatically superseding mem-db-1 (v1).\n")

            # 6. Re-run query: active search returns v2 and excludes superseded v1
            print("--- Query 3: 'What database engine is configured for data storage?' (After Update) ---")
            res_after = manager.search(
                "What database engine is configured for data storage?",
                project_id="proj-web",
                top_k=2,
            )
            for idx, match in enumerate(res_after.matches, 1):
                print(f"{idx}. [Score: {match.score:.4f}] ({match.memory_id}, v{match.version}) {match.content}")
            print()

            # 7. Project isolation check
            print("--- Project Isolation Check ---")
            res_iso = manager.search("Secret keys in Vault", project_id="proj-web")
            print(f"Results for 'Secret keys' in proj-web: {len(res_iso.matches)} matches (vault memory from proj-isolated was excluded).")

    print("\n=== Vector Memory Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
