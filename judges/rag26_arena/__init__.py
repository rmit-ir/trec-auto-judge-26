"""TREC RAG 2026 specialist judge (RAGree-RAG26).

`judge.py` (`RAG26ArenaJudge`, `arena-only` variant) is the first runnable
piece: RAGDoll/TREC RAG 2026-style pairwise Arena preference judging via
SPARROW, no rubric conditioning or evidence gate yet. `rubrics/` (schema,
stats, validation) and `search_index.py`/`hybrid_retrieval.py` (the
ClimbMix-compatible search clients) remain foundation for the rubric
generation and citation-support lanes described in
docs/trec-rag-2026-judge-implementation-plan.md and
docs/trec-rag-2026-researchrubrics-generation-plan.md.
"""

from .judge import RAG26ArenaJudge, RAG26_ARENA_SPEC

__all__ = ["RAG26ArenaJudge", "RAG26_ARENA_SPEC"]
