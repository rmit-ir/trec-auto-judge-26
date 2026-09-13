#!/usr/bin/env python3
"""Live smoke check for the dense/sparse ClimbMix retrieval backends. NOT part of the offline
test suite (docs/trec-rag-2026-researchrubrics-generation-plan.md: "do not make the normal
offline test suite depend on either live service") -- run manually when a real endpoint is up.

Runs a small, fixed number of representative queries (no large sweep) and reports: health,
per-hit normalized metadata (chunk_id/parent_docid/backend/rank/score + a text snippet), and a
side-by-side of dense vs. sparse top hits for one semantic-leaning and one lexical-leaning query,
to eyeball that they're complementary rather than identical.

    RAG26_DENSE_SEARCH_URL=http://127.0.0.1:8088 \\
    RAG26_SPARSE_SEARCH_URL=http://127.0.0.1:8085 \\
    uv run python scripts/live_search_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from judges.rag26_arena.hybrid_retrieval import (  # noqa: E402
    HybridConfig,
    HybridResearchRetriever,
    ResearchIntent,
    parent_docid,
)
from judges.rag26_arena.search_index import (  # noqa: E402
    SearchNotConfiguredError,
    SearchUnavailableError,
    dense_client_from_env,
    sparse_client_from_env,
)

QUERIES = [
    ("semantic", "what are the broader economic and social causes behind a major political revolution"),
    ("lexical", "Ptolemaic dynasty Egypt"),
]


def _print_hits(label: str, hits) -> None:
    print(f"  {label}: {len(hits)} hit(s)")
    for h in hits[:3]:
        snippet = (h.text or "")[:80].replace("\n", " ")
        print(f"    docid={h.docid} parent={parent_docid(h.docid)} rank={h.rank} "
              f"score={h.score:.4f} text={snippet!r}")


def main() -> int:
    try:
        dense = dense_client_from_env()
    except SearchNotConfiguredError as exc:
        print(f"Dense: NOT CONFIGURED -- {exc}", file=sys.stderr)
        dense = None
    try:
        sparse = sparse_client_from_env()
    except SearchNotConfiguredError as exc:
        print(f"Sparse: NOT CONFIGURED -- {exc}", file=sys.stderr)
        sparse = None

    if dense is None and sparse is None:
        return 1

    if dense is not None:
        print("=== Dense (search_serve) ===")
        try:
            health = dense.health()
            print(f"  health: status={health.get('status')} n_docs={health.get('n_docs')}")
        except SearchUnavailableError as exc:
            print(f"  health: UNAVAILABLE -- {exc}")
        for kind, query in QUERIES:
            try:
                hits = dense.search(query, k=3)
                _print_hits(f"search[{kind}] {query!r}", hits)
            except SearchUnavailableError as exc:
                print(f"  search[{kind}] {query!r}: UNAVAILABLE -- {exc}")

    if sparse is not None:
        print("=== Sparse (BM25) ===")
        try:
            health = sparse.health()
            print(f"  health: status={health.get('status')}")
        except SearchUnavailableError as exc:
            print(f"  health: UNAVAILABLE -- {exc}")
        for kind, query in QUERIES:
            try:
                hits = sparse.search(query, k=3)
                _print_hits(f"search[{kind}] {query!r}", hits)
            except SearchUnavailableError as exc:
                print(f"  search[{kind}] {query!r}: UNAVAILABLE -- {exc}")

    if dense is not None and sparse is not None:
        print("=== Hybrid (RRF fusion) ===")
        retriever = HybridResearchRetriever(dense, sparse, HybridConfig(k_per_query=10, doc_limit=5))
        for kind, query in QUERIES:
            try:
                intent_kind = "hybrid"
                hits = retriever.retrieve([ResearchIntent(text=query, kind=intent_kind)])
                print(f"  fused[{kind}] {query!r}: {len(hits)} hit(s)")
                for h in hits[:5]:
                    backends = sorted({o.backend for o in h.occurrences})
                    snippet = (h.text or "")[:60].replace("\n", " ")
                    print(f"    chunk={h.chunk_id} parent={h.parent_docid} "
                          f"fused_rank={h.fused_rank} fused_score={h.fused_score:.5f} "
                          f"backends={backends} text={snippet!r}")
            except SearchUnavailableError as exc:
                print(f"  fused[{kind}] {query!r}: UNAVAILABLE -- {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
