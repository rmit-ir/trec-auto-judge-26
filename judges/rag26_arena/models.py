"""Arena vote vocabulary for the RAG26 pairwise judge.

``judges.ragree_choice.sparrow.Ballot`` only accepts votes in {A, B, TIE,
ABSTAIN}. TREC RAG 2026 / RAGDoll's Arena adds a fifth vote, BOTH_BAD ("tie,
both answers are unacceptable") -- plan Section 10.3 says to keep it as a
distinct diagnostic label while treating it as a TIE for the Davidson-Bradley-
Terry fit. ``BALLOT_VOTE`` does that mapping once, at the point a raw vote
becomes a ``Ballot``; the raw vote itself is kept alongside for the
BOTH_BAD_RATE measure.
"""

from __future__ import annotations

RAW_VOTES = frozenset({"A", "B", "TIE", "BOTH_BAD", "ABSTAIN"})

# Citation-support verdicts (plan Section 9.2); see support.py for the metrics.
SUPPORT_LEVELS = frozenset({"FULL", "PARTIAL", "NONE"})

BALLOT_VOTE: dict[str, str] = {
    "A": "A",
    "B": "B",
    "TIE": "TIE",
    "BOTH_BAD": "TIE",
    "ABSTAIN": "ABSTAIN",
}
