"""RAGDoll / TREC RAG 2026-style Arena pairwise preference prompt.

Public 2026 preference semantics -- "which answer would the user rather
receive" -- not a generic quality rubric (see
docs/trec-rag-2026-judge-implementation-plan.md Section 3.1). By design the
prompt excludes system names, run metadata, current ranking scores, and
citation-support labels (Section 10.2 there / Section 41 of the rubric plan):
only the request text and the two answers are shown.
"""

from __future__ import annotations

from autojudge_base import Request

_SYSTEM = "You are a careful RAG-answer preference judge. Return only the requested JSON."

_CRITERIA = (
    "intent fulfilment, correctness/plausibility, completeness, specificity, "
    "synthesis of evidence, calibration of uncertainty, explanation quality/"
    "usefulness, and organisation/noise control"
)


def narrative(topic: Request) -> str:
    """The request text shown to the Arena judge -- never candidate-derived."""
    return topic.problem_statement or topic.title


def arena_messages(topic: Request, answer_a: str, answer_b: str, *, max_answer_chars: int) -> list[dict[str, str]]:
    prompt = (
        "Which answer would the user rather receive for this request? Judge on "
        f"{_CRITERIA}. Do not reward verbosity, polished prose, or formatting by "
        "themselves. Vote TIE when both are comparably good, BOTH_BAD when both "
        "are unacceptable, and ABSTAIN when you cannot judge from the material "
        "given. Reply with ONLY JSON: {\"vote\": \"A|B|TIE|BOTH_BAD|ABSTAIN\"}.\n\n"
        f"Request:\n{narrative(topic)}\n\n"
        f"Answer A:\n{answer_a[:max_answer_chars]}\n\n"
        f"Answer B:\n{answer_b[:max_answer_chars]}"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": prompt},
    ]
