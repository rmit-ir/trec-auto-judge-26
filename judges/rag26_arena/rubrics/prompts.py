"""Versioned prompt builders for rubric construction (execution-plan Section 8).

Every builder exposes a stable version constant; changing a prompt's text
requires bumping its version so cache keys and the freeze manifest correctly
invalidate stale entries. Temperature 0 (the closest deterministic setting
Bedrock/Azure support) for every stage unless a specific development
experiment demonstrates a benefit from diversity -- see llm.py's callers.
"""

from __future__ import annotations

CHARACTERIZE_PROMPT_VERSION = "v1"

_CHARACTERIZE_SYSTEM = "You are a careful research-task analyst. Return only the requested JSON."

_COMPLEXITY_DEFINITIONS = (
    "conceptual_breadth: Simple (one topic/domain), Moderate (2-5 weakly coupled "
    "subtopics/data sources), High (more than five sources/perspectives or clearly "
    "disjoint domains requiring cross-domain synthesis).\n"
    "logical_nesting: Shallow (direct retrieval or one inference operation), "
    "Intermediate (2-3 dependent reasoning/subquestion stages), Deep (4+ dependent "
    "stages or hierarchical analysis -> synthesis -> evaluation/planning).\n"
    "exploration: Low (goals and evaluation requirements are explicit), Medium (one "
    "or two important factors require prioritization/inference), High (substantially "
    "underspecified/open-ended, requires inferring several key evaluation dimensions)."
)

_RETRIEVAL_MODE_DEFINITIONS = (
    "dense: conceptual/paraphrase-oriented queries.\n"
    "sparse: names, dates, legislation, rare terms, technical terminology, "
    "identifiers, exact lexical anchors.\n"
    "hybrid: important or ambiguous facets that benefit from both."
)


def characterize_and_plan_messages(narrative: str) -> list[dict[str, str]]:
    """Stage A (execution-plan Section 9): task characterization and
    retrieval-intent planning in one call. Response-independent -- only the
    public request narrative is ever passed in."""
    prompt = (
        "Analyze this research request and produce a structured characterization "
        "plus a retrieval-intent plan for evidence-grounded rubric construction. "
        "There are no candidate answers available to you and none should be "
        "assumed -- work only from the request text below.\n\n"
        f"Complexity definitions:\n{_COMPLEXITY_DEFINITIONS}\n\n"
        f"Retrieval-mode definitions:\n{_RETRIEVAL_MODE_DEFINITIONS}\n\n"
        "Generate 6-10 research_intents that together cover: each explicit "
        "subquestion; names/entities/rare terminology; quantitative/date "
        "evidence; relevant context; counterarguments; stakeholder effects; "
        "uncertainty/limitations; causal confounds; and evidence that could "
        "falsify a simplistic conclusion. Route each intent to whichever "
        "retrieval_mode best fits the definitions above.\n\n"
        "Reply with ONLY this JSON object, no other text:\n"
        '{"domain": "...", "conceptual_breadth": "Simple|Moderate|High", '
        '"logical_nesting": "Shallow|Intermediate|Deep", '
        '"exploration": "Low|Medium|High", '
        '"explicit_subquestions": ["..."], "named_constraints": ["..."], '
        '"synthesis_goal": "...", '
        '"research_intents": [{"query": "...", "retrieval_mode": "dense|sparse|hybrid", '
        '"purpose": "..."}]}\n\n'
        f"Request:\n{narrative}"
    )
    return [
        {"role": "system", "content": _CHARACTERIZE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
