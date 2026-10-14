"""Versioned prompt builders for rubric construction (execution-plan Section 8).

Every builder exposes a stable version constant; changing a prompt's text
requires bumping its version so cache keys and the freeze manifest correctly
invalidate stale entries. Temperature 0 (the closest deterministic setting
Bedrock/Azure support) for every stage unless a specific development
experiment demonstrates a benefit from diversity -- see llm.py's callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .generation_models import EvidenceNote, RubricCandidate, TopicCharacterization

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


# --- D1-D5: axis-specific candidate generation (execution-plan Section 12 /
# rag26-rubric-next-stage-handoff.md) -------------------------------------

EXPLICIT_PROMPT_VERSION = "v1"
IMPLICIT_PROMPT_VERSION = "v2"
SYNTHESIS_PROMPT_VERSION = "v2"
NEGATIVE_PROMPT_VERSION = "v1"
LOW_VOLUME_PROMPT_VERSION = "v1"

_CANDIDATE_SYSTEM = "You are a careful rubric-criterion writer. Return only the requested JSON."

_EVIDENCE_SCOPE_RULE = (
    "CRITICAL SCOPE RULE: retrieved evidence is supporting material, not an answer outline. "
    "Never make a named study, technique, organization, law, case, event, statistic, example, "
    "or source-specific detail an unconditional positive requirement merely because it appears "
    "in the evidence. A positive criterion may require such a detail only when the user's "
    "request explicitly names it or when that specific detail is genuinely necessary to satisfy "
    "the user's stated information need. Otherwise generalize the criterion to the underlying "
    "capability or concept; if correctness matters only when an answer chooses to discuss that "
    "detail, make the expectation conditional; if neither applies, omit it. Evidence should help "
    "determine what a good answer gets right, not dictate which examples a good answer must use."
)

_CANDIDATE_ITEM_SHAPE = (
    '{"criterion": "The response ...", "polarity": "positive|negative", '
    '"rationale": "...", "narrative_spans": ["..."], "evidence_ids": ["..."], '
    '"confidence": 0.0-1.0}'
)


def _compact_candidates(candidates: Sequence["RubricCandidate"]) -> str:
    return "\n".join(f"- [{c.axis}] {c.criterion}" for c in candidates) or "(none yet)"


def _compact_evidence(evidence_notes: Sequence["EvidenceNote"]) -> str:
    return "\n".join(
        f"[{note.evidence_id}] ({note.subtopic}, {note.evidence_type}) {note.claim}"
        for note in evidence_notes
    ) or "(none)"


def explicit_candidates_messages(
    narrative: str, characterization: "TopicCharacterization", *, min_count: int, max_count: int,
) -> list[dict[str, str]]:
    """D1 (handoff Section "D1 -- Explicit criteria generator"). No evidence
    notes as input by design -- explicit criteria come from the request
    itself, never from retrieved corpus content."""
    prompt = (
        f"Generate {min_count}-{max_count} EXPLICIT rubric criteria for this request: "
        "requirements the user directly and literally asked for. Every explicit ask "
        "should be covered by exactly one candidate; do not force an exact count.\n\n"
        "Each candidate must:\n"
        '- correspond to one explicit user requirement;\n'
        '- start with "The response ..." or an equally observable formulation;\n'
        "- describe one primary, judgeable-from-the-answer-alone outcome;\n"
        "- avoid hidden reasoning requirements;\n"
        "- avoid duplicating an ask another candidate already covers;\n"
        "- preserve named entities, time periods, comparisons, stakeholders, and any "
        "requested recommendation/decision exactly;\n"
        "- use positive polarity only -- an explicit prohibition becomes a negative "
        "criterion in a later stage, not here.\n\n"
        "For narrative_spans, copy the EXACT substring of the request text or one of "
        "the listed explicit subquestions this criterion corresponds to -- verbatim, "
        "never paraphrased. Leave evidence_ids as [] (no evidence notes are available "
        "at this stage).\n\n"
        f"Request:\n{narrative}\n\n"
        f"Explicit subquestions: {list(characterization.explicit_subquestions)}\n"
        f"Named constraints: {list(characterization.named_constraints)}\n"
        f"Synthesis goal: {characterization.synthesis_goal}\n\n"
        f"Reply with ONLY a JSON array, each item exactly shaped: {_CANDIDATE_ITEM_SHAPE}"
    )
    return [{"role": "system", "content": _CANDIDATE_SYSTEM}, {"role": "user", "content": prompt}]


def implicit_candidates_messages(
    narrative: str, characterization: "TopicCharacterization", evidence_notes: Sequence["EvidenceNote"],
    explicit_candidates: Sequence["RubricCandidate"], *, min_count: int, max_count: int,
) -> list[dict[str, str]]:
    """D2 -- the highest-value, highest-risk generator."""
    prompt = (
        f"Generate {min_count}-{max_count} IMPLICIT rubric criteria. Anchor on: what "
        "would a knowledgeable expert expect a high-quality response to address even "
        "though the user did not literally ask for it? Prioritise: prerequisite "
        "context; important caveats; confounders; alternative explanations; "
        "stakeholder effects; evidence limitations; uncertainty; causal-vs-"
        "correlational distinctions; time sensitivity; missing decision-relevant "
        "quantities; meaningful counterarguments; second-order effects.\n\n"
        "Every factual/evidence-dependent candidate must cite at least one evidence_id "
        "from the list below by its bracketed id. A purely reasoning/usability "
        "candidate (asserting no fact) may have no evidence_ids only when its "
        "rationale clearly derives from the request structure itself, not the "
        "retrieved corpus. If evidence conflicts, write a criterion requiring "
        "calibrated treatment of the disagreement -- never assert one side as "
        "settled. Do not add a criterion just because a fact was interesting; only "
        "decision-relevant expectations belong here. Give each candidate a one-line "
        "rationale explaining why it is a reasonable expectation rather than "
        "arbitrary scope expansion. Use positive polarity only, and do not duplicate "
        "an explicit criterion already listed below.\n\n"
        f"{_EVIDENCE_SCOPE_RULE}\n\n"
        f"Request:\n{narrative}\n\n"
        f"Synthesis goal: {characterization.synthesis_goal}\n\n"
        f"Existing explicit criteria (do not duplicate):\n{_compact_candidates(explicit_candidates)}\n\n"
        f"Evidence notes (cite only by bracketed id):\n{_compact_evidence(evidence_notes)}\n\n"
        f"Reply with ONLY a JSON array, each item exactly shaped: {_CANDIDATE_ITEM_SHAPE}"
    )
    return [{"role": "system", "content": _CANDIDATE_SYSTEM}, {"role": "user", "content": prompt}]


def synthesis_candidates_messages(
    narrative: str, characterization: "TopicCharacterization", evidence_notes: Sequence["EvidenceNote"],
    explicit_candidates: Sequence["RubricCandidate"], implicit_candidates: Sequence["RubricCandidate"],
    *, min_count: int, max_count: int,
) -> list[dict[str, str]]:
    """D3. Rejects pseudo-synthesis (fact lists) by requiring a relational verb."""
    existing = _compact_candidates(list(explicit_candidates) + list(implicit_candidates))
    prompt = (
        f"Generate {min_count}-{max_count} SYNTHESIS rubric criteria. Each one must "
        "require a relationship that isolated fact-copying cannot satisfy -- prefer "
        "verbs such as compare, reconcile, connect, distinguish, weigh, explain, "
        "assess, contextualize. Reject pseudo-synthesis such as \"the response "
        "mentions A, B, and C\" unless the actual requirement is to explain how "
        "A/B/C interact. Where evidence conflicts, reward explaining the "
        "disagreement/evidence strength rather than forcing one conclusion. Retain "
        "the relevant evidence_ids/subtopics for each evidence-grounded candidate. "
        "Use positive polarity only.\n\n"
        f"{_EVIDENCE_SCOPE_RULE}\n\n"
        f"Request:\n{narrative}\n\n"
        f"Synthesis goal: {characterization.synthesis_goal}\n\n"
        f"Existing criteria (do not duplicate):\n{existing}\n\n"
        f"Evidence notes (cite only by bracketed id):\n{_compact_evidence(evidence_notes)}\n\n"
        f"Reply with ONLY a JSON array, each item exactly shaped: {_CANDIDATE_ITEM_SHAPE}"
    )
    return [{"role": "system", "content": _CANDIDATE_SYSTEM}, {"role": "user", "content": prompt}]


def negative_candidates_messages(
    narrative: str, characterization: "TopicCharacterization", evidence_notes: Sequence["EvidenceNote"],
    positive_candidates: Sequence["RubricCandidate"], *, min_count: int, max_count: int, allowed_axes: Sequence[str],
) -> list[dict[str, str]]:
    """D4. Negative criteria can belong to any of the six canonical axes (not
    a fixed one like D1-D3), so the model must supply an axis field here --
    validated against `allowed_axes` by the parser."""
    prompt = (
        f"Generate {min_count}-{max_count} NEGATIVE (failure-mode) rubric criteria. "
        "Do not generate negatives simply to hit a count. Prioritise materially "
        "damaging failures: factual reversal; causal overclaim; incorrect number or "
        "denominator; fabricated certainty; stale/out-of-scope information; "
        "misleading omission; one-sided treatment where balance is required; "
        "source/evidence misattribution; contradiction with strong evidence; scope "
        "violation; material audience/format violation.\n\n"
        "Do not create a negative criterion that is merely the logical negation of "
        "an existing positive criterion, unless committing the error would be "
        "materially worse than simply omitting it -- e.g. \"discusses uncertainty\" "
        "vs. its bad mirror \"fails to discuss uncertainty\" is not allowed; "
        "\"presents a disputed conclusion as established fact\" is.\n\n"
        f"Each candidate needs an axis field, one of exactly: {list(allowed_axes)}. "
        "Use negative polarity for every candidate.\n\n"
        f"Request:\n{narrative}\n\n"
        f"Existing positive criteria (for mirror-penalty avoidance):\n{_compact_candidates(positive_candidates)}\n\n"
        f"Evidence notes (cite only by bracketed id):\n{_compact_evidence(evidence_notes)}\n\n"
        "Reply with ONLY a JSON array, each item exactly shaped: "
        '{"criterion": "...", "axis": "...", "polarity": "negative", "rationale": "...", '
        '"narrative_spans": ["..."], "evidence_ids": ["..."], "confidence": 0.0-1.0}'
    )
    return [{"role": "system", "content": _CANDIDATE_SYSTEM}, {"role": "user", "content": prompt}]


def low_volume_candidates_messages(
    narrative: str, characterization: "TopicCharacterization",
) -> list[dict[str, str]]:
    """D5: References & Citation Quality, Communication Quality, and
    Instruction Following combined into one call. Each axis's items live
    under their own fixed JSON key, so axis is never a model-invented field
    here (unlike D4) -- deterministic code assigns it per key."""
    prompt = (
        "Generate a SMALL number of topic-specific rubric criteria for exactly "
        "three axes, kept light -- these are not the main content of the rubric.\n\n"
        "references (1-2 candidates, References & Citation Quality): important "
        "empirical/numerical claims should be backed by specific evidence; "
        "conflicting claims should be attributed rather than blended; sources "
        "should not be misrepresented. Do NOT duplicate sentence-level citation-"
        "support scoring (that is handled elsewhere) -- keep this rubric-level and "
        "topic-specific.\n\n"
        "communication (1-3 candidates, Communication Quality): topic-specific "
        "usability, e.g. trade-offs organised for easy comparison, specialist "
        "terminology needed for the decision explained, the requested bottom line "
        "clear and not buried. Not a universal essay-structure rule.\n\n"
        "instruction_following (1-2 candidates, Instruction Following): only actual "
        "scope/audience/format/decision constraints from the request -- never move a "
        "factual explicit requirement into this axis.\n\n"
        "Leave evidence_ids as [] for every candidate -- no evidence notes are "
        "supplied for this stage.\n\n"
        f"Request:\n{narrative}\n\n"
        f"Named constraints: {list(characterization.named_constraints)}\n\n"
        "Reply with ONLY this JSON object, no other text: "
        '{"references": [ITEM, ...], "communication": [ITEM, ...], '
        f'"instruction_following": [ITEM, ...]}} where each ITEM is exactly shaped: {_CANDIDATE_ITEM_SHAPE}'
    )
    return [{"role": "system", "content": _CANDIDATE_SYSTEM}, {"role": "user", "content": prompt}]
