"""RAGDoll-compatible citation-support metrics (pure data layer, no LLM/IO).

docs/trec-rag-2026-judge-implementation-plan.md Section 9. Judges each
(sentence, cited-document) pair independently: does the cited document
support that specific sentence, not the topic generally. Citations within a
sentence are already ordered strongest-to-weakest by
``Report.get_sentences_with_citations()`` (rank 0 = "first citation").

An answer sentence with no citation at all is excluded from precision but
still counts (as unsupported) in recall's fixed denominator -- the RAGDoll
asymmetry from plan Section 3.3. A citation that exists but whose document
text this Report cannot resolve, or whose judgment call never came back, is
treated the same way for recall (still 0, still counted) but is excluded
from precision's denominator entirely: precision is over "judged sentences
with citations" only (plan Section 9.4's exact phrase), never penalized for
an availability gap. ``RAG26_SUPPORT_EVALUATED`` (see judge.py) is how a
caller tells "no citation" apart from "citation we could not judge".

HARD_PREC/HARD_REC use the first-citation convention -- the public RAGDoll
metric list (plan Section 3.3) lists exactly one hard precision/recall pair
alongside separate first/all weighted pairs, so hard mirrors first rather
than introducing an unlisted hard-all variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from autojudge_base import Report

WEIGHTED_SUPPORT: dict[str, float] = {"FULL": 1.0, "PARTIAL": 0.5, "NONE": 0.0}
HARD_SUPPORT: dict[str, float] = {"FULL": 1.0, "PARTIAL": 0.0, "NONE": 0.0}


@dataclass(frozen=True)
class CitationPair:
    """One (sentence, cited-document) slot needing a support judgment.

    ``doc_text`` is None when the citation exists but this Report cannot
    resolve its text -- unjudgeable, never silently treated as NONE.
    """

    sentence_index: int
    citation_rank: int  # 0 = first/strongest citation for this sentence
    doc_id: str
    doc_text: Optional[str]


@dataclass(frozen=True)
class ExtractedReport:
    total_sentences: int
    sentence_texts: tuple[str, ...]
    pairs: tuple[CitationPair, ...]


def extract_citation_pairs(report: Report) -> ExtractedReport:
    """Never calls the offline ClimbMix research index (docs/rag26-tira-submission-plan.md
    Phase 3: the submitted TIRA judge must not depend on either local service) --
    only document text already resolved onto this Report is usable."""
    documents = report.documents or {}
    sentences = report.get_sentences_with_citations()
    pairs: list[CitationPair] = []
    for sentence_index, sentence in enumerate(sentences):
        for citation_rank, doc_id in enumerate(sentence.citations or []):
            document = documents.get(doc_id)
            pairs.append(CitationPair(
                sentence_index=sentence_index, citation_rank=citation_rank,
                doc_id=doc_id, doc_text=document.get_text() if document else None,
            ))
    return ExtractedReport(
        total_sentences=len(sentences),
        sentence_texts=tuple(s.text for s in sentences),
        pairs=tuple(pairs),
    )


@dataclass(frozen=True)
class SupportMetrics:
    prec_first: float
    rec_first: float
    prec_all: float
    rec_all: float
    hard_prec: float
    hard_rec: float
    evaluated: float


def compute_support_metrics(
    extracted: ExtractedReport,
    verdicts: Mapping[tuple[int, int], str],
) -> SupportMetrics:
    """``verdicts`` maps (sentence_index, citation_rank) -> "FULL"/"PARTIAL"/"NONE"
    for successfully judged pairs only; a pair absent from ``verdicts`` (text
    unresolved, or the judgment call never came back after retries) is
    unjudged, not NONE."""
    by_sentence: dict[int, list[CitationPair]] = {}
    for pair in extracted.pairs:
        by_sentence.setdefault(pair.sentence_index, []).append(pair)

    prec_first_sum = prec_first_n = rec_first_sum = 0.0
    hard_prec_sum = hard_prec_n = hard_rec_sum = 0.0
    prec_all_sum = prec_all_n = rec_all_sum = 0.0

    for sentence_pairs in by_sentence.values():
        first = next((p for p in sentence_pairs if p.citation_rank == 0), None)
        first_verdict = verdicts.get((first.sentence_index, first.citation_rank)) if first else None
        if first_verdict is not None:
            prec_first_sum += WEIGHTED_SUPPORT[first_verdict]
            prec_first_n += 1
            rec_first_sum += WEIGHTED_SUPPORT[first_verdict]
            hard_prec_sum += HARD_SUPPORT[first_verdict]
            hard_prec_n += 1
            hard_rec_sum += HARD_SUPPORT[first_verdict]

        judged_all = [
            WEIGHTED_SUPPORT[verdicts[(p.sentence_index, p.citation_rank)]]
            for p in sentence_pairs
            if (p.sentence_index, p.citation_rank) in verdicts
        ]
        if judged_all:
            mean_support = sum(judged_all) / len(judged_all)
            prec_all_sum += mean_support
            prec_all_n += 1
            rec_all_sum += mean_support

    total_sentences = extracted.total_sentences
    judged_pairs = sum(1 for pair in extracted.pairs if (pair.sentence_index, pair.citation_rank) in verdicts)

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    return SupportMetrics(
        prec_first=ratio(prec_first_sum, prec_first_n),
        rec_first=ratio(rec_first_sum, total_sentences),
        prec_all=ratio(prec_all_sum, prec_all_n),
        rec_all=ratio(rec_all_sum, total_sentences),
        hard_prec=ratio(hard_prec_sum, hard_prec_n),
        hard_rec=ratio(hard_rec_sum, total_sentences),
        evaluated=ratio(judged_pairs, len(extracted.pairs)) if extracted.pairs else 1.0,
    )


UNEVALUATED_SUPPORT_METRICS = SupportMetrics(
    prec_first=0.0, rec_first=0.0, prec_all=0.0, rec_all=0.0,
    hard_prec=0.0, hard_rec=0.0, evaluated=0.0,
)


def chunk_pairs(pairs: Sequence[CitationPair], chunk_size: int) -> list[list[CitationPair]]:
    """Only pairs with resolvable text can be sent to the LLM at all."""
    judgeable = [pair for pair in pairs if pair.doc_text is not None]
    return [judgeable[i:i + chunk_size] for i in range(0, len(judgeable), chunk_size)]
