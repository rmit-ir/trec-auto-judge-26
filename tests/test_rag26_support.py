"""Unit tests for judges.rag26_arena.support: pure metrics, no LLM/IO.

Scenarios mirror docs/trec-rag-2026-judge-implementation-plan.md Section 15.2:
full/partial/no-support first citation, uncited sentence, multiple citations
with mixed support, and failed/missing judgment.
"""

from autojudge_base import Request
from autojudge_base.report import Document, Report, ReportMetaData
from autojudge_base.report_sentences import NeuclirReportSentence

from judges.rag26_arena.support import (
    UNEVALUATED_SUPPORT_METRICS,
    chunk_pairs,
    compute_support_metrics,
    extract_citation_pairs,
)


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, text=text)


def _report(sentences: list[NeuclirReportSentence], documents: dict[str, Document]) -> Report:
    return Report(
        metadata=ReportMetaData(team_id="run", run_id="run", topic_id="t1"),
        responses=sentences, documents=documents,
    )


def test_full_support_first_citation_scores_one():
    report = _report(
        [NeuclirReportSentence(text="Sky is blue.", citations=["d1"])],
        {"d1": _doc("d1", "The sky appears blue due to Rayleigh scattering.")},
    )
    extracted = extract_citation_pairs(report)
    assert extracted.total_sentences == 1
    assert len(extracted.pairs) == 1
    metrics = compute_support_metrics(extracted, {(0, 0): "FULL"})
    assert metrics.prec_first == metrics.rec_first == 1.0
    assert metrics.hard_prec == metrics.hard_rec == 1.0
    assert metrics.evaluated == 1.0


def test_partial_support_scores_half():
    report = _report(
        [NeuclirReportSentence(text="X causes Y.", citations=["d1"])],
        {"d1": _doc("d1", "Some unrelated background on X.")},
    )
    extracted = extract_citation_pairs(report)
    metrics = compute_support_metrics(extracted, {(0, 0): "PARTIAL"})
    assert metrics.prec_first == metrics.rec_first == 0.5
    # Hard support treats PARTIAL as 0, not half credit.
    assert metrics.hard_prec == metrics.hard_rec == 0.0


def test_no_support_scores_zero_but_is_evaluated():
    report = _report(
        [NeuclirReportSentence(text="X causes Y.", citations=["d1"])],
        {"d1": _doc("d1", "Completely unrelated document.")},
    )
    extracted = extract_citation_pairs(report)
    metrics = compute_support_metrics(extracted, {(0, 0): "NONE"})
    assert metrics.prec_first == metrics.rec_first == 0.0
    assert metrics.evaluated == 1.0  # a real NONE verdict, not an availability gap


def test_uncited_sentence_lowers_recall_but_not_precision():
    report = _report(
        [
            NeuclirReportSentence(text="Cited claim.", citations=["d1"]),
            NeuclirReportSentence(text="Uncited claim.", citations=None),
        ],
        {"d1": _doc("d1", "Supports the cited claim fully.")},
    )
    extracted = extract_citation_pairs(report)
    assert extracted.total_sentences == 2
    metrics = compute_support_metrics(extracted, {(0, 0): "FULL"})
    assert metrics.prec_first == 1.0  # only the one judged-with-citation sentence counts
    assert metrics.rec_first == 0.5  # 1.0 / 2 total sentences


def test_multiple_citations_mixed_support_uses_first_and_mean():
    report = _report(
        [NeuclirReportSentence(text="Complex claim.", citations=["strong", "weak"])],
        {
            "strong": _doc("strong", "Fully supports the claim."),
            "weak": _doc("weak", "Barely related."),
        },
    )
    extracted = extract_citation_pairs(report)
    assert [p.citation_rank for p in extracted.pairs] == [0, 1]
    metrics = compute_support_metrics(extracted, {(0, 0): "FULL", (0, 1): "NONE"})
    assert metrics.prec_first == 1.0  # first citation only
    assert metrics.prec_all == 0.5    # mean(1.0, 0.0)


def test_unresolved_document_text_is_unjudgeable_not_none():
    report = _report(
        [NeuclirReportSentence(text="Claim citing a missing doc.", citations=["missing"])],
        {},  # "missing" never resolves to text
    )
    extracted = extract_citation_pairs(report)
    assert extracted.pairs[0].doc_text is None
    assert chunk_pairs(extracted.pairs, chunk_size=10) == []  # nothing judgeable to send
    metrics = compute_support_metrics(extracted, {})  # no judgment came back
    assert metrics.prec_first == 0.0  # excluded from the denominator, not penalized as NONE
    assert metrics.rec_first == 0.0   # still counts against recall's fixed denominator
    assert metrics.evaluated == 0.0   # distinguishes "unjudged" from a real 0.0 score


def test_missing_judgment_after_failed_llm_call_is_unavailable_not_none():
    report = _report(
        [NeuclirReportSentence(text="Claim.", citations=["d1"])],
        {"d1": _doc("d1", "Some text.")},
    )
    extracted = extract_citation_pairs(report)
    metrics = compute_support_metrics(extracted, {})  # judgment call never came back
    assert metrics.evaluated == 0.0
    assert metrics.prec_first == 0.0
    assert metrics.rec_first == 0.0


def test_no_citations_at_all_is_vacuously_evaluated():
    report = _report([NeuclirReportSentence(text="No citations here.", citations=None)], {})
    extracted = extract_citation_pairs(report)
    assert extracted.pairs == ()
    metrics = compute_support_metrics(extracted, {})
    assert metrics.evaluated == 1.0  # nothing needed judging
    assert metrics.prec_first == 0.0
    assert metrics.rec_first == 0.0


def test_unevaluated_placeholder_signals_via_evaluated_flag():
    assert UNEVALUATED_SUPPORT_METRICS.evaluated == 0.0
    assert UNEVALUATED_SUPPORT_METRICS.prec_first == 0.0


def test_chunk_pairs_splits_only_judgeable_pairs():
    report = _report(
        [NeuclirReportSentence(text="s", citations=["d1", "missing", "d2"])],
        {"d1": _doc("d1", "text1"), "d2": _doc("d2", "text2")},
    )
    extracted = extract_citation_pairs(report)
    chunks = chunk_pairs(extracted.pairs, chunk_size=1)
    assert len(chunks) == 2  # only d1 and d2 are judgeable; "missing" is dropped
    assert all(len(chunk) == 1 for chunk in chunks)
