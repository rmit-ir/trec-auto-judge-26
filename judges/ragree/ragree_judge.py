#!/usr/bin/env python3
"""
RAGreeEvidenceJudge: LLM-based Evidence Judge lane from docs/ragree-judge-design.md.

Scoped-down v1 of the "Evidence Judge" (design doc Section 3) only. Claims are
SENTENCE-level (each response sentence plus its already-attached citations),
not sub-sentence atomic claims -- true atomic decomposition would orphan the
citation attachments the format already gives us for free (re-attributing
citations to sub-sentence claims is flagged as an open problem in the design
doc's own Section 3.5, point 2).

Verification only checks claims against `Report.documents` (the retrieved/
cited context inlined with the run) -- there is no "authoritative corpus"
access in this framework, so the design doc's second verification pass
(Section 3.2, "Authoritative-corpus verification") is not implemented.

Not implemented at all (see docs/ragree-judge-design.md "Implementation
status" section for the full list): retrieval precision/recall (needs
nugget-vs-every-retrieved-document verification and `Report.documents` is
frequently absent), the Preference Judge, the Trajectory Judge (no
agent-trace field on Report/Request), and everything past Section 7 of the
design doc (training pipeline, fine-tuning, meta-evaluation harness).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from autojudge_base import (
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LlmConfigProtocol,
    MeasureSpec,
    NuggetBanks,
    NuggetBanksProtocol,
    Qrels,
    Report,
    Request,
)
from autojudge_base.nugget_data import NuggetBank, NuggetQuestion
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm


RAGREE_SPEC = LeaderboardSpec(measures=(
    MeasureSpec(
        "CONTEXT_FAITHFULNESS",
        description=(
            "Weighted claim-support score over the response's sentence-claims. A "
            "sentence with no citation at all counts as unsupported (0.0) -- a real "
            "grounding gap. A sentence whose citation could not be resolved to a "
            "document in Report.documents, or whose verification LLM call failed or "
            "returned unparseable output, is EXCLUDED from the denominator entirely "
            "(see EVIDENCE_COVERAGE_RATE for how much evidence was actually checkable). "
            "Checks grounding in the retrieved/cited context only, not an authoritative "
            "corpus. 0.0 if no sentence-claim was evaluable."
        ),
    ),
    MeasureSpec(
        "ESTIMATED_COVERAGE",
        description=(
            "Fraction of query-only, LLM-generated nugget questions (not human-vetted "
            "gold nuggets) that the response appears to address. An estimate of answer "
            "recall, not a measurement against a gold denominator -- see design doc "
            "Section 1, 'A crucial point about recall'."
        ),
    ),
    MeasureSpec(
        "CLAIM_F1",
        description=(
            "Harmonic mean of CONTEXT_FAITHFULNESS and ESTIMATED_COVERAGE. Inherits "
            "both measures' scoping caveats -- this is not RAGChecker-style claim F1 "
            "against gold nuggets."
        ),
    ),
    MeasureSpec(
        "CITATION_PRECISION",
        description=(
            "Fraction of individual citation links (one per cited document per "
            "sentence, not per sentence) verified as supporting the sentence they are "
            "attached to. Only counts links whose cited document resolved in "
            "Report.documents AND whose verification LLM call returned a parseable "
            "per-link verdict -- a failed or malformed response excludes that link "
            "rather than guessing its support."
        ),
    ),
    MeasureSpec(
        "CITATION_COVERAGE",
        description=(
            "Fraction of substantive sentences (>= substantive_min_words words, a "
            "length heuristic, not an LLM-judged importance label) carrying at least "
            "one citation. Approximates ALCE-style citation recall without a "
            "per-claim importance judgment."
        ),
    ),
    MeasureSpec(
        "CRITICAL_ERROR_RATE",
        description=(
            "Fraction of evaluable sentence-claims (same denominator as "
            "CONTEXT_FAITHFULNESS) verified CONTRADICTED by their cited context. Does "
            "not distinguish minor/major/critical severity (design doc Section 3.3) -- "
            "every contradiction counts equally."
        ),
    ),
    MeasureSpec(
        "EVIDENCE_COVERAGE_RATE",
        description=(
            "Fraction of cited sentences whose citations actually resolved to a "
            "document in Report.documents (vs. UNAVAILABLE because the run omitted "
            "Report.documents or the cited doc_id was not in it). Low values mean "
            "CONTEXT_FAITHFULNESS/CITATION_PRECISION above are based on little "
            "evidence, not that the run is unfaithful."
        ),
    ),
))


_VERDICTS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE")
_SUPPORT_SCORE = {"SUPPORTED": 1.0, "PARTIALLY_SUPPORTED": 0.5, "CONTRADICTED": 0.0, "NOT_ENOUGH_EVIDENCE": 0.0}


def _backend(llm_config: LlmConfigProtocol) -> OpenAIMinimaLlm:
    full_config = MinimaLlmConfig.from_dict(llm_config.raw) if getattr(llm_config, "raw", None) else MinimaLlmConfig.from_env()
    return OpenAIMinimaLlm(full_config)


def _extract_json_value(text: str, opener: str, wanted_type: type) -> Any:
    """Find the first balanced JSON value of `wanted_type` starting at each `opener`
    ('{' or '[') in `text`, left to right. A plain greedy regex (`\\{.*\\}`) breaks on
    reasoning-model preambles or trailing prose that contain stray braces -- this scans
    from every candidate start and lets json.JSONDecoder find the real matching end.
    """
    decoder = json.JSONDecoder()
    idx = text.find(opener)
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(text, idx)
            if isinstance(obj, wanted_type):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find(opener, idx + 1)
    return None


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM reply. None on any failure."""
    return _extract_json_value(text, "{", dict)


def _extract_json_array(text: str) -> Optional[list]:
    """Pull the first JSON array out of an LLM reply. None on any failure."""
    return _extract_json_value(text, "[", list)


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2 * a * b / (a + b)


# =============================================================================
# RAGreeNuggetCreator - NuggetCreatorProtocol
# =============================================================================

class RAGreeNuggetCreator:
    """Query-only nugget generation: one LLM call per topic, from the topic title/
    background alone. Cheap, but the resulting nuggets are LLM guesses at what a
    complete answer should cover, not human-vetted gold nuggets -- RAGreeEvidenceJudge
    reports coverage against them as ESTIMATED_COVERAGE, never as gold recall.
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        questions_per_topic: int = 4,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[NuggetBanksProtocol]:
        topics = list(rag_topics)
        requests: List[MinimaLlmRequest] = []
        for topic in topics:
            prompt = (
                f"A complete, correct answer to this question should cover several "
                f"distinct pieces of information.\n\nQuestion: {topic.title}\n"
                f"{'Background: ' + topic.background if topic.background else ''}\n\n"
                f"List exactly {questions_per_topic} short, distinct sub-questions "
                f"that a complete answer must address. Reply with ONLY a JSON array "
                f"of strings, nothing else."
            )
            requests.append(MinimaLlmRequest(
                request_id=f"nugget-{topic.request_id}",
                messages=[
                    {"role": "system", "content": "You decompose questions into required sub-questions. Reply with only a JSON array of strings."},
                    {"role": "user", "content": prompt},
                ],
            ))

        backend = _backend(llm_config)
        results = asyncio.run(backend.run_batched(requests))

        banks: List[NuggetBank] = []
        for topic, result in zip(topics, results):
            questions = self._parse_questions(result, questions_per_topic)
            if not questions:
                # A failed LLM call must not create an empty bank: the workflow runner
                # rejects it before judge() can score an otherwise valid empty report.
                # This deterministic topic-only fallback keeps the pipeline complete;
                # its resulting ESTIMATED_COVERAGE is intentionally conservative.
                questions = [topic.title or topic.request_id]
            bank = NuggetBank(query_id=topic.request_id, title_query=topic.title or topic.request_id)
            bank.add_nuggets([
                NuggetQuestion.from_lazy(query_id=topic.request_id, question=q)
                for q in questions
            ])
            banks.append(bank)

        print(f"RAGreeNuggetCreator: generated nuggets for {len(banks)} topics")
        return NuggetBanks.from_banks_list(banks)

    @staticmethod
    def _parse_questions(result: Any, expected: int) -> List[str]:
        if not isinstance(result, MinimaLlmResponse):
            print(f"[RAGreeNuggetCreator] LLM error, topic skipped: {result}")
            return []
        text = result.text.strip()
        parsed = _extract_json_array(text)
        if parsed is not None:
            return [str(q).strip() for q in parsed if str(q).strip()][:expected]
        # Fallback: one question per non-empty line.
        lines = [
            question
            for question in (ln.strip(" -*0123456789.") for ln in text.splitlines())
            if question
        ]
        return lines[:expected]


# =============================================================================
# RAGreeEvidenceJudge - LeaderboardJudgeProtocol
# =============================================================================

class RAGreeEvidenceJudge:
    """See module docstring for the (large) list of what this does NOT implement."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        substantive_min_words: int = 6,
        max_doc_chars: int = 1500,
        on_missing_evals: str = "fix_aggregate",
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        topics_by_id: Dict[str, Request] = {t.request_id: t for t in rag_topics}
        expected_topic_ids: List[str] = list(topics_by_id.keys())

        responses: List[Report] = sorted(
            rag_responses, key=lambda r: (r.metadata.run_id, r.metadata.topic_id)
        )

        # Per-response bookkeeping, filled in as verification/coverage results land.
        claim_score_sum: Dict[Tuple[str, str], float] = {}
        claim_count: Dict[Tuple[str, str], int] = {}
        contradicted_count: Dict[Tuple[str, str], int] = {}
        citation_link_supported: Dict[Tuple[str, str], int] = {}
        citation_link_total: Dict[Tuple[str, str], int] = {}
        substantive_sentences: Dict[Tuple[str, str], int] = {}
        substantive_with_citation: Dict[Tuple[str, str], int] = {}
        cited_sentences_total: Dict[Tuple[str, str], int] = {}
        cited_sentences_resolved: Dict[Tuple[str, str], int] = {}
        covered_nuggets: Dict[Tuple[str, str], int] = {}
        total_nuggets: Dict[Tuple[str, str], int] = {}

        def key(r: Report) -> Tuple[str, str]:
            return (r.metadata.run_id, r.metadata.topic_id)

        verify_requests: List[MinimaLlmRequest] = []
        verify_index: List[Tuple[Tuple[str, str], List[str]]] = []  # (key, cited_doc_ids)

        for response in responses:
            k = key(response)
            claim_score_sum.setdefault(k, 0.0)
            claim_count.setdefault(k, 0)
            contradicted_count.setdefault(k, 0)
            citation_link_supported.setdefault(k, 0)
            citation_link_total.setdefault(k, 0)
            substantive_sentences.setdefault(k, 0)
            substantive_with_citation.setdefault(k, 0)
            cited_sentences_total.setdefault(k, 0)
            cited_sentences_resolved.setdefault(k, 0)

            documents = response.documents or {}
            for sentence in response.get_sentences_with_citations():
                text = (sentence.text or "").strip()
                if not text:
                    continue
                is_substantive = len(text.split()) >= substantive_min_words
                if is_substantive:
                    substantive_sentences[k] += 1

                citation_ids = list(sentence.citations or [])
                if not citation_ids:
                    # No citation at all: a real grounding gap, not a data-availability
                    # problem -- counts against context_faithfulness as unsupported.
                    claim_count[k] += 1
                    continue
                if is_substantive:
                    substantive_with_citation[k] += 1

                cited_sentences_total[k] += 1
                resolved_docs = [(doc_id, documents[doc_id]) for doc_id in citation_ids if doc_id in documents]
                if not resolved_docs:
                    # Citations exist but we structurally can't check them (report.documents
                    # missing or doesn't contain the cited doc_id) -- excluded from
                    # context_faithfulness/citation_precision, not scored as unsupported.
                    continue
                cited_sentences_resolved[k] += 1

                doc_texts = [doc.get_text()[:max_doc_chars] for _doc_id, doc in resolved_docs]
                evidence_block = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(doc_texts))
                prompt = (
                    f"Claim: {text}\n\nCited evidence:\n{evidence_block}\n\n"
                    f"Judge whether the cited evidence supports the claim. Reply with ONLY this JSON:\n"
                    f'{{"claim_verdict": "SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|NOT_ENOUGH_EVIDENCE", '
                    f'"citation_supports": [true or false, one per numbered evidence item, same order]}}'
                )
                verify_requests.append(MinimaLlmRequest(
                    request_id=f"verify-{len(verify_index)}",
                    messages=[
                        {"role": "system", "content": "You are a strict fact-checker. Reply with only the requested JSON."},
                        {"role": "user", "content": prompt},
                    ],
                ))
                verify_index.append((k, [doc_id for doc_id, _doc in resolved_docs]))

        coverage_requests: List[MinimaLlmRequest] = []
        coverage_index: List[Tuple[Tuple[str, str], str]] = []  # (key, question)
        if nugget_banks is not None:
            for response in responses:
                k = key(response)
                bank = nugget_banks.banks.get(response.metadata.topic_id)
                if bank is None:
                    continue
                questions = [q.question for q in bank.nuggets_as_list() if getattr(q, "question", None)]
                total_nuggets[k] = len(questions)
                covered_nuggets.setdefault(k, 0)
                if not questions:
                    continue
                answer_text = response.get_report_text()
                for question in questions:
                    coverage_requests.append(MinimaLlmRequest(
                        request_id=f"coverage-{len(coverage_index)}",
                        messages=[
                            {"role": "system", "content": "Reply with only YES or NO."},
                            {"role": "user", "content": (
                                f"Answer:\n{answer_text}\n\nDoes the answer address this required point?\n"
                                f"Point: {question}\n\nReply with only YES or NO."
                            )},
                        ],
                    ))
                    coverage_index.append((k, question))

        backend = _backend(llm_config)
        verify_results = asyncio.run(backend.run_batched(verify_requests)) if verify_requests else []
        coverage_results = asyncio.run(backend.run_batched(coverage_requests)) if coverage_requests else []

        for (k, doc_ids), result in zip(verify_index, verify_results):
            parsed = self._parse_verdict(result, expected_citations=len(doc_ids))
            # A failed/unparseable LLM call is excluded from the denominator entirely
            # (not scored as unsupported) -- a request timeout or malformed reply is a
            # judge-infrastructure failure, not a semantic verdict, and must not bias
            # CONTEXT_FAITHFULNESS/CRITICAL_ERROR_RATE downward.
            if parsed["claim_verdict"] is not None:
                claim_count[k] += 1
                claim_score_sum[k] += _SUPPORT_SCORE[parsed["claim_verdict"]]
                if parsed["claim_verdict"] == "CONTRADICTED":
                    contradicted_count[k] += 1
            for supports in parsed["citation_supports"]:
                if supports is None:
                    continue
                citation_link_total[k] += 1
                if supports:
                    citation_link_supported[k] += 1

        for (k, _question), result in zip(coverage_index, coverage_results):
            if self._parse_yes(result):
                covered_nuggets[k] += 1

        builder = LeaderboardBuilder(RAGREE_SPEC)
        seen_keys: set = set()
        for response in responses:
            k = key(response)
            if k in seen_keys:
                # Two Report rows for the same (run_id, topic_id): the per-key counters
                # above already merged both, so a second builder.add() would just repeat
                # the same values under one key rather than expressing anything new.
                print(f"[RAGreeEvidenceJudge] Duplicate response for run={k[0]!r} topic={k[1]!r}, skipping repeat leaderboard row")
                continue
            seen_keys.add(k)
            run_id, topic_id = k

            n_claims = claim_count.get(k, 0)
            context_faithfulness = (claim_score_sum.get(k, 0.0) / n_claims) if n_claims else 0.0
            critical_error_rate = (contradicted_count.get(k, 0) / n_claims) if n_claims else 0.0

            n_links = citation_link_total.get(k, 0)
            citation_precision = (citation_link_supported.get(k, 0) / n_links) if n_links else 0.0

            n_substantive = substantive_sentences.get(k, 0)
            citation_coverage = (substantive_with_citation.get(k, 0) / n_substantive) if n_substantive else 0.0

            n_cited = cited_sentences_total.get(k, 0)
            evidence_coverage_rate = (cited_sentences_resolved.get(k, 0) / n_cited) if n_cited else 0.0

            n_nuggets = total_nuggets.get(k, 0)
            estimated_coverage = (covered_nuggets.get(k, 0) / n_nuggets) if n_nuggets else 0.0

            claim_f1 = _harmonic_mean(context_faithfulness, estimated_coverage)

            builder.add(
                run_id=run_id,
                topic_id=topic_id,
                values={
                    "CONTEXT_FAITHFULNESS": context_faithfulness,
                    "ESTIMATED_COVERAGE": estimated_coverage,
                    "CLAIM_F1": claim_f1,
                    "CITATION_PRECISION": citation_precision,
                    "CITATION_COVERAGE": citation_coverage,
                    "CRITICAL_ERROR_RATE": critical_error_rate,
                    "EVIDENCE_COVERAGE_RATE": evidence_coverage_rate,
                },
            )

        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing=on_missing_evals)
        leaderboard.verify(expected_topic_ids=expected_topic_ids, warn=True, on_missing=on_missing_evals)
        print(f"RAGreeEvidenceJudge: built leaderboard with {len(leaderboard.entries)} entries")
        return leaderboard

    @staticmethod
    def _parse_verdict(result: Any, expected_citations: int) -> Dict[str, Any]:
        """Returns {"claim_verdict": str-or-None, "citation_supports": list[bool-or-None]}.

        None means "excluded from the denominator" (request failed, or that
        particular field couldn't be parsed) -- distinct from a genuine
        NOT_ENOUGH_EVIDENCE verdict, which the model actually returned. A
        malformed claim_verdict does not imply anything about citation_supports
        or vice versa; each is parsed and defaulted to None independently so one
        bad field doesn't discard the other.
        """
        unavailable = {"claim_verdict": None, "citation_supports": [None] * expected_citations}
        if not isinstance(result, MinimaLlmResponse):
            print(f"[RAGreeEvidenceJudge] LLM error on claim verification: {result}")
            return unavailable
        parsed = _extract_json(result.text)
        if not parsed:
            print(f"[RAGreeEvidenceJudge] Could not parse claim verdict JSON: {result.text[:200]!r}")
            return unavailable

        verdict_raw = str(parsed.get("claim_verdict", "")).strip().upper()
        verdict = verdict_raw if verdict_raw in _VERDICTS else None
        if verdict is None:
            print(f"[RAGreeEvidenceJudge] Unrecognized claim_verdict {verdict_raw!r}, excluding from CONTEXT_FAITHFULNESS")

        supports_raw = parsed.get("citation_supports")
        if (
            isinstance(supports_raw, list)
            and len(supports_raw) == expected_citations
            and all(isinstance(s, bool) for s in supports_raw)
        ):
            supports: List[Optional[bool]] = list(supports_raw)
        else:
            if expected_citations:
                print(f"[RAGreeEvidenceJudge] Malformed citation_supports {supports_raw!r}, excluding {expected_citations} link(s) from CITATION_PRECISION")
            supports = [None] * expected_citations
        return {"claim_verdict": verdict, "citation_supports": supports}

    @staticmethod
    def _parse_yes(result: Any) -> bool:
        if not isinstance(result, MinimaLlmResponse):
            print(f"[RAGreeEvidenceJudge] LLM error on nugget coverage: {result}")
            return False
        text = result.text.strip().lower()
        return text.startswith("yes") or "yes" in text.split()
