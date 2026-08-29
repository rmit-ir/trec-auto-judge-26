#!/usr/bin/env python3
"""RAGree Choice: an evidence-gated preferential AutoJudge powered by SPARROW.

This adapter uses the task-injected OpenAI-compatible endpoint.  It deliberately
does not hard-code an AWS model or credentials: a multi-model Bedrock jury is
implemented by an external orchestrator supplying differently calibrated
``Ballot`` objects to :mod:`judges.ragree_choice.sparrow`.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from autojudge_base import (
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LlmConfigProtocol,
    MeasureSpec,
    NuggetBanksProtocol,
    Qrels,
    Report,
    Request,
)
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm

from .sparrow import Ballot, RankingAudit, audit_ranking, davidson_win_probability, seen_edges, select_adaptive_pairs, sparse_seed_graph


RAGREE_CHOICE_SPEC = LeaderboardSpec(measures=(
    MeasureSpec(
        "CHOICE_SCORE",
        description=(
            "Pool-relative preferential score (0-100): the fitted Davidson--Bradley--"
            "Terry probability of beating a uniformly selected eligible competitor, "
            "scaled to percent. This is not comparable across changing candidate pools "
            "without frozen reference anchors."
        ),
    ),
    MeasureSpec(
        "EVIDENCE_ELIGIBLE",
        description=(
            "1.0 when the pointwise evidence gate found no critical error and both "
            "factual-support and citation-validity estimates meet configured thresholds; "
            "0.0 otherwise. Ineligible candidates are excluded from preference wins."
        ),
    ),
    MeasureSpec(
        "EVIDENCE_EVALUATED",
        description=(
            "1.0 when a complete, schema-valid pointwise evidence brief was obtained; "
            "0.0 when calls failed or returned malformed output after bounded retries. "
            "Unavailable evidence is excluded from eligibility decisions, not treated "
            "as an evidence-ineligible verdict."
        ),
    ),
    MeasureSpec(
        "POINTWISE_EVIDENCE_SCORE",
        description=(
            "Mean of the LLM's pointwise factual-support, completeness, and citation-"
            "validity estimates. It is an evidence brief for this prototype, not a gold "
            "claim-precision/recall measurement. Interpret only when EVIDENCE_EVALUATED "
            "is 1.0; unavailable briefs use 0.0 solely as a numeric placeholder."
        ),
    ),
    MeasureSpec(
        "PAIRWISE_WIN_RATE",
        description=(
            "Observed weighted fraction of non-abstaining pairwise ballots won by this "
            "candidate within the query. Ties contribute half a win."
        ),
    ),
    MeasureSpec(
        "CONDORCET_WINNER",
        description=(
            "1.0 if this candidate strictly defeats every other eligible candidate in "
            "the observed weighted preference matrix; 0.0 otherwise."
        ),
    ),
    MeasureSpec(
        "RANKING_STABLE",
        description=(
            "1.0 when the operational Davidson--Bradley--Terry and independent Kemeny "
            "audit select the same winner; 0.0 when they disagree or no eligible answer "
            "exists."
        ),
    ),
    MeasureSpec(
        "CYCLE_RATE",
        description=(
            "Fraction of eligible candidate triples that form a directed three-cycle in "
            "the weighted observed preference matrix. Lower is more consistent."
        ),
    ),
    MeasureSpec(
        "BALLOT_COVERAGE",
        description=(
            "Fraction of possible opponents for which this eligible candidate received "
            "at least one non-abstaining ballot. Sparse SPARROW acquisition need not be "
            "a complete round robin."
        ),
    ),
    MeasureSpec(
        "NO_ACCEPTABLE_ANSWER",
        description=(
            "1.0 when every candidate for this topic failed the evidence eligibility "
            "gate *after every candidate was successfully evidence-evaluated*; 0.0 "
            "otherwise. Transient LLM failure alone can never trigger this verdict."
        ),
    ),
))


@dataclass(frozen=True)
class EvidenceBrief:
    critical_error: bool
    factual_support: float
    completeness: float
    citation_validity: float
    confidence: float

    @property
    def score(self) -> float:
        return (self.factual_support + self.completeness + self.citation_validity) / 3.0


def _backend(llm_config: LlmConfigProtocol) -> OpenAIMinimaLlm:
    config = MinimaLlmConfig.from_dict(llm_config.raw) if getattr(llm_config, "raw", None) else MinimaLlmConfig.from_env()
    return OpenAIMinimaLlm(config)


def _json_object(text: str) -> Optional[dict[str, Any]]:
    decoder = json.JSONDecoder()
    index = text.find("{")
    while index >= 0:
        try:
            value, _end = decoder.raw_decode(text, index)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        index = text.find("{", index + 1)
    return None


def _bounded_score(value: Any) -> Optional[float]:
    """Return a strict JSON number in [0, 1], or None for malformed fields."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


class RAGreeChoiceJudge:
    """Evidence-gated pairwise ranking for all run responses to each topic."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        *,
        factual_support_threshold: float = 0.65,
        citation_validity_threshold: float = 0.65,
        full_round_robin_max_candidates: int = 5,
        initial_graph_degree: int = 4,
        adaptive_comparisons_per_topic: int = 4,
        top_k: int = 1,
        max_answer_chars: int = 2000,
        max_evidence_chars: int = 4000,
        judge_weight: float = 1.0,
        exact_kemeny_limit: int = 8,
        llm_max_attempts: int = 3,
        llm_retry_backoff_seconds: float = 0.25,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        self._validate_config(
            factual_support_threshold=factual_support_threshold,
            citation_validity_threshold=citation_validity_threshold,
            full_round_robin_max_candidates=full_round_robin_max_candidates,
            initial_graph_degree=initial_graph_degree,
            adaptive_comparisons_per_topic=adaptive_comparisons_per_topic,
            top_k=top_k,
            max_answer_chars=max_answer_chars,
            max_evidence_chars=max_evidence_chars,
            judge_weight=judge_weight,
            exact_kemeny_limit=exact_kemeny_limit,
            llm_max_attempts=llm_max_attempts,
            llm_retry_backoff_seconds=llm_retry_backoff_seconds,
        )
        topics = {topic.request_id: topic for topic in rag_topics}
        expected_topic_ids = list(topics)
        reports = self._unique_reports(rag_responses)
        reports_by_topic: Dict[str, list[Report]] = {}
        for report in reports:
            reports_by_topic.setdefault(report.metadata.topic_id, []).append(report)
        for topic_reports in reports_by_topic.values():
            topic_reports.sort(key=lambda report: report.metadata.run_id)

        briefs = self._make_evidence_briefs(
            reports, topics, llm_config, max_answer_chars=max_answer_chars,
            max_evidence_chars=max_evidence_chars,
            llm_max_attempts=llm_max_attempts,
            llm_retry_backoff_seconds=llm_retry_backoff_seconds,
        )
        eligibility = {
            key: brief is not None and self._eligible(
                brief, factual_support_threshold=factual_support_threshold,
                citation_validity_threshold=citation_validity_threshold,
            )
            for key, brief in briefs.items()
        }

        topic_ballots: Dict[str, list[Ballot]] = {topic_id: [] for topic_id in reports_by_topic}
        initial_pairs: list[tuple[str, Report, Report]] = []
        for topic_id, topic_reports in reports_by_topic.items():
            eligible = [report for report in topic_reports if eligibility[self._key(report)]]
            by_run = {report.metadata.run_id: report for report in eligible}
            for run_a, run_b in sparse_seed_graph(
                list(by_run), initial_graph_degree, full_round_robin_max_candidates,
            ):
                initial_pairs.append((topic_id, by_run[run_a], by_run[run_b]))

        for topic_id, ballot in self._cast_pair_ballots(
            initial_pairs, topics, briefs, llm_config, judge_weight, max_answer_chars,
            llm_max_attempts, llm_retry_backoff_seconds,
        ):
            topic_ballots[topic_id].append(ballot)

        # One micro-batch of boundary-focused refinement. The max-heap scheduler
        # avoids a serial noisy-sort path while concentrating calls where rank can move.
        adaptive_pairs: list[tuple[str, Report, Report]] = []
        for topic_id, topic_reports in reports_by_topic.items():
            eligible = [report for report in topic_reports if eligibility[self._key(report)]]
            by_run = {report.metadata.run_id: report for report in eligible}
            if len(by_run) < 2:
                continue
            audit = audit_ranking(list(by_run), topic_ballots[topic_id], exact_kemeny_limit)
            seen = seen_edges(topic_ballots[topic_id])
            for run_a, run_b in select_adaptive_pairs(
                list(by_run), seen, audit.utilities, top_k=top_k,
                budget=adaptive_comparisons_per_topic,
            ):
                adaptive_pairs.append((topic_id, by_run[run_a], by_run[run_b]))
        for topic_id, ballot in self._cast_pair_ballots(
            adaptive_pairs, topics, briefs, llm_config, judge_weight, max_answer_chars,
            llm_max_attempts, llm_retry_backoff_seconds,
        ):
            topic_ballots[topic_id].append(ballot)

        builder = LeaderboardBuilder(RAGREE_CHOICE_SPEC)
        for topic_id, topic_reports in reports_by_topic.items():
            eligible_reports = [report for report in topic_reports if eligibility[self._key(report)]]
            eligible_ids = [report.metadata.run_id for report in eligible_reports]
            audit = (
                audit_ranking(eligible_ids, topic_ballots[topic_id], exact_kemeny_limit)
                if eligible_ids else None
            )
            no_acceptable_answer = self._no_acceptable_answer(
                [briefs[self._key(report)] for report in topic_reports], eligible_ids,
            )
            for report in topic_reports:
                key = self._key(report)
                run_id = report.metadata.run_id
                brief = briefs[key]
                if brief is None:
                    values = self._unavailable_values(no_acceptable_answer)
                elif audit is None or run_id not in audit.utilities:
                    values = self._ineligible_values(brief, no_acceptable_answer)
                else:
                    values = self._eligible_values(run_id, eligible_ids, audit, topic_ballots[topic_id], brief)
                    values["NO_ACCEPTABLE_ANSWER"] = no_acceptable_answer
                builder.add(run_id=run_id, topic_id=topic_id, values=values)

        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing="fix_aggregate")
        leaderboard.verify(expected_topic_ids=expected_topic_ids, warn=True, on_missing="fix_aggregate")
        print(f"RAGreeChoiceJudge: ranked {len(reports)} responses across {len(reports_by_topic)} topics")
        return leaderboard

    @staticmethod
    def _key(report: Report) -> tuple[str, str]:
        return (report.metadata.run_id, report.metadata.topic_id)

    @staticmethod
    def _validate_config(**settings: Any) -> None:
        bounded = ("factual_support_threshold", "citation_validity_threshold")
        for name in bounded:
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite number in [0, 1]")

        positive_integers = (
            "full_round_robin_max_candidates", "top_k", "max_answer_chars",
            "max_evidence_chars", "exact_kemeny_limit", "llm_max_attempts",
        )
        for name in positive_integers:
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        non_negative_integers = ("initial_graph_degree", "adaptive_comparisons_per_topic")
        for name in non_negative_integers:
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        for name in ("judge_weight", "llm_retry_backoff_seconds"):
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if settings["judge_weight"] <= 0:
            raise ValueError("judge_weight must be positive")
        if settings["llm_retry_backoff_seconds"] < 0:
            raise ValueError("llm_retry_backoff_seconds must be non-negative")

    def _unique_reports(self, rag_responses: Iterable[Report]) -> list[Report]:
        unique: Dict[tuple[str, str], Report] = {}
        for report in sorted(rag_responses, key=lambda item: (item.metadata.run_id, item.metadata.topic_id)):
            key = self._key(report)
            if key in unique:
                print(f"[RAGreeChoiceJudge] Duplicate response for run={key[0]!r} topic={key[1]!r}; keeping first")
                continue
            unique[key] = report
        return list(unique.values())

    def _make_evidence_briefs(
        self,
        reports: Sequence[Report],
        topics: Mapping[str, Request],
        llm_config: LlmConfigProtocol,
        *,
        max_answer_chars: int,
        max_evidence_chars: int,
        llm_max_attempts: int,
        llm_retry_backoff_seconds: float,
    ) -> Dict[tuple[str, str], Optional[EvidenceBrief]]:
        requests: list[MinimaLlmRequest] = []
        request_keys: list[tuple[str, str]] = []
        for index, report in enumerate(reports):
            topic = topics.get(report.metadata.topic_id)
            question = topic.title if topic else report.metadata.topic_id
            answer = report.get_report_text()[:max_answer_chars]
            evidence = self._evidence_packet(report, max_evidence_chars)
            prompt = (
                "Assess this answer pointwise before any pairwise comparison. Ground the "
                "assessment only in the supplied cited evidence. A missing or unresolvable "
                "citation is a weakness; mark critical_error true only for a material "
                "contradiction, fabrication, or unsafe failure visible from the packet. "
                "Reply with ONLY JSON: {\"critical_error\": boolean, "
                "\"factual_support\": number_0_to_1, \"completeness\": number_0_to_1, "
                "\"citation_validity\": number_0_to_1, \"confidence\": number_0_to_1}.\n\n"
                f"Question:\n{question}\n\nAnswer:\n{answer}\n\nCited evidence:\n{evidence}"
            )
            requests.append(MinimaLlmRequest(
                request_id=f"choice-brief-{index}",
                messages=[
                    {"role": "system", "content": "You are an evidence-first RAG evaluator. Return only the requested JSON."},
                    {"role": "user", "content": prompt},
                ],
            ))
            request_keys.append(self._key(report))

        results = self._run_parsed_with_retries(
            llm_config, requests, self._parse_brief,
            max_attempts=llm_max_attempts,
            retry_backoff_seconds=llm_retry_backoff_seconds,
            operation="evidence brief",
        )
        return {
            key: result
            for key, result in zip(request_keys, results)
        }

    @staticmethod
    def _evidence_packet(report: Report, max_chars: int) -> str:
        documents = report.documents or {}
        snippets: list[str] = []
        seen: set[str] = set()
        for sentence in report.get_sentences_with_citations():
            for document_id in sentence.citations or []:
                if document_id in seen or document_id not in documents:
                    continue
                seen.add(document_id)
                snippets.append(f"[{document_id}] {documents[document_id].get_text()}")
        packet = "\n\n".join(snippets)
        return packet[:max_chars] if packet else "No cited document text was available."

    @staticmethod
    def _parse_brief(result: Any) -> Optional[EvidenceBrief]:
        if not isinstance(result, MinimaLlmResponse):
            return None
        parsed = _json_object(result.text)
        if not parsed:
            return None
        critical_error = parsed.get("critical_error")
        factual_support = _bounded_score(parsed.get("factual_support"))
        completeness = _bounded_score(parsed.get("completeness"))
        citation_validity = _bounded_score(parsed.get("citation_validity"))
        confidence = _bounded_score(parsed.get("confidence"))
        # Schema failures are unavailable, never an inferred "no critical error"
        # or a plausible-looking all-zero evidence verdict.
        if (
            not isinstance(critical_error, bool)
            or factual_support is None
            or completeness is None
            or citation_validity is None
            or confidence is None
        ):
            return None
        return EvidenceBrief(
            critical_error=critical_error,
            factual_support=factual_support,
            completeness=completeness,
            citation_validity=citation_validity,
            confidence=confidence,
        )

    @staticmethod
    def _eligible(brief: EvidenceBrief, *, factual_support_threshold: float, citation_validity_threshold: float) -> bool:
        return (
            not brief.critical_error
            and brief.factual_support >= factual_support_threshold
            and brief.citation_validity >= citation_validity_threshold
        )

    @staticmethod
    def _no_acceptable_answer(
        briefs: Sequence[Optional[EvidenceBrief]], eligible_ids: Sequence[str],
    ) -> float:
        """Only a fully evaluated, unanimously rejected topic has this verdict."""
        return float(bool(briefs) and all(brief is not None for brief in briefs) and not eligible_ids)

    @staticmethod
    def _run_parsed_with_retries(
        llm_config: LlmConfigProtocol,
        requests: Sequence[MinimaLlmRequest],
        parser: Callable[[Any], Optional[Any]],
        *,
        max_attempts: int,
        retry_backoff_seconds: float,
        operation: str,
    ) -> list[Optional[Any]]:
        """Retry only failed or malformed items, retaining valid first results."""
        parsed_results: list[Optional[Any]] = [None] * len(requests)
        pending = list(range(len(requests)))
        for attempt in range(max_attempts):
            if not pending:
                break
            if attempt:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
            batch = [requests[index] for index in pending]
            try:
                raw_results = asyncio.run(_backend(llm_config).run_batched(batch))
            except Exception as error:
                # Do not surface a provider payload: it may contain a restricted prompt.
                print(f"[RAGreeChoiceJudge] {operation} batch failed with {type(error).__name__}; retrying")
                raw_results = []
            next_pending: list[int] = []
            for index, result in zip(pending, raw_results):
                parsed = parser(result)
                if parsed is None:
                    next_pending.append(index)
                else:
                    parsed_results[index] = parsed
            # minima-llm promises one result per request; retain a defensive retry
            # path if a custom backend nevertheless returns a short list.
            next_pending.extend(pending[len(raw_results):])
            pending = next_pending
        if pending:
            print(f"[RAGreeChoiceJudge] {len(pending)} {operation} request(s) unavailable after {max_attempts} attempt(s)")
        return parsed_results

    def _cast_pair_ballots(
        self,
        pairs: Sequence[tuple[str, Report, Report]],
        topics: Mapping[str, Request],
        briefs: Mapping[tuple[str, str], Optional[EvidenceBrief]],
        llm_config: LlmConfigProtocol,
        judge_weight: float,
        max_answer_chars: int,
        llm_max_attempts: int,
        llm_retry_backoff_seconds: float,
    ) -> list[tuple[str, Ballot]]:
        requests: list[MinimaLlmRequest] = []
        metadata: list[tuple[str, str, str]] = []
        for index, (topic_id, report_a, report_b) in enumerate(pairs):
            brief_a = briefs[self._key(report_a)]
            brief_b = briefs[self._key(report_b)]
            # Pairs are created only from evidence-eligible reports, which implies
            # valid briefs. Keep the invariant explicit rather than coercing None.
            assert brief_a is not None and brief_b is not None
            topic = topics.get(topic_id)
            question = topic.title if topic else topic_id
            prompt = (
                "Use pointwise-first, pairwise-final judging. Prefer factual support, "
                "material completeness, directness, and appropriate uncertainty over "
                "style or verbosity. Do not infer model identity. Vote TIE when the "
                "difference is immaterial and ABSTAIN when evidence is insufficient. "
                "Reply with ONLY JSON: {\"vote\": \"A|B|TIE|ABSTAIN\"}.\n\n"
                f"Question:\n{question}\n\n"
                f"Pointwise evidence brief A: {self._brief_text(brief_a)}\n"
                f"Answer A:\n{report_a.get_report_text()[:max_answer_chars]}\n\n"
                f"Pointwise evidence brief B: {self._brief_text(brief_b)}\n"
                f"Answer B:\n{report_b.get_report_text()[:max_answer_chars]}"
            )
            requests.append(MinimaLlmRequest(
                request_id=f"choice-ballot-{index}",
                messages=[
                    {"role": "system", "content": "You are an evidence-informed preferential RAG judge. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
            ))
            metadata.append((topic_id, report_a.metadata.run_id, report_b.metadata.run_id))

        votes = self._run_parsed_with_retries(
            llm_config, requests, self._parse_vote,
            max_attempts=llm_max_attempts,
            retry_backoff_seconds=llm_retry_backoff_seconds,
            operation="pairwise ballot",
        )
        ballots: list[tuple[str, Ballot]] = []
        for (topic_id, candidate_a, candidate_b), vote in zip(metadata, votes):
            if vote is None:
                continue
            ballots.append((topic_id, Ballot(
                candidate_a=candidate_a,
                candidate_b=candidate_b,
                vote=vote,
                weight=judge_weight,
            )))
        return ballots

    @staticmethod
    def _brief_text(brief: EvidenceBrief) -> str:
        return (
            f"critical_error={brief.critical_error}; factual_support={brief.factual_support:.2f}; "
            f"completeness={brief.completeness:.2f}; citation_validity={brief.citation_validity:.2f}; "
            f"confidence={brief.confidence:.2f}"
        )

    @staticmethod
    def _parse_vote(result: Any) -> Optional[str]:
        if not isinstance(result, MinimaLlmResponse):
            return None
        parsed = _json_object(result.text)
        if not parsed:
            return None
        vote = str(parsed.get("vote", "")).strip().upper()
        return vote if vote in {"A", "B", "TIE", "ABSTAIN"} else None

    @staticmethod
    def _unavailable_values(no_acceptable_answer: float) -> dict[str, float]:
        return {
            "CHOICE_SCORE": 0.0,
            "EVIDENCE_ELIGIBLE": 0.0,
            "EVIDENCE_EVALUATED": 0.0,
            "POINTWISE_EVIDENCE_SCORE": 0.0,
            "PAIRWISE_WIN_RATE": 0.0,
            "CONDORCET_WINNER": 0.0,
            "RANKING_STABLE": 0.0,
            "CYCLE_RATE": 0.0,
            "BALLOT_COVERAGE": 0.0,
            "NO_ACCEPTABLE_ANSWER": no_acceptable_answer,
        }

    @staticmethod
    def _ineligible_values(brief: EvidenceBrief, no_acceptable_answer: float) -> dict[str, float]:
        return {
            "CHOICE_SCORE": 0.0,
            "EVIDENCE_ELIGIBLE": 0.0,
            "EVIDENCE_EVALUATED": 1.0,
            "POINTWISE_EVIDENCE_SCORE": brief.score,
            "PAIRWISE_WIN_RATE": 0.0,
            "CONDORCET_WINNER": 0.0,
            "RANKING_STABLE": 0.0,
            "CYCLE_RATE": 0.0,
            "BALLOT_COVERAGE": 0.0,
            "NO_ACCEPTABLE_ANSWER": no_acceptable_answer,
        }

    @staticmethod
    def _eligible_values(
        candidate: str,
        eligible_ids: Sequence[str],
        audit: RankingAudit,
        ballots: Sequence[Ballot],
        brief: EvidenceBrief,
    ) -> dict[str, float]:
        opponents = [other for other in eligible_ids if other != candidate]
        if opponents:
            choice_score = 100.0 * sum(
                davidson_win_probability(audit.utilities[candidate], audit.utilities[opponent])
                for opponent in opponents
            ) / len(opponents)
        else:
            choice_score = 50.0

        wins = ties = total = 0.0
        compared_opponents: set[str] = set()
        for ballot in ballots:
            if candidate not in {ballot.candidate_a, ballot.candidate_b} or ballot.vote == "ABSTAIN":
                continue
            opponent = ballot.candidate_b if candidate == ballot.candidate_a else ballot.candidate_a
            compared_opponents.add(opponent)
            total += ballot.weight
            if ballot.vote == "TIE":
                ties += ballot.weight
            elif (ballot.vote == "A" and candidate == ballot.candidate_a) or (ballot.vote == "B" and candidate == ballot.candidate_b):
                wins += ballot.weight
        return {
            "CHOICE_SCORE": choice_score,
            "EVIDENCE_ELIGIBLE": 1.0,
            "EVIDENCE_EVALUATED": 1.0,
            "POINTWISE_EVIDENCE_SCORE": brief.score,
            "PAIRWISE_WIN_RATE": (wins + 0.5 * ties) / total if total else 0.0,
            "CONDORCET_WINNER": float(audit.condorcet_winner == candidate),
            "RANKING_STABLE": float(audit.kemeny_agrees_on_winner),
            "CYCLE_RATE": audit.cycle_rate,
            "BALLOT_COVERAGE": len(compared_opponents) / len(opponents) if opponents else 1.0,
            "NO_ACCEPTABLE_ANSWER": 0.0,
        }
