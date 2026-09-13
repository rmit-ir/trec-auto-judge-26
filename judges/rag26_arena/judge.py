#!/usr/bin/env python3
"""RAG26 Arena: RAGDoll/TREC RAG 2026-style pairwise preference AutoJudge.

``arena-only``: pairwise Arena ranking via SPARROW, with no rubric
conditioning or evidence gate (docs/trec-rag-2026-judge-implementation-plan.md
Phase 1 / docs/rag26-tira-submission-plan.md Phase 1). Every response in a
topic is comparison-eligible -- unlike ``judges.ragree_choice``, there is no
pointwise evidence gate here by design (plan Section 8.3 / Section 27): the
public 2026 evaluator keeps preference, rubric, and citation-support signals
separate, so this judge produces only the preference signal.

Reuses ``judges.ragree_choice.sparrow`` unchanged for the seed graph, adaptive
acquisition, Davidson-Bradley-Terry fit, and Condorcet/Kemeny/cycle audits,
and a few small private helpers from ``judges.ragree_choice.ragree_choice_judge``
(JSON extraction, the injected-endpoint backend, retry loop, and the A/B
position-reversal map) rather than duplicating them -- see that module's
docstring and docs/trec-rag-2026-judge-implementation-plan.md Section 5 ("do
not refactor into judges/common/ immediately").
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

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
from minima_llm import MinimaLlmRequest, MinimaLlmResponse

from judges.ragree_choice.ragree_choice_judge import _REVERSE_VOTE_MAP, _backend, _json_object
from judges.ragree_choice.sparrow import (
    Ballot,
    RankingAudit,
    audit_ranking,
    davidson_win_probability,
    seen_edges,
    select_adaptive_pairs,
    select_reversal_ballots,
    sparse_seed_graph,
)

from .models import BALLOT_VOTE, RAW_VOTES
from .prompts import arena_messages

RAG26_ARENA_SPEC = LeaderboardSpec(measures=(
    MeasureSpec(
        "RAG26_ARENA_SCORE",
        description=(
            "Pool-relative Arena preference score (0-100): the fitted "
            "Davidson--Bradley--Terry probability of this response beating a "
            "uniformly selected other response to the same topic, scaled to "
            "percent. Not comparable across changing candidate pools without "
            "frozen reference anchors. 50.0 for a topic with only one response."
        ),
    ),
    MeasureSpec(
        "PAIRWISE_WIN_RATE",
        description=(
            "Observed weighted fraction of non-abstaining Arena ballots won by "
            "this response within the topic; ties (including BOTH_BAD, which "
            "the fit treats as a tie) contribute half a win. 0.0 when no ballot "
            "was cast for this response."
        ),
    ),
    MeasureSpec(
        "CONDORCET_WINNER",
        description=(
            "1.0 if this response strictly beats every other response to the "
            "topic in the observed weighted preference matrix; 0.0 otherwise."
        ),
    ),
    MeasureSpec(
        "RANKING_STABLE",
        description=(
            "1.0 when the operational Davidson--Bradley--Terry ranking and the "
            "independent Kemeny audit agree on the topic's top response; 0.0 "
            "when they disagree."
        ),
    ),
    MeasureSpec(
        "CYCLE_RATE",
        description=(
            "Fraction of response triples for this topic that form a directed "
            "three-cycle in the weighted observed preference matrix. Lower is "
            "more internally consistent."
        ),
    ),
    MeasureSpec(
        "BALLOT_COVERAGE",
        description=(
            "Fraction of this topic's other responses against which this "
            "response received at least one non-abstaining ballot. Sparse "
            "SPARROW acquisition need not be a complete round robin. 1.0 for a "
            "topic with only one response."
        ),
    ),
    MeasureSpec(
        "POSITION_REVERSAL_CHECKED",
        description=(
            "Topic-level count of already-cast ballots re-checked with answer "
            "order swapped (0 when position_reversal_budget is 0 or no pair "
            "qualified). A reversal call that fails every retry leaves the "
            "original ballot unchanged and is not counted here."
        ),
    ),
    MeasureSpec(
        "POSITION_REVERSAL_DISAGREEMENT_RATE",
        description=(
            "Fraction of checked reversal ballots (see "
            "POSITION_REVERSAL_CHECKED) whose vote direction flipped when "
            "answer order was swapped. Such ballots are converted to ABSTAIN, "
            "not TIE. 0.0 when nothing was checked."
        ),
    ),
    MeasureSpec(
        "BOTH_BAD_RATE",
        description=(
            "Fraction of all Arena votes cast for this topic (initial, "
            "adaptive, and reversal) that were BOTH_BAD -- both compared "
            "answers judged unacceptable. Same value for every response in a "
            "topic; a diagnostic on the topic, not a per-response quality "
            "signal. 0.0 when no vote was cast."
        ),
    ),
))


class RAG26ArenaJudge:
    """SPARROW-ranked Arena preference judging for all responses to each topic."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        *,
        full_round_robin_max_candidates: int = 5,
        initial_graph_degree: int = 4,
        adaptive_comparisons_per_topic: int = 4,
        top_k: int = 1,
        arena_max_answer_chars: int = 2000,
        judge_weight: float = 1.0,
        exact_kemeny_limit: int = 8,
        llm_max_attempts: int = 3,
        llm_retry_backoff_seconds: float = 0.25,
        position_reversal_budget: int = 2,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        self._validate_config(
            full_round_robin_max_candidates=full_round_robin_max_candidates,
            initial_graph_degree=initial_graph_degree,
            adaptive_comparisons_per_topic=adaptive_comparisons_per_topic,
            top_k=top_k,
            arena_max_answer_chars=arena_max_answer_chars,
            judge_weight=judge_weight,
            exact_kemeny_limit=exact_kemeny_limit,
            llm_max_attempts=llm_max_attempts,
            llm_retry_backoff_seconds=llm_retry_backoff_seconds,
            position_reversal_budget=position_reversal_budget,
        )
        topics = {topic.request_id: topic for topic in rag_topics}
        expected_topic_ids = list(topics)
        reports = self._unique_reports(rag_responses)
        reports_by_topic: Dict[str, list[Report]] = {}
        for report in reports:
            reports_by_topic.setdefault(report.metadata.topic_id, []).append(report)
        for topic_reports in reports_by_topic.values():
            topic_reports.sort(key=lambda report: report.metadata.run_id)

        topic_ballots: Dict[str, list[Ballot]] = {topic_id: [] for topic_id in reports_by_topic}
        both_bad_count: Dict[str, int] = {}
        total_vote_count: Dict[str, int] = {}

        def record(topic_id: str, candidate_a: str, candidate_b: str, raw_vote: str) -> None:
            total_vote_count[topic_id] = total_vote_count.get(topic_id, 0) + 1
            if raw_vote == "BOTH_BAD":
                both_bad_count[topic_id] = both_bad_count.get(topic_id, 0) + 1
            topic_ballots[topic_id].append(Ballot(
                candidate_a=candidate_a, candidate_b=candidate_b,
                vote=BALLOT_VOTE[raw_vote], weight=judge_weight,
            ))

        initial_pairs: list[tuple[str, Report, Report]] = []
        for topic_id, topic_reports in reports_by_topic.items():
            by_run = {report.metadata.run_id: report for report in topic_reports}
            for run_a, run_b in sparse_seed_graph(
                list(by_run), initial_graph_degree, full_round_robin_max_candidates,
            ):
                initial_pairs.append((topic_id, by_run[run_a], by_run[run_b]))
        for topic_id, candidate_a, candidate_b, raw_vote in self._cast_pairs(
            initial_pairs, topics, llm_config, max_answer_chars=arena_max_answer_chars,
            llm_max_attempts=llm_max_attempts, llm_retry_backoff_seconds=llm_retry_backoff_seconds,
        ):
            record(topic_id, candidate_a, candidate_b, raw_vote)

        # One micro-batch of boundary-focused refinement, same acquisition
        # strategy as judges.ragree_choice.
        adaptive_pairs: list[tuple[str, Report, Report]] = []
        for topic_id, topic_reports in reports_by_topic.items():
            by_run = {report.metadata.run_id: report for report in topic_reports}
            if len(by_run) < 2:
                continue
            audit = audit_ranking(list(by_run), topic_ballots[topic_id], exact_kemeny_limit)
            seen = seen_edges(topic_ballots[topic_id])
            for run_a, run_b in select_adaptive_pairs(
                list(by_run), seen, audit.utilities, top_k=top_k,
                budget=adaptive_comparisons_per_topic,
            ):
                adaptive_pairs.append((topic_id, by_run[run_a], by_run[run_b]))
        for topic_id, candidate_a, candidate_b, raw_vote in self._cast_pairs(
            adaptive_pairs, topics, llm_config, max_answer_chars=arena_max_answer_chars,
            llm_max_attempts=llm_max_attempts, llm_retry_backoff_seconds=llm_retry_backoff_seconds,
        ):
            record(topic_id, candidate_a, candidate_b, raw_vote)

        reversal_checked: Dict[str, int] = {topic_id: 0 for topic_id in reports_by_topic}
        reversal_disagreed: Dict[str, int] = {topic_id: 0 for topic_id in reports_by_topic}
        if position_reversal_budget > 0:
            reversal_selections: list[tuple[str, Ballot]] = []
            for topic_id, topic_reports in reports_by_topic.items():
                by_run = {report.metadata.run_id: report for report in topic_reports}
                if len(by_run) < 2:
                    continue
                audit = audit_ranking(list(by_run), topic_ballots[topic_id], exact_kemeny_limit)
                for ballot in select_reversal_ballots(
                    list(by_run), topic_ballots[topic_id], audit.utilities,
                    top_k=top_k, budget=position_reversal_budget,
                ):
                    reversal_selections.append((topic_id, ballot))

            reports_by_run_by_topic = {
                topic_id: {report.metadata.run_id: report for report in topic_reports}
                for topic_id, topic_reports in reports_by_topic.items()
            }
            for topic_id, original_ballot, raw_vote in self._cast_reversal(
                reversal_selections, reports_by_run_by_topic, topics,
                llm_config, max_answer_chars=arena_max_answer_chars,
                llm_max_attempts=llm_max_attempts, llm_retry_backoff_seconds=llm_retry_backoff_seconds,
            ):
                total_vote_count[topic_id] = total_vote_count.get(topic_id, 0) + 1
                if raw_vote == "BOTH_BAD":
                    both_bad_count[topic_id] = both_bad_count.get(topic_id, 0) + 1
                reversed_vote = _REVERSE_VOTE_MAP[BALLOT_VOTE[raw_vote]]
                if reversed_vote == original_ballot.vote:
                    final_ballot, disagreed = original_ballot, False
                else:
                    final_ballot, disagreed = Ballot(
                        candidate_a=original_ballot.candidate_a,
                        candidate_b=original_ballot.candidate_b,
                        vote="ABSTAIN", weight=original_ballot.weight,
                    ), True
                edge = tuple(sorted((original_ballot.candidate_a, original_ballot.candidate_b)))
                topic_ballots[topic_id] = [
                    ballot for ballot in topic_ballots[topic_id]
                    if tuple(sorted((ballot.candidate_a, ballot.candidate_b))) != edge
                ] + [final_ballot]
                reversal_checked[topic_id] += 1
                if disagreed:
                    reversal_disagreed[topic_id] += 1

        builder = LeaderboardBuilder(RAG26_ARENA_SPEC)
        for topic_id, topic_reports in reports_by_topic.items():
            run_ids = [report.metadata.run_id for report in topic_reports]
            audit = audit_ranking(run_ids, topic_ballots[topic_id], exact_kemeny_limit)
            checked = reversal_checked.get(topic_id, 0)
            reversal_rate = (reversal_disagreed.get(topic_id, 0) / checked) if checked else 0.0
            total_votes = total_vote_count.get(topic_id, 0)
            both_bad_rate = (both_bad_count.get(topic_id, 0) / total_votes) if total_votes else 0.0
            for report in topic_reports:
                values = self._arena_values(report.metadata.run_id, run_ids, audit, topic_ballots[topic_id])
                values["POSITION_REVERSAL_CHECKED"] = float(checked)
                values["POSITION_REVERSAL_DISAGREEMENT_RATE"] = reversal_rate
                values["BOTH_BAD_RATE"] = both_bad_rate
                builder.add(run_id=report.metadata.run_id, topic_id=topic_id, values=values)

        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing="fix_aggregate")
        leaderboard.verify(expected_topic_ids=expected_topic_ids, warn=True, on_missing="fix_aggregate")
        print(f"RAG26ArenaJudge: ranked {len(reports)} responses across {len(reports_by_topic)} topics")
        return leaderboard

    @staticmethod
    def _key(report: Report) -> tuple[str, str]:
        return (report.metadata.run_id, report.metadata.topic_id)

    @staticmethod
    def _validate_config(**settings: Any) -> None:
        positive_integers = (
            "full_round_robin_max_candidates", "top_k", "arena_max_answer_chars",
            "exact_kemeny_limit", "llm_max_attempts",
        )
        for name in positive_integers:
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        non_negative_integers = ("initial_graph_degree", "adaptive_comparisons_per_topic", "position_reversal_budget")
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
                print(f"[RAG26ArenaJudge] Duplicate response for run={key[0]!r} topic={key[1]!r}; keeping first")
                continue
            unique[key] = report
        return list(unique.values())

    def _cast_pairs(
        self,
        pairs: Sequence[tuple[str, Report, Report]],
        topics: Mapping[str, Request],
        llm_config: LlmConfigProtocol,
        *,
        max_answer_chars: int,
        llm_max_attempts: int,
        llm_retry_backoff_seconds: float,
    ) -> list[tuple[str, str, str, str]]:
        requests: list[MinimaLlmRequest] = []
        metadata: list[tuple[str, str, str]] = []
        for index, (topic_id, report_a, report_b) in enumerate(pairs):
            requests.append(MinimaLlmRequest(
                request_id=f"arena-ballot-{index}",
                messages=arena_messages(
                    topics[topic_id], report_a.get_report_text(), report_b.get_report_text(),
                    max_answer_chars=max_answer_chars,
                ),
            ))
            metadata.append((topic_id, report_a.metadata.run_id, report_b.metadata.run_id))

        raw_votes = self._run_parsed_with_retries(
            llm_config, requests, self._parse_vote,
            max_attempts=llm_max_attempts, retry_backoff_seconds=llm_retry_backoff_seconds,
            operation="arena ballot",
        )
        return [
            (topic_id, candidate_a, candidate_b, raw_vote)
            for (topic_id, candidate_a, candidate_b), raw_vote in zip(metadata, raw_votes)
            if raw_vote is not None
        ]

    def _cast_reversal(
        self,
        selections: Sequence[tuple[str, Ballot]],
        reports_by_run_by_topic: Mapping[str, Mapping[str, Report]],
        topics: Mapping[str, Request],
        llm_config: LlmConfigProtocol,
        *,
        max_answer_chars: int,
        llm_max_attempts: int,
        llm_retry_backoff_seconds: float,
    ) -> list[tuple[str, Ballot, str]]:
        """Re-ask each selected ballot's pair with A/B swapped.

        A failed/unparseable reversal call yields nothing for that ballot --
        the caller leaves the original ballot untouched and does not count it
        as checked (mirrors judges.ragree_choice_judge._cast_reversal_ballots).
        """
        requests: list[MinimaLlmRequest] = []
        metadata: list[tuple[str, Ballot]] = []
        for index, (topic_id, original_ballot) in enumerate(selections):
            by_run = reports_by_run_by_topic[topic_id]
            # Swapped: the original candidate_b is presented as "Answer A" here.
            report_a = by_run[original_ballot.candidate_b]
            report_b = by_run[original_ballot.candidate_a]
            requests.append(MinimaLlmRequest(
                request_id=f"arena-reversal-{index}",
                messages=arena_messages(
                    topics[topic_id], report_a.get_report_text(), report_b.get_report_text(),
                    max_answer_chars=max_answer_chars,
                ),
            ))
            metadata.append((topic_id, original_ballot))

        raw_votes = self._run_parsed_with_retries(
            llm_config, requests, self._parse_vote,
            max_attempts=llm_max_attempts, retry_backoff_seconds=llm_retry_backoff_seconds,
            operation="position-reversal ballot",
        )
        return [
            (topic_id, original_ballot, raw_vote)
            for (topic_id, original_ballot), raw_vote in zip(metadata, raw_votes)
            if raw_vote is not None
        ]

    @staticmethod
    def _parse_vote(result: Any) -> Optional[str]:
        if not isinstance(result, MinimaLlmResponse):
            return None
        parsed = _json_object(result.text)
        if not parsed:
            return None
        vote = str(parsed.get("vote", "")).strip().upper()
        return vote if vote in RAW_VOTES else None

    @staticmethod
    def _run_parsed_with_retries(
        llm_config: LlmConfigProtocol,
        requests: Sequence[MinimaLlmRequest],
        parser: Any,
        *,
        max_attempts: int,
        retry_backoff_seconds: float,
        operation: str,
    ) -> list[Optional[str]]:
        """Retry only failed or malformed items, retaining valid first results.

        Duplicated from judges.ragree_choice_judge._run_parsed_with_retries
        (mature, tested) rather than importing an instance method off a
        foreign judge class; see this module's docstring.
        """
        parsed_results: list[Optional[str]] = [None] * len(requests)
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
                print(f"[RAG26ArenaJudge] {operation} batch failed with {type(error).__name__}; retrying")
                raw_results = []
            next_pending: list[int] = []
            for index, result in zip(pending, raw_results):
                parsed = parser(result)
                if parsed is None:
                    next_pending.append(index)
                else:
                    parsed_results[index] = parsed
            next_pending.extend(pending[len(raw_results):])
            pending = next_pending
        if pending:
            print(f"[RAG26ArenaJudge] {len(pending)} {operation} request(s) unavailable after {max_attempts} attempt(s)")
        return parsed_results

    @staticmethod
    def _arena_values(
        candidate: str,
        run_ids: Sequence[str],
        audit: RankingAudit,
        ballots: Sequence[Ballot],
    ) -> dict[str, float]:
        opponents = [other for other in run_ids if other != candidate]
        if opponents:
            arena_score = 100.0 * sum(
                davidson_win_probability(audit.utilities[candidate], audit.utilities[opponent])
                for opponent in opponents
            ) / len(opponents)
        else:
            arena_score = 50.0

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
            "RAG26_ARENA_SCORE": arena_score,
            "PAIRWISE_WIN_RATE": (wins + 0.5 * ties) / total if total else 0.0,
            "CONDORCET_WINNER": float(audit.condorcet_winner == candidate),
            "RANKING_STABLE": float(audit.kemeny_agrees_on_winner),
            "CYCLE_RATE": audit.cycle_rate,
            "BALLOT_COVERAGE": len(compared_opponents) / len(opponents) if opponents else 1.0,
        }
