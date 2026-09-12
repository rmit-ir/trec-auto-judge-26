#!/usr/bin/env python3
"""
RelevanceJudge: LLM-based judge that scores overall response relevance.

Asks an LLM to grade how relevant a full response is to its topic on a
0-3 scale. Use TinyJudge (judges/tinyjudge/) as a simpler starting point;
this judge additionally shows how to score the whole report text rather
than just the first sentence.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


RELEVANCE_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("RELEVANCE", description="LLM-graded overall relevance of the response to the query (0-3)"),
))


class RelevanceJudge:
    """
    Judges the overall relevance of each response to its topic on a 0-3 scale.

    Implements LeaderboardJudgeProtocol. Configure in workflow.yml:
        judge_class: "judges.testing_simple_leaderboard.relevance_judge:RelevanceJudge"
    """

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        # Standard output path settings (auto-filled by judge_runner)
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        """Judge overall relevance using LLM (batched for efficiency)."""
        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topic_titles.keys())

        sorted_responses = sorted(rag_responses, key=lambda r: r.metadata.run_id)

        requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []  # (run_id, topic_id, request)
        for i, response in enumerate(sorted_responses):
            query = topic_titles.get(response.metadata.topic_id, "")
            report_text = response.get_report_text()
            requests_info.append((
                response.metadata.run_id,
                response.metadata.topic_id,
                MinimaLlmRequest(
                    request_id=f"q{i}",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a relevance evaluator. Grade how relevant the response is "
                                "to the query on a scale of 0-3, where 0 means not relevant at all "
                                "and 3 means highly relevant and directly addresses the query. "
                                "Respond with only the number."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Query: {query}\n\nResponse: {report_text}",
                        },
                    ],
                    temperature=0.0,
                ),
            ))

        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        backend = OpenAIMinimaLlm(full_config)
        llm_results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))

        builder = LeaderboardBuilder(RELEVANCE_SPEC)
        for (run_id, topic_id, _), result in zip(requests_info, llm_results):
            relevance = self._parse_relevance(result)
            builder.add(run_id=run_id, topic_id=topic_id, values={"RELEVANCE": relevance})

        return builder.build(expected_topic_ids=expected_topic_ids, on_missing="fix_aggregate")

    def _parse_relevance(self, result: Any) -> int:
        """Parse LLM response to a relevance score in {0, 1, 2, 3}."""
        if not isinstance(result, MinimaLlmResponse):
            print(f"[RelevanceJudge] LLM error: {result}")
            return 0

        text = result.text.strip()
        for score in ("3", "2", "1", "0"):
            if text.startswith(score):
                return int(score)
        return 0
