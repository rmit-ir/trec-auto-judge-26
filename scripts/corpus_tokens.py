#!/usr/bin/env python3
"""Tokenize a full dataset's runs+topics to get a free, offline token-count floor.

Reads every run and topic in a dataset (same as the judge does) but only ever
emits AGGREGATE token counts -- dataset-level totals, and per-topic sums
combined across all runs. Per the data-handling policy in CLAUDE.md, per-run
or per-team token counts would be a length/style fingerprint of a specific
run's origin, so this script never computes or prints one. It never prints
response text, document text, file paths, or file-loading progress either.

Usage:
    python scripts/corpus_tokens.py --dataset rag26-generation --model gpt-5.6-luna
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import azure_cost  # noqa: E402
from run_all_datasets import load_datasets  # noqa: E402

from autojudge_base.report import load_report  # noqa: E402
from autojudge_base.request import load_requests_from_file  # noqa: E402


@dataclass
class CorpusCounts:
    dataset: str
    topics: int = 0
    reports: int = 0
    skipped_files: int = 0
    topic_tokens: int = 0
    report_text_tokens: int = 0
    cited_sentences: int = 0
    cited_sentences_resolved: int = 0  # ~= ragree verify-call count
    cited_sentence_tokens: int = 0
    cited_evidence_tokens: int = 0  # summed per resolved citation link
    encoding: str = ""
    encoding_source: str = ""
    max_doc_chars: int = 0
    max_answer_chars: int = 0
    per_topic_tokens: Dict[str, int] = field(default_factory=dict)  # summed across ALL runs


def resolve_encoding(model_name: str) -> tuple[str, str]:
    """tiktoken's own model map first, then this repo's price-table column, then a default.

    Never guesses silently: the source is always returned so callers can flag an
    approximation (see main()'s warning when the source isn't the model map).
    """
    try:
        return tiktoken.encoding_for_model(model_name).name, "tiktoken-model-map"
    except KeyError:
        pass
    price = azure_cost.load_model_prices().get(model_name)
    if price and price.tokenizer_encoding:
        return price.tokenizer_encoding, "price-table"
    return "o200k_base", "default-o200k_base"


def _run_files(responses_dir: Path) -> list[Path]:
    return sorted(p for p in responses_dir.resolve().rglob("*") if p.is_file())


def count_corpus(
    responses_dir: Path,
    topics_path: Path,
    enc: "tiktoken.Encoding",
    *,
    dataset_name: str,
    max_doc_chars: int = 1500,
    max_answer_chars: int = 0,
) -> CorpusCounts:
    counts = CorpusCounts(
        dataset=dataset_name,
        encoding=enc.name,
        max_doc_chars=max_doc_chars,
        max_answer_chars=max_answer_chars,
    )

    for request in load_requests_from_file(topics_path):
        counts.topics += 1
        topic_text = request.title + " " + (request.background or "")
        counts.topic_tokens += len(enc.encode_ordinary(topic_text))

    for run_file in _run_files(responses_dir):
        try:
            reports = load_report(run_file)
        except Exception:
            counts.skipped_files += 1
            continue

        for report in reports:
            counts.reports += 1
            topic_id = report.metadata.topic_id or "unknown"
            report_total = 0

            answer_text = report.get_report_text()
            if max_answer_chars > 0:
                answer_text = answer_text[:max_answer_chars]
            answer_tokens = len(enc.encode_ordinary(answer_text))
            counts.report_text_tokens += answer_tokens
            report_total += answer_tokens

            docs = report.documents or {}
            for sentence in report.get_sentences_with_citations():
                # ragree_judge skips blank sentences before counting/queuing a verify call;
                # mirror that so cited_sentences_resolved tracks the real call count.
                if not (sentence.text or "").strip():
                    continue
                if not sentence.citations:
                    continue
                counts.cited_sentences += 1
                sentence_tokens = len(enc.encode_ordinary(sentence.text))
                counts.cited_sentence_tokens += sentence_tokens
                report_total += sentence_tokens

                resolved = [docs[d] for d in sentence.citations if d in docs]
                if resolved:
                    counts.cited_sentences_resolved += 1
                    for doc in resolved:
                        doc_text = doc.get_text()
                        if max_doc_chars > 0:
                            doc_text = doc_text[:max_doc_chars]
                        evidence_tokens = len(enc.encode_ordinary(doc_text))
                        counts.cited_evidence_tokens += evidence_tokens
                        report_total += evidence_tokens

            # ponytail: pooled unconditionally across however many reports touch this topic.
            # Safe while every fetched dataset has 25+ runs per topic (verified); would become
            # a de-facto per-run fingerprint for a topic with very few contributing runs -- add
            # a minimum-report-count guard (omit + count the omission) before pointing this at
            # a sparser release.
            counts.per_topic_tokens[topic_id] = counts.per_topic_tokens.get(topic_id, 0) + report_total

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--datasets", default="datasets.yml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-doc-chars", type=int, default=1500)
    parser.add_argument("--max-answer-chars", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    datasets = {d.name: d for d in load_datasets(Path(args.datasets))}
    if args.dataset not in datasets:
        print(f"Unknown or unfetched dataset: {args.dataset}", file=sys.stderr)
        return 1
    dataset = datasets[args.dataset]

    encoding_name, source = resolve_encoding(args.model)
    if source != "tiktoken-model-map":
        print(
            f"warning: {args.model!r} not in tiktoken's model map; "
            f"using {encoding_name} ({source})",
            file=sys.stderr,
        )
    enc = tiktoken.get_encoding(encoding_name)
    counts = count_corpus(
        Path(dataset.responses),
        Path(dataset.topics),
        enc,
        dataset_name=args.dataset,
        max_doc_chars=args.max_doc_chars,
        max_answer_chars=args.max_answer_chars,
    )
    counts.encoding_source = source

    out_path = Path(args.out) if args.out else Path("output/token-counts") / f"{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(counts), indent=2))

    summary = asdict(counts)
    del summary["per_topic_tokens"]
    print(json.dumps(summary, indent=2))
    print(f"Per-topic totals ({counts.topics} topics) written to {out_path}", file=sys.stderr)

    if counts.skipped_files:
        print(
            f"warning: {counts.skipped_files} run file(s) failed to parse and were skipped -- "
            "the floor above is UNDERCOUNTED, not a complete corpus total",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
