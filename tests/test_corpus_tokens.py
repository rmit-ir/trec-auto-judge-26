import json

import tiktoken

from scripts.corpus_tokens import count_corpus, resolve_encoding


def test_resolve_encoding_known_model():
    encoding, source = resolve_encoding("gpt-5.6-luna")
    assert encoding == "o200k_base"
    assert source in ("tiktoken-model-map", "price-table")


def test_resolve_encoding_unknown_model_falls_back():
    encoding, source = resolve_encoding("totally-unknown-model-xyz")
    assert encoding == "o200k_base"
    assert source == "default-o200k_base"


def _write_synthetic_dataset(tmp_path):
    responses_dir = tmp_path / "runs"
    responses_dir.mkdir()
    reports = [
        {
            "metadata": {"team_id": "t1", "run_id": "r1", "topic_id": "1"},
            "responses": [
                {"text": "Cited sentence one.", "citations": ["doc1"]},
                {"text": "Uncited sentence.", "citations": None},
            ],
            "documents": {
                "doc1": {"id": "doc1", "text": "Evidence text " * 20},
            },
        },
        {
            "metadata": {"team_id": "t2", "run_id": "r2", "topic_id": "1"},
            "responses": [
                {"text": "Another cited sentence.", "citations": ["missing_doc"]},
            ],
            "documents": {},
        },
    ]
    run_file = responses_dir / "run.jsonl"
    run_file.write_text("\n".join(json.dumps(r) for r in reports))

    topics_path = tmp_path / "topics.jsonl"
    topics_path.write_text(json.dumps({"request_id": "1", "title": "topic title", "background": "topic background"}))

    return responses_dir, topics_path


def test_count_corpus_aggregates_and_resolves_citations(tmp_path):
    responses_dir, topics_path = _write_synthetic_dataset(tmp_path)
    enc = tiktoken.get_encoding("o200k_base")

    counts = count_corpus(responses_dir, topics_path, enc, dataset_name="synthetic", max_doc_chars=1500)

    assert counts.topics == 1
    assert counts.reports == 2
    assert counts.cited_sentences == 2  # one resolved, one with a missing doc_id
    assert counts.cited_sentences_resolved == 1
    assert counts.cited_evidence_tokens > 0
    assert counts.per_topic_tokens.keys() == {"1"}


def test_count_corpus_max_doc_chars_shrinks_evidence_tokens(tmp_path):
    responses_dir, topics_path = _write_synthetic_dataset(tmp_path)
    enc = tiktoken.get_encoding("o200k_base")

    full = count_corpus(responses_dir, topics_path, enc, dataset_name="synthetic", max_doc_chars=1500)
    truncated = count_corpus(responses_dir, topics_path, enc, dataset_name="synthetic", max_doc_chars=10)

    assert truncated.cited_evidence_tokens < full.cited_evidence_tokens


def test_count_corpus_max_doc_chars_zero_is_unlimited(tmp_path):
    responses_dir, topics_path = _write_synthetic_dataset(tmp_path)
    enc = tiktoken.get_encoding("o200k_base")

    unlimited = count_corpus(responses_dir, topics_path, enc, dataset_name="synthetic", max_doc_chars=0)
    capped_high = count_corpus(responses_dir, topics_path, enc, dataset_name="synthetic", max_doc_chars=10_000)

    assert unlimited.cited_evidence_tokens == capped_high.cited_evidence_tokens
    assert unlimited.cited_evidence_tokens > 0


def test_count_corpus_skips_unparseable_files_without_dropping_the_rest(tmp_path):
    responses_dir, topics_path = _write_synthetic_dataset(tmp_path)
    (responses_dir / "corrupt.jsonl").write_text("not valid json at all")
    enc = tiktoken.get_encoding("o200k_base")

    counts = count_corpus(responses_dir, topics_path, enc, dataset_name="synthetic", max_doc_chars=1500)

    assert counts.skipped_files == 1
    assert counts.reports == 2  # the two good reports are still counted
