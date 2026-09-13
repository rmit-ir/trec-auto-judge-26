"""Contract tests for the search-index adapters (docs/trec-rag-2026-judge-implementation-plan.md
Section 15.0) against dummy local HTTP servers -- no real network dependency, no LLM calls.
Covers both wire contracts actually in use, verified live on 2026-09-12 (see
judges/rag26_arena/README.md): the FastAPI `search_serve` dense contract, and the sparse
Elasticsearch-shaped `POST /api/search` contract (see judges/rag26_arena/search_index.py
docstring for why this replaced an earlier, wrong pyserini-rest-api assumption).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from judges.rag26_arena.search_index import (
    DenseConfig,
    DummySearchClient,
    SearchHit,
    SearchNotConfiguredError,
    SearchServeClient,
    SearchUnavailableError,
    SparseBM25Client,
    SparseConfig,
    dense_client_from_env,
    load_dense_config,
    load_sparse_config,
    sparse_client_from_env,
)


# --- dummy HTTP servers, one per real wire contract -------------------------------------------

class _State:
    def __init__(self):
        self.force_status: int | None = None
        self.doc_batch_seen: list[str] | None = None
        self.search_batch_seen: list[str] | None = None
        self.total_hits: int = 1  # sparse: hits.total.value on a normal (non-empty) response


def _search_serve_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _reply(self, status: int, body: dict):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            if state.force_status:
                return self._reply(state.force_status, {"error": "forced"})
            if self.path == "/health":
                return self._reply(200, {"status": "ok", "n_docs": 3})
            if self.path.startswith("/doc/"):
                docid = self.path.split("/doc/")[1]
                if docid == "missing":
                    return self._reply(404, {"detail": "not found"})
                return self._reply(200, {"docid": docid, "text": f"text-of-{docid}"})
            return self._reply(404, {"error": "no route"})

        def do_POST(self):
            if state.force_status:
                return self._reply(state.force_status, {"error": "forced"})
            body = self._read_json()
            if self.path == "/search":
                return self._reply(200, {
                    "query": body["query"],
                    "hits": [{"docid": "d1", "score": 0.9, "rank": 1, "text": "hit for " + body["query"]}],
                })
            if self.path == "/search/batch":
                state.search_batch_seen = body["queries"]
                return self._reply(200, {
                    "queries": body["queries"],
                    "results": [
                        [{"docid": f"d-{q}", "score": 1.0, "rank": 1, "text": f"hit for {q}"}]
                        for q in body["queries"]
                    ],
                })
            if self.path == "/doc/batch":
                state.doc_batch_seen = body["docids"]
                docs = [{"docid": d, "text": (None if d == "missing" else f"text-of-{d}")}
                        for d in body["docids"]]
                missing = [d for d in body["docids"] if d == "missing"]
                return self._reply(200, {"docs": docs, "found": len(docs) - len(missing),
                                          "missing": missing})
            return self._reply(404, {"error": "no route"})

    return Handler


def _sparse_handler(state: _State):
    """Mimics the real Elasticsearch-shaped sparse service: POST /api/search only,
    GET /actuator/health, response under hits.hits[]._id/_score/_source.contents."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, status: int, body: dict):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if state.force_status:
                return self._reply(state.force_status, {"status": "DOWN"})
            if self.path == "/actuator/health":
                return self._reply(200, {"status": "UP"})
            return self._reply(404, {"timestamp": "...", "status": 404, "error": "Not Found",
                                      "path": self.path})

        def do_POST(self):
            if state.force_status:
                return self._reply(state.force_status, {"error": "forced"})
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path != "/api/search":
                return self._reply(404, {"timestamp": "...", "status": 404, "error": "Not Found",
                                          "path": self.path})
            if "query" not in body:
                return self._reply(400, {"error": "query required"})
            query = body["query"]
            k = body.get("hits", 10)
            if state.total_hits == 0:
                hits = []
            else:
                hits = [{"_index": "default", "_id": f"shard_00001_{i}_p1", "_score": 10.0 - i,
                         "_source": {"contents": f"chunk {i} for {query}", "id": f"shard_00001_{i}_p1"}}
                        for i in range(min(k, state.total_hits))]
            return self._reply(200, {
                "took": 5, "query": f"(contents:{query})^1.0", "timed_out": False,
                "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
                "hits": {"total": {"value": len(hits), "relation": "eq"},
                         "max_score": (hits[0]["_score"] if hits else 0.0), "hits": hits},
            })

    return Handler


@pytest.fixture
def search_serve_server():
    state = _State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _search_serve_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def sparse_server():
    state = _State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _sparse_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        thread.join()


# --- SearchServeClient (dense) ------------------------------------------------------------------

def test_search_serve_search_request_shape_and_hit_normalization(search_serve_server):
    base_url, _ = search_serve_server
    client = SearchServeClient(base_url)
    hits = client.search("hello world", k=5)
    assert hits == [SearchHit(docid="d1", score=0.9, rank=1, text="hit for hello world")]


def test_search_serve_batch_preserves_query_order(search_serve_server):
    base_url, state = search_serve_server
    client = SearchServeClient(base_url)
    results = client.search_batch(["q1", "q2", "q3"])
    assert [r[0].docid for r in results] == ["d-q1", "d-q2", "d-q3"]
    assert state.search_batch_seen == ["q1", "q2", "q3"]


def test_search_serve_single_doc_fetch(search_serve_server):
    base_url, _ = search_serve_server
    client = SearchServeClient(base_url)
    assert client.get_document("abc") == "text-of-abc"
    assert client.get_document("missing") is None


def test_search_serve_batch_doc_partial_miss(search_serve_server):
    base_url, state = search_serve_server
    client = SearchServeClient(base_url)
    docs = client.get_documents(["abc", "missing", "def"])
    assert docs == {"abc": "text-of-abc", "missing": None, "def": "text-of-def"}
    assert state.doc_batch_seen == ["abc", "missing", "def"]


def test_search_serve_health(search_serve_server):
    base_url, _ = search_serve_server
    client = SearchServeClient(base_url)
    assert client.health()["status"] == "ok"


def test_search_serve_5xx_becomes_typed_unavailable_not_empty_results(search_serve_server):
    base_url, state = search_serve_server
    state.force_status = 500
    client = SearchServeClient(base_url)
    with pytest.raises(SearchUnavailableError):
        client.search("hello")
    with pytest.raises(SearchUnavailableError):
        client.health()


def test_search_serve_connection_refused_is_typed_unavailable():
    client = SearchServeClient("http://127.0.0.1:1")  # nothing listens on port 1
    with pytest.raises(SearchUnavailableError):
        client.search("hello")


# --- SparseBM25Client -----------------------------------------------------------------------

def test_sparse_search_request_shape_and_hit_normalization(sparse_server):
    base_url, state = sparse_server
    state.total_hits = 2
    client = SparseBM25Client(base_url)
    hits = client.search("anserini", k=2)
    assert hits == [
        SearchHit(docid="shard_00001_0_p1", score=10.0, rank=1, text="chunk 0 for anserini"),
        SearchHit(docid="shard_00001_1_p1", score=9.0, rank=2, text="chunk 1 for anserini"),
    ]


def test_sparse_search_empty_result_is_a_valid_empty_list_not_an_error(sparse_server):
    base_url, state = sparse_server
    state.total_hits = 0
    client = SparseBM25Client(base_url)
    assert client.search("zzzznonexistentqueryterm") == []


def test_sparse_search_batch_loops_and_preserves_order(sparse_server):
    base_url, _ = sparse_server
    client = SparseBM25Client(base_url)
    results = client.search_batch(["a", "b"])
    assert [r[0].text for r in results] == ["chunk 0 for a", "chunk 0 for b"]


def test_sparse_get_document_not_implemented(sparse_server):
    base_url, _ = sparse_server
    client = SparseBM25Client(base_url)
    with pytest.raises(NotImplementedError):
        client.get_document("shard_00001_0_p1")
    with pytest.raises(NotImplementedError):
        client.get_documents(["shard_00001_0_p1"])


def test_sparse_health_via_actuator(sparse_server):
    base_url, _ = sparse_server
    client = SparseBM25Client(base_url)
    assert client.health()["status"] == "UP"


def test_sparse_health_down_status_is_unavailable(sparse_server):
    base_url, state = sparse_server
    state.force_status = 200  # server still answers 200 but with status: DOWN in a real actuator
    client = SparseBM25Client(base_url)
    with pytest.raises(SearchUnavailableError):
        client.health()


def test_sparse_5xx_becomes_typed_unavailable(sparse_server):
    base_url, state = sparse_server
    state.force_status = 500
    client = SparseBM25Client(base_url)
    with pytest.raises(SearchUnavailableError):
        client.search("anserini")


def test_sparse_malformed_request_status_becomes_typed_unavailable(sparse_server):
    """A missing 'query' triggers a 400 on the real service; our client always sends one, so in
    practice this only fires on a genuine contract mismatch -- still typed, not silently empty."""
    base_url, state = sparse_server
    state.force_status = 400
    client = SparseBM25Client(base_url)
    with pytest.raises(SearchUnavailableError):
        client.search("anserini")


def test_sparse_route_not_mounted_becomes_unavailable_not_empty(sparse_server):
    """Regression: found live against the real sparse service before its /api/search route was
    mounted -- a generic container 404 (path mismatch) must not look like a valid zero-hit
    search. Simulated here by pointing the client at a base URL whose /api/search the dummy
    server's own catch-all 404s (same code path a genuinely unmounted route would hit)."""
    base_url, _ = sparse_server
    broken_client = SparseBM25Client(base_url + "/wrong-prefix")
    with pytest.raises(SearchUnavailableError):
        broken_client.search("anserini")


# --- config / factories -- no hardcoded URL, injected only, one per backend -------------------

def test_load_dense_config_requires_url():
    with pytest.raises(SearchNotConfiguredError):
        load_dense_config(env={})


def test_load_sparse_config_requires_url():
    with pytest.raises(SearchNotConfiguredError):
        load_sparse_config(env={})


def test_load_dense_config_reads_from_mapping_not_os_environ():
    cfg = load_dense_config(env={"RAG26_DENSE_SEARCH_URL": "http://example.invalid:8088"})
    assert cfg == DenseConfig(base_url="http://example.invalid:8088", timeout_seconds=30.0, api_key=None)


def test_load_sparse_config_reads_from_mapping_not_os_environ():
    cfg = load_sparse_config(env={"RAG26_SPARSE_SEARCH_URL": "http://example.invalid:8085"})
    assert cfg == SparseConfig(base_url="http://example.invalid:8085", timeout_seconds=30.0, api_key=None)


def test_dense_client_from_env_builds_search_serve_client():
    client = dense_client_from_env({"RAG26_DENSE_SEARCH_URL": "http://example.invalid:8088"})
    assert isinstance(client, SearchServeClient)


def test_sparse_client_from_env_builds_sparse_bm25_client():
    client = sparse_client_from_env({"RAG26_SPARSE_SEARCH_URL": "http://example.invalid:8085"})
    assert isinstance(client, SparseBM25Client)


# --- DummySearchClient (also a Protocol conformance check) -------------------------------------

def test_dummy_search_client_basic_behavior():
    hit = SearchHit(docid="d1", rank=1, score=1.0, text="hello")
    client = DummySearchClient(hits_by_query={"q": [hit]}, docs={"d1": "hello"})
    assert client.search("q") == [hit]
    assert client.search("unknown") == []
    assert client.search("q", with_text=False)[0].text is None
    assert client.get_document("d1") == "hello"
    assert client.get_document("missing") is None
    assert client.get_documents(["d1", "missing"]) == {"d1": "hello", "missing": None}
    assert client.health()["status"] == "ok"
