"""Configurable ClimbMix-compatible search-index clients (docs/trec-rag-2026-judge-
implementation-plan.md Section 7.1.1-7.1.3, docs/trec-rag-2026-researchrubrics-generation-plan.md
Section 15.0).

Two concrete HTTP adapters, matching the two real backends actually deployed for this project
(verified live on 2026-09-12, see judges/rag26_arena/README.md "Live endpoint status"):

- `SearchServeClient` -- the FastAPI `trec-rag-26/tasks/search_serve` dense/ANN contract
  (`POST /search`, `POST /search/batch`, `GET /doc/{docid}`, `POST /doc/batch`, `GET /health`).
  Verified field-for-field against the live service at :8088.
- `SparseBM25Client` -- the sparse/BM25 index-server contract, `POST /api/search` only, returning
  a native Elasticsearch-shaped body (`hits.hits[]._id/_score/_source.contents`). This is NOT the
  public castorini `pyserini-rest-api` (`/v1/{index}/search`, `candidates[]`) an earlier version
  of this module assumed -- that assumption was wrong for this project's actual sparse service and
  has been replaced. No batch or per-document-fetch endpoint exists on this backend (see class
  docstring); `get_document`/`get_documents` are not supported here on purpose.

Both implement `SearchClient`, so callers never see the wire-format difference. Config is
dual-URL (`RAG26_DENSE_SEARCH_URL`, `RAG26_SPARSE_SEARCH_URL`) because the hybrid retriever
(`hybrid_retrieval.py`) uses both backends at once, not a choice between them.

`requests` is used directly (not added to pyproject.toml): it is already a hard transitive
dependency via `tira` (a core dependency of this project), so it is always present without a new
declared dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Protocol

import requests
from pydantic import BaseModel


class SearchHit(BaseModel):
    docid: str
    rank: int
    score: float
    text: str | None = None


class SearchUnavailableError(Exception):
    """The transport failed or the service reported itself unhealthy -- distinct from a valid
    empty result set. Callers must not treat this as 'no evidence exists' (plan requirement)."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class SearchNotConfiguredError(Exception):
    """No base URL was injected (env var unset). Distinguished from SearchUnavailableError
    because it means "nothing to connect to was configured", not "configured but failed"."""


class SearchClient(Protocol):
    def search(self, query: str, *, k: int = 20, with_text: bool = True,
               **backend_options: object) -> list[SearchHit]: ...

    def search_batch(self, queries: list[str], *, k: int = 20, with_text: bool = True,
                      **backend_options: object) -> list[list[SearchHit]]: ...

    def get_document(self, docid: str) -> str | None: ...

    def get_documents(self, docids: list[str]) -> dict[str, str | None]: ...

    def health(self) -> dict: ...


# ---------------------------------------------------------------------------------------------
# Adapter 1: trec-rag-26/tasks/search_serve (FastAPI, dense/ANN) -- verified live at :8088
# ---------------------------------------------------------------------------------------------

class SearchServeClient:
    """POST /search, POST /search/batch, GET /doc/{docid}, POST /doc/batch, GET /health."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0, api_key: str | None = None,
                 complexity: int = 64, beam_width: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._defaults = {"complexity": complexity, "beam_width": beam_width}

    def _request(self, method: str, path: str, **kwargs: object) -> requests.Response:
        try:
            resp = self._session.request(method, f"{self.base_url}{path}",
                                          timeout=self.timeout_seconds, **kwargs)
        except requests.RequestException as exc:
            raise SearchUnavailableError(f"{method} {path}: transport error: {exc}") from exc
        if resp.status_code >= 500 or resp.status_code == 503:
            raise SearchUnavailableError(f"{method} {path}: server error {resp.status_code}",
                                          status_code=resp.status_code)
        if resp.status_code >= 400 and resp.status_code != 404:
            raise SearchUnavailableError(f"{method} {path}: client error {resp.status_code}: "
                                          f"{resp.text[:200]}", status_code=resp.status_code)
        return resp

    def search(self, query: str, *, k: int = 20, with_text: bool = True,
               **backend_options: object) -> list[SearchHit]:
        payload = {"query": query, "k": k, "with_text": with_text, **self._defaults, **backend_options}
        resp = self._request("POST", "/search", json=payload)
        return [SearchHit(**h) for h in resp.json()["hits"]]

    def search_batch(self, queries: list[str], *, k: int = 20, with_text: bool = True,
                      **backend_options: object) -> list[list[SearchHit]]:
        payload = {"queries": queries, "k": k, "with_text": with_text, **self._defaults, **backend_options}
        resp = self._request("POST", "/search/batch", json=payload)
        return [[SearchHit(**h) for h in hits] for hits in resp.json()["results"]]

    def get_document(self, docid: str) -> str | None:
        resp = self._request("GET", f"/doc/{docid}")
        if resp.status_code == 404:
            return None
        return resp.json()["text"]

    def get_documents(self, docids: list[str]) -> dict[str, str | None]:
        resp = self._request("POST", "/doc/batch", json={"docids": docids})
        return {d["docid"]: d["text"] for d in resp.json()["docs"]}

    def health(self) -> dict:
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise SearchUnavailableError(f"GET /health: transport error: {exc}") from exc
        if resp.status_code != 200:
            raise SearchUnavailableError(f"GET /health: status {resp.status_code}",
                                          status_code=resp.status_code)
        return resp.json()


# ---------------------------------------------------------------------------------------------
# Adapter 2: sparse BM25 index-server (Elasticsearch-shaped) -- verified live at :8085
# ---------------------------------------------------------------------------------------------

class SparseBM25Client:
    """POST /api/search only -- a native Elasticsearch/OpenSearch-shaped response:
    `{"hits": {"total": {...}, "hits": [{"_id", "_score", "_source": {"contents", "id"}}]}}`.
    Verified live: a zero-hit query returns 200 with `hits.hits: []` (a genuine empty result, not
    an error); a malformed request (missing `query`) returns 400.

    No batch search endpoint -- `search_batch` loops sequentially. No per-document-fetch endpoint
    at all on this backend (unlike SearchServeClient's `/doc/{docid}`) -- each search hit already
    carries its full chunk text inline via `_source.contents`, and `get_document`/`get_documents`
    raise `NotImplementedError` rather than pretending to support something this service does not
    expose. Resolve an arbitrary docid via `SearchServeClient.get_document` instead (both indices
    share the same corpus/chunking scheme) -- that's what `hybrid_retrieval.HybridResearchRetriever`
    does.

    BM25 scores are unbounded Lucene relevance scores, not comparable to dense's cosine/MIPS
    scores -- never mix them without going through rank-based fusion (see hybrid_retrieval.py).
    This backend uses its own bag-of-words query analyzer (effectively OR-style lexical
    retrieval): no +required/-excluded terms, boolean operators, phrase, or proximity syntax."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def search(self, query: str, *, k: int = 20, with_text: bool = True,
               **backend_options: object) -> list[SearchHit]:
        try:
            resp = self._session.post(f"{self.base_url}/api/search",
                                       json={"query": query, "hits": k},
                                       timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise SearchUnavailableError(f"POST /api/search: transport error: {exc}") from exc
        if resp.status_code != 200:
            raise SearchUnavailableError(f"POST /api/search: status {resp.status_code}: "
                                          f"{resp.text[:200]}", status_code=resp.status_code)
        raw_hits = resp.json()["hits"]["hits"]
        return [
            SearchHit(docid=h["_id"], score=h["_score"], rank=i + 1,
                      text=(h["_source"].get("contents") if with_text else None))
            for i, h in enumerate(raw_hits)
        ]

    def search_batch(self, queries: list[str], *, k: int = 20, with_text: bool = True,
                      **backend_options: object) -> list[list[SearchHit]]:
        return [self.search(q, k=k, with_text=with_text, **backend_options) for q in queries]

    def get_document(self, docid: str) -> str | None:
        raise NotImplementedError(
            "SparseBM25Client exposes no per-document-fetch endpoint; resolve docids via "
            "SearchServeClient.get_document (both backends share the same corpus/chunking)."
        )

    def get_documents(self, docids: list[str]) -> dict[str, str | None]:
        raise NotImplementedError(
            "SparseBM25Client exposes no per-document-fetch endpoint; resolve docids via "
            "SearchServeClient.get_documents (both backends share the same corpus/chunking)."
        )

    def health(self) -> dict:
        """No /health on this service's own API; it's a Spring Boot app so /actuator/health is
        the closest equivalent (verified live: {"status": "UP"})."""
        try:
            resp = self._session.get(f"{self.base_url}/actuator/health", timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise SearchUnavailableError(f"GET /actuator/health: transport error: {exc}") from exc
        if resp.status_code != 200:
            raise SearchUnavailableError(f"GET /actuator/health: status {resp.status_code}",
                                          status_code=resp.status_code)
        body = resp.json()
        if body.get("status") != "UP":
            raise SearchUnavailableError(f"GET /actuator/health: status field {body.get('status')!r}")
        return body


# ---------------------------------------------------------------------------------------------
# Adapter 3: in-memory dummy, for tests and the prompt-only ablation/fallback
# ---------------------------------------------------------------------------------------------

class DummySearchClient:
    """Deterministic in-memory adapter. `hits_by_query` and `docs` are keyed exactly as given --
    no fuzzy matching. Always healthy."""

    def __init__(self, hits_by_query: Mapping[str, list[SearchHit]] | None = None,
                 docs: Mapping[str, str | None] | None = None):
        self._hits_by_query = dict(hits_by_query or {})
        self._docs = dict(docs or {})

    def search(self, query: str, *, k: int = 20, with_text: bool = True,
               **backend_options: object) -> list[SearchHit]:
        hits = self._hits_by_query.get(query, [])[:k]
        if not with_text:
            hits = [h.model_copy(update={"text": None}) for h in hits]
        return hits

    def search_batch(self, queries: list[str], *, k: int = 20, with_text: bool = True,
                      **backend_options: object) -> list[list[SearchHit]]:
        return [self.search(q, k=k, with_text=with_text) for q in queries]

    def get_document(self, docid: str) -> str | None:
        return self._docs.get(docid)

    def get_documents(self, docids: list[str]) -> dict[str, str | None]:
        return {d: self._docs.get(d) for d in docids}

    def health(self) -> dict:
        return {"status": "ok", "backend": "dummy"}


# ---------------------------------------------------------------------------------------------
# Config-driven factories -- the only place environment variables are read
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DenseConfig:
    base_url: str
    timeout_seconds: float
    api_key: str | None


@dataclass(frozen=True)
class SparseConfig:
    base_url: str
    timeout_seconds: float
    api_key: str | None


def _require(env: Mapping[str, str], var: str) -> str:
    value = env.get(var)
    if not value:
        raise SearchNotConfiguredError(
            f"{var} is not set -- no search index is configured. Never hardcode this URL; "
            f"set {var} to enable evidence-grounded rubric generation and document resolution."
        )
    return value


def load_dense_config(env: Mapping[str, str] | None = None) -> DenseConfig:
    env = env if env is not None else os.environ
    return DenseConfig(
        base_url=_require(env, "RAG26_DENSE_SEARCH_URL"),
        timeout_seconds=float(env.get("RAG26_SEARCH_TIMEOUT_SECONDS", "30")),
        api_key=env.get("RAG26_DENSE_SEARCH_API_KEY"),
    )


def load_sparse_config(env: Mapping[str, str] | None = None) -> SparseConfig:
    env = env if env is not None else os.environ
    return SparseConfig(
        base_url=_require(env, "RAG26_SPARSE_SEARCH_URL"),
        timeout_seconds=float(env.get("RAG26_SEARCH_TIMEOUT_SECONDS", "30")),
        api_key=env.get("RAG26_SPARSE_SEARCH_API_KEY"),
    )


def dense_client_from_env(env: Mapping[str, str] | None = None) -> SearchServeClient:
    config = load_dense_config(env)
    return SearchServeClient(config.base_url, timeout_seconds=config.timeout_seconds,
                              api_key=config.api_key)


def sparse_client_from_env(env: Mapping[str, str] | None = None) -> SparseBM25Client:
    config = load_sparse_config(env)
    return SparseBM25Client(config.base_url, timeout_seconds=config.timeout_seconds,
                             api_key=config.api_key)
