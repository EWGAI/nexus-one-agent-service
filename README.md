# NexusOne Agent Service

Production-ready RAG agent service for an ERP: **FastAPI + LangChain + LangGraph**, with a
**pluggable vector store** (Chroma / FAISS / Pinecone), **pluggable LLM & embedding providers**,
**JWT authentication on every request**, and **domain-separated knowledge bases** so a query
against `finance` can never return `hr` content.

---

## Table of contents

- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Switching vector stores with one env change](#switching-vector-stores-with-one-env-change)
- [Domain separated knowledge bases](#domain-separated-knowledge-bases)
- [Multi-tenancy](#multi-tenancy)
- [Authentication & authorization](#authentication--authorization)
- [API reference](#api-reference)
- [Example curl calls](#example-curl-calls)
- [Docker & docker-compose](#docker--docker-compose)
- [Development workflow](#development-workflow)
- [CI pipeline & branch protection](#ci-pipeline--branch-protection)
- [Extending: a new store adapter](#extending-a-new-store-adapter)
- [Extending: a new tool or graph node](#extending-a-new-tool-or-graph-node)
- [Project layout](#project-layout)

---

## Quick start

```bash
# 1. environment
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"          # chroma + faiss + test/lint tooling

# 2. configuration
cp .env.example .env
python - <<'PY' >> .env
import secrets; print(f"JWT_SECRET_KEY={secrets.token_urlsafe(48)}")
PY

# 3. run
make dev                            # uvicorn with reload on :8000
open http://localhost:8000/docs     # click "Authorize" and paste a token
```

Mint a development token (never exposed as an API endpoint):

```bash
python scripts/mint_token.py --sub dev@example.com --scopes "kb:*:read kb:*:write kb:admin agent:chat"
```

Run everything the CI runs, locally:

```bash
make ci
```

### Offline / zero-cost mode

Set `LLM_PROVIDER=fake` and `EMBEDDING_PROVIDER=fake` to run the whole service with no API keys.
The fake embedding model is a deterministic hashing-trick encoder (lexically meaningful, so
ranking assertions are real) and the fake chat model echoes the retrieved context. This is what
the test suite and the docker smoke test use.

---

## Configuration

Everything is environment driven. See [`.env.example`](.env.example) for the annotated list; the
table below is the summary.

| Variable | Applies to | Default | Purpose |
| --- | --- | --- | --- |
| `VECTOR_STORE` | all | `chroma` | `chroma` \| `faiss` \| `pinecone` |
| `LLM_PROVIDER` | all | `openai` | `openai` \| `anthropic` \| `fake` |
| `EMBEDDING_PROVIDER` | all | `openai` | `openai` \| `huggingface` \| `fake` |
| `KB_DOMAINS` | all | `hr,finance,engineering,it,company` | Registered business domains |
| `TENANT_MODE` | all | `off` | `off` \| `metadata` \| `prefix` |
| `JWT_SECRET_KEY` | all | *(required, ≥32 chars)* | Shared HMAC secret |
| `CHROMA_PERSIST_DIR` | chroma | `./data/chroma` | Embedded persistent client path |
| `CHROMA_HOST` / `CHROMA_PORT` | chroma | – | Use a Chroma server instead |
| `FAISS_INDEX_PATH` | faiss | `./data/faiss` | Root of the per-domain index folders |
| `PINECONE_API_KEY` | pinecone | – | **Required** |
| `PINECONE_INDEX` | pinecone | – | Required for `PINECONE_STRATEGY=namespace` |
| `PINECONE_STRATEGY` | pinecone | `namespace` | `namespace` (shared index) or `index` (one per domain) |
| `PINECONE_AUTO_CREATE` | pinecone | `false` | Auto-create serverless indexes |
| `ENABLE_DOCS` | all | `true` unless `ENVIRONMENT=production` | Serve `/docs` + `/redoc` |

The factory **fails fast at startup** with an actionable message when a store is selected but its
required variables are missing, when `JWT_SECRET_KEY` is absent or shorter than 32 characters, or
when the embedding dimension does not match an existing index.

---

## Switching vector stores with one env change

No code changes. All three adapters implement the same
[`VectorStoreAdapter`](agent_service/kb/base.py) interface.

```bash
# Local, persistent, zero setup
VECTOR_STORE=chroma CHROMA_PERSIST_DIR=./data/chroma

# Local, file based, save/load to disk
VECTOR_STORE=faiss FAISS_INDEX_PATH=./data/faiss

# Managed
VECTOR_STORE=pinecone PINECONE_API_KEY=pc-... PINECONE_INDEX=nexusone PINECONE_AUTO_CREATE=true
```

Install the matching extra: `pip install -e ".[chroma]"`, `".[faiss]"`, `".[pinecone]"`, or
`".[all]"`.

| Concern | Chroma | FAISS | Pinecone |
| --- | --- | --- | --- |
| One domain = one… | collection `kb_<domain>` | index directory `FAISS_INDEX_PATH/kb_<domain>/` | namespace (default) or index |
| Persistence | `PersistentClient` dir or HTTP server | `index.faiss` + `records.json` rewritten after every mutation, loaded on startup | managed |
| Metadata filter | native `where` | compiled python post-filter | native `filter` |
| Atomic reindex | create staging collection → `modify(name=…)` | write staging dir → `Path.replace` | copy staging namespace → drop |

The FAISS adapter is built directly on `faiss` and stores chunk text/metadata as
JSON, so loading an index never unpickles arbitrary objects.

---

## Domain separated knowledge bases

Each business domain is an **isolated index**. Routes are domain scoped
(`/kb/{domain}/...`) and unknown domains are rejected with `422`.

| Domain | ERP modules it backs | Typical content |
| --- | --- | --- |
| `hr` | Payroll, leave, attendance, employee policies | Handbooks, leave policy, payroll SOPs, org charts |
| `finance` | GL, AP, AR, tax, compliance | Chart of accounts, invoice rules, tax circulars, audit notes |
| `engineering` | Product & platform | Specs, ADRs, API contracts, design docs |
| `it` | Helpdesk, infrastructure | Runbooks, incident playbooks, access request guides |
| `company` | Cross-cutting | Holidays, code of conduct, announcements, general FAQ |

`company` is the **fallback domain**: it stays eligible for general questions whenever the caller
can read it (`KB_FALLBACK_DOMAIN`).

**Adding a domain is config only** — append it to `KB_DOMAINS` and restart. The collection /
namespace / directory is created lazily on first ingest.

### New domain, or just a metadata tag?

Create a **new domain** when:

- content has a different *access* boundary (different people should be allowed to read it);
- you want hard blast-radius isolation (dropping or re-indexing one must not touch the others);
- the vocabulary is so different that mixing it degrades retrieval quality.

Use a **metadata tag** (`module`, `fiscal_year`, `region`, …) when:

- everybody who can read the domain may read the subset;
- you only need to *narrow* a search (`?filter={"module":"payroll"}`);
- the partition changes often (per year, per project).

Rule of thumb: **domains are for authorization and isolation, tags are for precision.**

### How retrieval works across domains

1. `route_domain` — if the caller passed `domains`, they are used verbatim. Otherwise the LLM
   picks from the domains the token may read; the decision and its reason are logged and returned
   in `routing_reason`. The fallback domain is appended when readable.
2. `retrieve` — the query fans out to every selected domain in parallel.
3. Results are merged with **reciprocal rank fusion** (`score = Σ 1/(k + rank)`), which is robust
   when scores from different collections are not directly comparable.
4. Every source in the response is tagged with the `domain` it came from.

---

## Multi-tenancy

`TENANT_MODE` controls how tenants are isolated. **The tenant always comes from the JWT
`tenant_id` claim, never from the request body.**

| Mode | Physical layout | Filtering |
| --- | --- | --- |
| `off` | `kb_<domain>` | none |
| `metadata` | `kb_<domain>` | every chunk stores `tenant_id`; every search adds a `tenant_id` predicate |
| `prefix` | `kb_<tenant>_<domain>` | physical separation, no predicate needed |

In `metadata` and `prefix` mode a token without `tenant_id` (and without `DEFAULT_TENANT_ID`) is
rejected with `403 missing_tenant`.

---

## Authentication & authorization

- Every endpoint **except** `/health`, `/ready`, `/docs`, `/redoc` and `/openapi.json` requires
  `Authorization: Bearer <jwt>`.
- Tokens are signed by the calling backend with the shared secret and verified here with
  **HS256** (`JWT_ALGORITHM`). `alg: none` and any algorithm other than the configured one are
  rejected. `exp`, `nbf`, `iat` are validated, plus `iss`/`aud` when `JWT_ISSUER` / `JWT_AUDIENCE`
  are set. `JWT_LEEWAY_SECONDS` absorbs clock skew.
- Verification happens in a **router level dependency**
  (`dependencies=[Depends(verify_jwt)]` on `/kb` and `/agent`), so a new route physically cannot
  be added unprotected.
- The verified [`TokenClaims`](agent_service/core/security.py) are attached to `request.state.claims`.
- Request logs include the token `sub`; the raw token is never logged.

### Scopes

| Scope | Grants |
| --- | --- |
| `kb:<domain>:read` | `GET /kb/<domain>/search`, retrieval for that domain |
| `kb:<domain>:write` | `POST /kb/<domain>/ingest/*`, `DELETE /kb/<domain>/documents` |
| `kb:*:read` / `kb:*:write` | the same for every domain |
| `kb:admin` | `/kb/domains`, `/kb/<domain>/stats`, `/kb/<domain>/reindex`, `DELETE /kb/<domain>` |
| `agent:chat` | `POST /agent/chat`, `POST /agent/chat/stream` |

A scope mismatch returns `403` with the same body whether or not the domain holds data, so the
response never leaks the existence of content.

### Error envelope

```json
{
  "error": {
    "code": "forbidden",
    "message": "Insufficient scope for domain 'hr'",
    "details": { "required_scope": "kb:hr:write" },
    "request_id": "3f1c1a0e9f4b4c0a8f2a"
  }
}
```

### Other hardening

- `POST /kb/{domain}/ingest/urls` refuses URLs that resolve to private, loopback,
  link-local, reserved or multicast addresses (SSRF guard), and only accepts
  `http`/`https`.
- Uploads are capped by `MAX_UPLOAD_BYTES` and restricted to a known extension
  allow-list.
- FAISS indexes are persisted as `index.faiss` + JSON, never pickle.
- Background job records are only readable by the subject that created them (or
  `kb:admin`); a foreign job id returns `404`, not `403`.
- `pip-audit` runs in CI. Four ChromaDB advisories (`PYSEC-2026-311`, `-3813`,
  `-3814`, `-3815`) have **no fixed release** and affect the ChromaDB *server*
  (auth bypass / RBAC / code injection), not the embedded client this service
  uses by default. They are ignored explicitly in the Makefile and `ci.yml` with
  that justification - **re-review on every Dependabot bump**, and if you run
  Chroma with `--profile server`, keep it on a private network and never expose
  port 8000 publicly.

---

## API reference

Interactive docs at `/docs` (Swagger UI, with a working **Authorize** button) and `/redoc`.
The committed [`openapi.json`](openapi.json) is regenerated with `make openapi` and CI fails if it
drifts.

| Method | Path | Scope | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | – | Liveness |
| `GET` | `/ready` | – | Readiness (checks the store) |
| `GET` | `/kb/domains` | `kb:admin` | Configured vs. materialised domains |
| `POST` | `/kb/{domain}/ingest/files` | `kb:<d>:write` | Multipart upload, many files |
| `POST` | `/kb/{domain}/ingest/text` | `kb:<d>:write` | Raw text + tags |
| `POST` | `/kb/{domain}/ingest/urls` | `kb:<d>:write` | List of URLs |
| `GET` | `/kb/{domain}/search` | `kb:<d>:read` | `?q=&k=&filter=` → ranked chunks |
| `DELETE` | `/kb/{domain}/documents` | `kb:<d>:write` | By ids or metadata filter |
| `GET` | `/kb/{domain}/stats` | `kb:admin` | Store, counts, model, dimension |
| `POST` | `/kb/{domain}/reindex` | `kb:admin` | Re-embed + atomic swap |
| `DELETE` | `/kb/{domain}` | `kb:admin` | Drop the whole domain |
| `GET` | `/kb/jobs/{id}` | authenticated | Background ingestion job status |
| `POST` | `/agent/chat` | `agent:chat` | Answer + sources + `thread_id` |
| `POST` | `/agent/chat/stream` | `agent:chat` | SSE tokens and tool events |

Ingestion above `BACKGROUND_INGEST_THRESHOLD` sources returns `{"mode":"async","job_id":"..."}`.

---

## Example curl calls

```bash
export TOKEN=$(python scripts/mint_token.py --quiet \
  --sub alice@example.com \
  --scopes "kb:*:read kb:*:write kb:admin agent:chat")
export BASE=http://localhost:8000
```

Ingest text into `hr`:

```bash
curl -sS -X POST "$BASE/kb/hr/ingest/text" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
        "text": "Employees accrue 1.75 days of paid leave per completed month of service.",
        "source": "hr-leave-policy-2026",
        "tags": {"module": "leave", "fiscal_year": 2026}
      }' | jq
```

Upload files:

```bash
curl -sS -X POST "$BASE/kb/finance/ingest/files" \
  -H "Authorization: Bearer $TOKEN" \
  -F 'files=@./docs/ap-policy.pdf' \
  -F 'files=@./docs/tax-2026.docx' \
  -F 'tags={"module":"ap","fiscal_year":2026}' | jq
```

Search with a metadata filter:

```bash
curl -sS -G "$BASE/kb/hr/search" \
  -H "Authorization: Bearer $TOKEN" \
  --data-urlencode 'q=how much leave do I accrue' \
  --data-urlencode 'k=5' \
  --data-urlencode 'filter={"module":"leave"}' | jq
```

Chat (router picks the domains):

```bash
curl -sS -X POST "$BASE/agent/chat" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message": "How many leave days do I accrue each month?"}' | jq
```

Stream:

```bash
curl -sS -N -X POST "$BASE/agent/chat/stream" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message": "Summarise the AP approval limits", "domains": ["finance"]}'
```

Delete every chunk of one source, then inspect the domain:

```bash
curl -sS -X DELETE "$BASE/kb/hr/documents" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"filter": {"source": "hr-leave-policy-2026"}}' | jq

curl -sS "$BASE/kb/hr/stats" -H "Authorization: Bearer $TOKEN" | jq
```

---

## Docker & docker-compose

```bash
docker compose up --build            # agent-service on :8000, embedded Chroma
docker compose --profile server up   # agent-service + a standalone Chroma server
```

The `server` profile starts `chromadb/chroma` and points the service at it through
`CHROMA_HOST=chroma`. Nothing else changes.

End-to-end smoke test against a running container:

```bash
make smoke            # mint token -> ingest -> search -> chat
```

---

## Development workflow

| Target | What it does |
| --- | --- |
| `make install` | create `.venv` and install `.[dev]` |
| `make dev` | uvicorn with reload |
| `make lint` | `ruff check` (includes import sorting) |
| `make format` | `ruff format` (`make format-check` for CI) |
| `make typecheck` | `mypy --strict agent_service/` |
| `make spell` | `codespell` over source, docs and README |
| `make test` | `pytest` |
| `make cov` | `pytest --cov --cov-fail-under=85` |
| `make security` | `bandit` + `pip-audit` |
| `make openapi` | regenerate `openapi.json` |
| `make openapi-check` | fail if `openapi.json` drifted |
| `make smoke` | ingest → search → chat against a running service |
| `make ci` | everything above, in CI order |

Install the git hooks once: `pre-commit install`.

---

## CI pipeline & branch protection

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request with a
**Python 3.11 / 3.12 matrix** and a cached `uv` environment:

| Job | Gate |
| --- | --- |
| `lint` | `ruff check` — fails on any finding |
| `format` | `ruff format --check` |
| `typecheck` | `mypy --strict agent_service/` |
| `spelling` | `codespell` (allow-list in [`.codespellrc`](.codespellrc)) |
| `tests` | `pytest --cov --cov-fail-under=85`, coverage XML artifact, Codecov when `CODECOV_TOKEN` is set. FAISS and Chroma run with the fake embedding model; Pinecone tests are marked `integration` and skipped unless `PINECONE_API_KEY` is present |
| `security` | `bandit`, `pip-audit`, `gitleaks` secret scan |
| `openapi-drift` | regenerates the spec and diffs it against the committed `openapi.json` |
| `docker` | builds the image, runs the container, hits `/health`, then mints a token and runs ingest → search → chat |
| `dependency-review` | pull requests only |

**Branch protection (configure once in repository settings → Branches → `main`):**

- Require a pull request before merging, with at least one approval.
- Require status checks to pass: `lint`, `format`, `typecheck`, `spelling`,
  `tests (3.11)`, `tests (3.12)`, `security`, `openapi-drift`, `docker`.
- Require branches to be up to date before merging.
- Require conversation resolution; do not allow force pushes or deletions.

Dependabot ([`.github/dependabot.yml`](.github/dependabot.yml)) keeps `pip` and
`github-actions` dependencies current.

---

## Extending: a new store adapter

1. Create `agent_service/kb/qdrant_store.py` and subclass
   [`VectorStoreAdapter`](agent_service/kb/base.py), implementing `add_documents`,
   `existing_ids`, `similarity_search`, `delete`, `iter_documents`, `list_domains`,
   `domain_stats`, `drop_domain` and `swap_domain` (plus `persist` / `initialize` if relevant).
   Use `self.physical_name(domain, tenant_id)` for naming and
   `self.effective_filter(filters, tenant_id)` for tenancy — that is all domain isolation and
   multi-tenancy needs.
2. Register it in [`agent_service/kb/factory.py`](agent_service/kb/factory.py):

   ```python
   BUILTIN_STORES["qdrant"] = ("agent_service.kb.qdrant_store", "QdrantStore", "qdrant")
   ```

   or, from an out-of-tree package:

   ```python
   from agent_service.kb.factory import register_store

   @register_store("qdrant")
   def build_qdrant(settings, embeddings):
       return QdrantStore(settings, embeddings)
   ```
3. Add `QDRANT = "qdrant"` to `VectorStoreType` in `agent_service/config.py`, add the optional
   extra to `pyproject.toml`, and document its variables in `.env.example`.
4. Add a parametrised case to `tests/test_store_contract.py` — the shared contract test suite
   already covers domain isolation, dedupe, filtering, deletion and re-index.

Then `VECTOR_STORE=qdrant` is all any caller needs.

## Extending: a new tool or graph node

**A new tool** — add it to `build_tools()` in
[`agent_service/agent/tools.py`](agent_service/agent/tools.py):

```python
class RaiseTicketInput(BaseModel):
    summary: str = Field(description="One line summary.")

async def raise_ticket(summary: str, config: RunnableConfig | None = None) -> str:
    tenant = (config or {}).get("configurable", {}).get("tenant_id")
    ...

StructuredTool.from_function(
    coroutine=raise_ticket, name="raise_ticket",
    description="Open an IT helpdesk ticket.", args_schema=RaiseTicketInput,
)
```

Request-scoped data (allowed domains, tenant, filters) arrives through
`config["configurable"]`, so the graph stays compiled once per process while authorization stays
per request. `ToolNode` picks the new tool up automatically.

**A new node** — write `async def rerank_node(state, deps) -> dict` in
[`agent_service/agent/nodes.py`](agent_service/agent/nodes.py) returning only the state keys it
changes, add any new keys to `AgentState`, then wire it in `AgentRuntime._build()`:

```python
builder.add_node("rerank", _rerank)
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")
```

---

## Project layout

```
agent_service/
  main.py              FastAPI factory, lifespan, OpenAPI, error handlers
  config.py            Pydantic settings, provider/store validation
  api/
    routes_kb.py       /kb endpoints (domain scoped)
    routes_agent.py    /agent chat + SSE stream
    routes_health.py   /health, /ready
    schemas.py         request/response models with examples
  kb/
    base.py            VectorStoreAdapter ABC + generic retriever
    chroma_store.py    collection per domain
    faiss_store.py     index directory per domain (raw faiss, JSON sidecars)
    pinecone_store.py  namespace or index per domain
    factory.py         registry, lazy imports, env validation
    embeddings.py      embedding provider factory (+ deterministic fake)
    ingestion.py       load / chunk / stamp / dedupe / upsert
    filters.py         metadata filter dialect -> native filters
    fusion.py          reciprocal rank fusion
    jobs.py            background ingestion jobs
  agent/
    graph.py           StateGraph assembly + AgentRuntime
    state.py           AgentState
    nodes.py           route_domain / retrieve / generate
    tools.py           kb_search, list_kb_domains
    llm.py             chat model factory (+ deterministic fake)
  core/
    security.py        JWT verification, TokenClaims, scope helpers
    deps.py            FastAPI dependencies / singletons
    logging.py         structured JSON logs with request ids
    middleware.py      request id + access logging
    exceptions.py      error hierarchy and envelope
    retry.py           tenacity policies
tests/                 contract tests per store, auth, API, agent, OpenAPI
scripts/               mint_token.py, export_openapi.py, smoke_test.sh
```

## License

MIT
