.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV        ?= .venv
PY          := $(VENV)/bin/python
PIP         := uv pip install --python $(PY)
RUFF        := $(VENV)/bin/ruff
MYPY        := $(VENV)/bin/mypy
PYTEST      := $(VENV)/bin/pytest
BANDIT      := $(VENV)/bin/bandit
PIP_AUDIT   := $(VENV)/bin/pip-audit
CODESPELL   := $(VENV)/bin/codespell
COV_MIN     ?= 85
BASE_URL    ?= http://localhost:8000

# Advisories with no fixed release upstream. Re-review on every Dependabot bump.
#   PYSEC-2026-311 / -3813 / -3814 / -3815: ChromaDB *server* auth, RBAC and
#   code-injection issues. This service uses the embedded PersistentClient by
#   default; when CHROMA_HOST is set, keep the Chroma server on a private
#   network (see the README).
PIP_AUDIT_IGNORE ?= --ignore-vuln PYSEC-2026-311 --ignore-vuln PYSEC-2026-3813 \
                    --ignore-vuln PYSEC-2026-3814 --ignore-vuln PYSEC-2026-3815

# Offline providers so every target runs without credentials.
TEST_ENV := JWT_SECRET_KEY=test-secret-key-that-is-long-enough-32ch \
            LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake VECTOR_STORE=chroma

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Create the venv and install the project with dev extras
	uv venv --python 3.12 $(VENV)
	$(PIP) -e ".[dev]"
	$(VENV)/bin/pre-commit install || true

.PHONY: dev
dev: ## Run the API with autoreload
	$(VENV)/bin/uvicorn agent_service.main:get_app --factory --reload --host 0.0.0.0 --port 8000

.PHONY: lint
lint: ## ruff check (includes import sorting)
	$(RUFF) check agent_service tests scripts

.PHONY: format
format: ## ruff format
	$(RUFF) format agent_service tests scripts

.PHONY: format-check
format-check: ## ruff format --check
	$(RUFF) format --check agent_service tests scripts

.PHONY: typecheck
typecheck: ## mypy --strict
	$(MYPY) agent_service

.PHONY: spell
spell: ## codespell over source, scripts and docs
	$(CODESPELL) agent_service tests scripts README.md .env.example

.PHONY: test
test: ## Run the test suite
	$(TEST_ENV) $(PYTEST)

.PHONY: cov
cov: ## Run tests with coverage and enforce the threshold
	$(TEST_ENV) $(PYTEST) --cov=agent_service --cov-report=term-missing \
	  --cov-report=xml:coverage.xml --cov-fail-under=$(COV_MIN)

.PHONY: security
security: ## bandit + pip-audit
	$(BANDIT) -c pyproject.toml -r agent_service
	$(PIP_AUDIT) --progress-spinner off $(PIP_AUDIT_IGNORE)

.PHONY: openapi
openapi: ## Regenerate openapi.json
	$(PY) scripts/export_openapi.py

.PHONY: openapi-check
openapi-check: ## Fail if openapi.json drifted from the code
	$(PY) scripts/export_openapi.py --check

.PHONY: smoke
smoke: ## End-to-end smoke test against BASE_URL
	BASE_URL=$(BASE_URL) PYTHON=$(PY) bash scripts/smoke_test.sh

.PHONY: docker-build
docker-build: ## Build the container image
	docker build -t nexusone/agent-service:local .

.PHONY: docker-smoke
docker-smoke: docker-build ## Run the image and smoke test it
	docker rm -f agent-service-smoke >/dev/null 2>&1 || true
	docker run -d --name agent-service-smoke -p 8000:8000 \
	  -e JWT_SECRET_KEY=local-development-secret-change-me-32chars \
	  -e LLM_PROVIDER=fake -e EMBEDDING_PROVIDER=fake -e VECTOR_STORE=chroma \
	  nexusone/agent-service:local
	JWT_SECRET_KEY=local-development-secret-change-me-32chars \
	  BASE_URL=http://localhost:8000 PYTHON=$(PY) bash scripts/smoke_test.sh; \
	  status=$$?; docker rm -f agent-service-smoke >/dev/null; exit $$status

.PHONY: clean
clean: ## Remove caches and local data
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage data

.PHONY: ci
ci: lint format-check typecheck spell cov security openapi-check ## Everything CI runs
	@echo "CI checks passed."
