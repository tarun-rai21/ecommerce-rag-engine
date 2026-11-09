# Common tasks. Python commands run through uv (https://docs.astral.sh/uv/), which uses
# pyproject.toml + uv.lock. `make help` lists the targets.

COMPOSE     := docker compose
OBS_COMPOSE := $(COMPOSE) -f docker-compose.yml -f docker-compose.observability.yml
RUN         := uv run

.PHONY: help install hooks up down logs observability ingest ingest-versioned rollback \
        test lint fix eval-retrieval eval-filters eval-sizes eval-generation bench cost dashboard

help:            ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | expand -t 20

install:         ## Create .venv with every dependency group (from uv.lock)
	uv sync --all-groups

hooks:           ## Install the pre-commit hooks
	$(RUN) pre-commit install

up:              ## Start Qdrant, Redis, backend and frontend
	$(COMPOSE) up -d --build

down:            ## Stop the stack (data volumes are kept)
	$(OBS_COMPOSE) down

logs:            ## Follow the backend log
	$(COMPOSE) logs -f backend

observability:   ## Start the stack with Prometheus and Grafana (http://localhost:3000)
	$(OBS_COMPOSE) up -d --build

ingest:          ## Index WANDS into the collection named by COLLECTION_NAME
	$(RUN) python -m src.ingestion.ingest

ingest-versioned: ## Build the next version, validate it, switch the alias (zero downtime)
	$(RUN) python -m src.ingestion.ingest --versioned

rollback:        ## Point the alias back to the previous index version
	$(RUN) python -m src.ingestion.ingest --rollback

test:            ## Run the test suite
	$(RUN) pytest

lint:            ## Lint (ruff) and run the pre-commit checks
	$(RUN) pre-commit run --all-files

fix:             ## Apply ruff's safe fixes
	$(RUN) ruff check --fix .

eval-retrieval:  ## Retrieval benchmark on the test split
	$(RUN) python -m src.evaluation.retrieval_eval --split test

eval-filters:    ## Price-constrained retrieval benchmark
	$(RUN) python -m src.evaluation.filter_eval --split test

eval-sizes:      ## Room-size rug benchmark
	$(RUN) python -m src.evaluation.size_eval

eval-generation: ## Answer-quality evaluation (calls Groq; cached)
	$(RUN) python -m src.evaluation.generation_eval

bench:           ## Load test against the running stack
	$(RUN) python -m benchmarks.load_test

cost:            ## Cost per 1,000 questions from the latest load test
	$(RUN) python -m benchmarks.cost

dashboard:       ## Regenerate the Grafana dashboard JSON
	$(RUN) python observability/build_dashboard.py
