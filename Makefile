.DEFAULT_GOAL := help
.PHONY: help setup models fixtures lint format typecheck test test-models e2e licenses \
        gen-api check dev-api dev-web build-web

UV  ?= uv
WEB := apps/web
HAS_WEB := $(wildcard $(WEB)/package.json)

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## install Python workspace (and web deps when present)
	$(UV) sync --all-groups
ifneq ($(HAS_WEB),)
	npm --prefix $(WEB) ci
endif

models:  ## download pinned model weights into $$DFD_MODELS_DIR (default ./models)
	$(UV) run python tools/fetch_models.py

fixtures:  ## download public-domain test fixtures into tests/.fixtures
	$(UV) run python tools/fetch_fixtures.py

lint:  ## ruff (and eslint)
	$(UV) run ruff check .
	$(UV) run ruff format --check .
ifneq ($(HAS_WEB),)
	npm --prefix $(WEB) run lint
endif

format:  ## apply ruff formatting and fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:  ## mypy (and tsc)
	$(UV) run mypy
ifneq ($(HAS_WEB),)
	npm --prefix $(WEB) run typecheck
endif

test:  ## pytest (model tests skip without weights) and vitest
	$(UV) run pytest
ifneq ($(HAS_WEB),)
	npm --prefix $(WEB) run test
endif

test-models:  ## only the tests that need model weights
	$(UV) run pytest -m models

e2e:  ## Playwright end-to-end test against the API in fake-analyzer mode
	npm --prefix $(WEB) run build
	npm --prefix $(WEB) run e2e

licenses:  ## check dependencies/licenses.yml against installed packages
	$(UV) run python tools/check_licenses.py

gen-api:  ## regenerate the web app's API types from the FastAPI OpenAPI schema
	$(UV) run python tools/export_openapi.py $(WEB)/openapi.json
	npm --prefix $(WEB) run gen:api

check: lint typecheck test licenses  ## everything CI would run

dev-api:  ## run the API with auto-reload on http://127.0.0.1:8000
	$(UV) run uvicorn deepfake_api.main:create_app --factory --reload --host 127.0.0.1 --port 8000

dev-web:  ## run the Vite dev server (proxies /api to dev-api)
	npm --prefix $(WEB) run dev

build-web:  ## build the SPA into apps/web/dist (served by the API)
	npm --prefix $(WEB) run build
