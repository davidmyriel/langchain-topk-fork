.PHONY: all format lint test tests integration_test integration_tests check_imports help

all: help

TEST_FILE ?= tests/unit_tests/

integration_test integration_tests: TEST_FILE = tests/integration_tests/

test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

integration_test integration_tests:
	uv run --group test --group test_integration --extra fastembed pytest -v --tb=short $(TEST_FILE)

PYTHON_FILES=.
lint format: PYTHON_FILES=.

lint:
	./scripts/lint_imports.sh
	uv run --all-groups ruff check $(PYTHON_FILES)
	uv run --all-groups ruff format $(PYTHON_FILES) --diff
	uv run --all-groups mypy langchain_topk tests

format:
	uv run --all-groups ruff format $(PYTHON_FILES)
	uv run --all-groups ruff check --fix $(PYTHON_FILES)

check_imports: $(shell find langchain_topk -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

help:
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'integration_test             - run integration tests (needs TOPK_API_KEY)'
	@echo 'check_imports                - check imports'
