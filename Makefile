# freight-parser — thin wrappers over `uv run ...`.
# Every target runs inside the uv-managed venv; nothing needs a global Python.
.PHONY: demo qa synth train train-all eval serve playground retrain test fmt clean help
.DEFAULT_GOAL := help

MODEL ?= newer
PORT  ?= 8000

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

demo:  ## Run the whole pipeline end to end (QA -> synth -> train old+new -> eval -> serve)
	uv run python scripts/demo.py

qa:  ## Launch the Django QA review portal (migrate + seed + runserver at :8000)
	uv run python packages/qa_portal/manage.py migrate
	uv run python packages/qa_portal/manage.py seed
	@# Demo-only admin so the review cockpit is usable out of the box: admin / admin.
	DJANGO_SUPERUSER_PASSWORD=admin uv run python packages/qa_portal/manage.py \
		createsuperuser --noinput --username admin --email admin@example.com 2>/dev/null || true
	@echo "QA portal admin: http://localhost:8000/  (login: admin / admin)"
	uv run python packages/qa_portal/manage.py runserver

synth:  ## Generate synthetic data into DuckDB and export it
	uv run python -m synthetic.cli generate --n 500 --seed 0
	uv run python -m synthetic.cli export --out data/training/synthetic.jsonl

train:  ## Fine-tune one model (MODEL=older|newer|lightweight)
	uv run python -m finetune.train --model $(MODEL)

train-all:  ## Fine-tune the older and newer models
	uv run python -m finetune.train_all

eval:  ## Benchmark the trained models and print the comparison table
	uv run python -m eval.predict --model older
	uv run python -m eval.predict --model newer
	uv run python -m eval.report

serve:  ## Boot the FastAPI serving API + browser playground (http://localhost:$(PORT)/)
	uv run uvicorn serving.app:app --host 0.0.0.0 --port $(PORT)

playground: serve  ## Alias for `serve` — the playground is served at /

retrain:  ## Closed-loop retrain: train candidate, benchmark, promote only if it wins
	uv run python scripts/retrain.py

test:  ## Run every package's pytest suite
	uv run pytest

fmt:  ## Format + autofix with ruff
	uv run ruff format .
	uv run ruff check --fix .

clean:  ## Remove generated demo artifacts (adapters, DBs, datasets, predictions)
	rm -rf data/adapters data/predictions data/training data/synthetic.duckdb data/synthetic.duckdb.wal
	rm -f packages/qa_portal/db.sqlite3
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
