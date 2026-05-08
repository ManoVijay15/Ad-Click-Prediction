.PHONY: install lint test train serve infra-up infra-down clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	mypy src

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

baseline:
	.venv/bin/python -m src.models.baseline

train:
	.venv/bin/python -m src.models.train

tune:
	.venv/bin/python -m src.models.tune

promote:
	.venv/bin/python -m src.models.promote

populate-store:
	.venv/bin/python -m src.features.populate_store

batch-score:
	.venv/bin/python -m src.models.batch_score \
		--input data/processed/test.csv \
		--output data/processed/test_predictions.parquet

serve:
	.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mkdir -p mlruns
	.venv/bin/mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 \
		--port 5000

infra-up:
	docker compose up -d postgres redis
	@echo "Postgres:  localhost:5432"
	@echo "Redis:     localhost:6379"
	@echo "Run 'make mlflow' in a separate terminal for MLflow UI"

infra-down:
	docker compose down

infra-up-all:
	docker compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/
