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

train:
	python -m src.models.train

tune:
	python -m src.models.tune

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

infra-up:
	docker compose up -d postgres mlflow redis
	@echo "MLflow UI: http://localhost:5000"
	@echo "Redis:     localhost:6379"

infra-down:
	docker compose down

infra-up-all:
	docker compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/
