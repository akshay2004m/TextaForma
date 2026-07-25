.PHONY: help install run test lint clean train check-env

help:
	@echo "TextForma - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make check-env      Validate .env configuration"
	@echo ""
	@echo "Running:"
	@echo "  make run            Start Flask development server"
	@echo "  make train          Train BART model (requires training deps)"
	@echo ""
	@echo "Testing & Linting:"
	@echo "  make test           Run unit tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make lint           Check code style with flake8"
	@echo "  make format         Auto-format code with black"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean temporary files and caches"
	@echo "  make clean-db       Remove database file"
	@echo "  make stats          Show database statistics"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -r requirements-training.txt

check-env:
	@python scripts/check_env.py

run:
	@echo "Starting TextForma on http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	@python app.py

train:
	@echo "Training BART model on Samanantar dataset..."
	@python scripts/train_bart.py

test:
	@pytest tests/ -v

test-cov:
	@pytest tests/ --cov=app --cov-report=html --cov-report=term

lint:
	@flake8 app.py tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	@flake8 app.py tests/ --count --max-complexity=10 --max-line-length=127 --statistics

format:
	@black app.py tests/
	@isort app.py tests/

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/ 2>/dev/null || true
	@rm -rf dist/ build/ *.egg-info/ 2>/dev/null || true
	@echo "Cleaned temporary files"

clean-db:
	@rm -f text_formalizer.db
	@echo "Database removed"

stats:
	@python check_stats.py || echo "No database found. Run the app first."
