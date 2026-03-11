.PHONY: setup install clean test run

setup: install
	@echo "✅ Setup complete! Run: python main.py"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "📥 Downloading ONNX model..."
	python -c "from src.core.onnx_server import ONNXServer; s = ONNXServer(); s.download_model()"

clean:
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__ .pytest_cache *.pyc
	rm -f yuzu_local.duckdb yuzu_local.duckdb.wal

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

run:
	@echo "🚀 Starting Yuzu Memory Builder..."
	python main.py

help:
	@echo "Available commands:"
	@echo "  make setup    - Install all dependencies"
	@echo "  make clean    - Clean temporary files"
	@echo "  make test     - Run tests"
	@echo "  make run      - Start the application"
