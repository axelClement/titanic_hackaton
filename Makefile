.PHONY: help setup install train predict ensemble compare clean test

help:
	@echo "Titanic ML Project - Available Commands"
	@echo "========================================"
	@echo "make setup      - Run initial setup"
	@echo "make install    - Install dependencies"
	@echo "make train      - Train all models"
	@echo "make predict    - Generate predictions"
	@echo "make ensemble   - Train ensemble model"
	@echo "make compare    - Compare all models"
	@echo "make clean      - Clean generated files"
	@echo "make test       - Run syntax checks"

setup:
	@echo "Running setup script..."
	@bash setup.sh

install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

train:
	@echo "Training models..."
	@python src/train.py

predict:
	@echo "Generating predictions..."
	@python src/predict.py

ensemble:
	@echo "Training ensemble model..."
	@python src/ensemble.py

compare:
	@echo "Comparing models..."
	@python src/compare_models.py

clean:
	@echo "Cleaning generated files..."
	@rm -rf models/*.pkl models/*.joblib
	@rm -rf submissions/*.csv
	@rm -f model_comparison_results.csv
	@rm -rf __pycache__ src/__pycache__
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "Clean complete"

test:
	@echo "Running syntax checks..."
	@python -m py_compile src/*.py
	@echo "All syntax checks passed!"
