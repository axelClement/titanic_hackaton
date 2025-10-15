#!/bin/bash

# Setup script for Titanic ML project

set -e

echo "=========================================="
echo "Titanic ML Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if python version is >= 3.7
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
    echo "Error: Python 3.7 or higher is required"
    exit 1
fi

echo "✓ Python version is compatible"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate  (Linux/Mac)"
echo "  venv\\Scripts\\activate     (Windows)"
echo ""

# Install dependencies
echo "Installing dependencies..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "Please activate virtual environment and run:"
    echo "  pip install -r requirements.txt"
fi
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data models submissions
echo "✓ Directories created"
echo ""

# Check for data files
echo "Checking for data files..."
if [ -f "data/train.csv" ] && [ -f "data/test.csv" ]; then
    echo "✓ Data files found"
else
    echo "⚠ Data files not found"
    echo ""
    echo "Please download the following files from Kaggle:"
    echo "  https://www.kaggle.com/c/titanic/data"
    echo ""
    echo "Required files:"
    echo "  - data/train.csv"
    echo "  - data/test.csv"
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment"
echo "2. Download data from Kaggle (if not done)"
echo "3. Run: python src/train.py"
echo "4. Run: python src/predict.py"
echo ""
echo "For more information, see QUICKSTART.md"
