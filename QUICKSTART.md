# Quick Start Guide

This guide will help you get started with the Titanic ML solution.

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/axelClement/titanic_hackaton.git
cd titanic_hackaton
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Getting the Data

1. Visit the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
2. Download `train.csv` and `test.csv`
3. Place them in the `data/` directory

## Usage Workflow

### 1. Exploratory Data Analysis (Optional)

Explore the data using the provided Jupyter notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

### 2. Train Models

Train machine learning models on the training data:

```bash
# Train individual models (Random Forest, Gradient Boosting, Logistic Regression)
python src/train.py

# Or train an ensemble model for better performance
python src/ensemble.py
```

This will:
- Load and preprocess the data
- Train multiple models with hyperparameter tuning
- Evaluate model performance
- Save trained models to `models/` directory

### 3. Generate Predictions

Create a submission file for Kaggle:

```bash
python src/predict.py
```

This will:
- Load the trained model
- Make predictions on the test set
- Generate a submission CSV file in `submissions/` directory

### 4. Submit to Kaggle

1. Go to the [Titanic Competition Submissions](https://www.kaggle.com/c/titanic/submit)
2. Upload your submission CSV file
3. View your score on the leaderboard!

## Project Structure

```
titanic_hackaton/
├── data/
│   ├── README.md              # Data directory documentation
│   ├── sample_submission.csv  # Sample submission format
│   ├── train.csv             # Training data (download from Kaggle)
│   └── test.csv              # Test data (download from Kaggle)
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis notebook
├── src/
│   ├── __init__.py           # Package initialization
│   ├── data_processing.py    # Data preprocessing and feature engineering
│   ├── train.py              # Model training script
│   ├── ensemble.py           # Ensemble model training
│   └── predict.py            # Prediction and submission generation
├── models/                    # Saved model files (generated)
├── submissions/              # Generated submission files
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── README.md                # Main project documentation
└── QUICKSTART.md           # This file
```

## Model Information

### Individual Models

- **Random Forest**: Ensemble of decision trees with hyperparameter tuning
- **Gradient Boosting**: Sequential ensemble method with boosting
- **Logistic Regression**: Linear model for binary classification

### Ensemble Model

Combines all individual models using soft voting for more robust predictions.

## Feature Engineering

The solution includes several engineered features:

- **Title**: Extracted from passenger names (Mr, Mrs, Miss, Master, etc.)
- **FamilySize**: Total family members aboard (SibSp + Parch + 1)
- **IsAlone**: Binary indicator for solo travelers
- **AgeGroup**: Categorical age bins (Child, Teen, Adult, MiddleAge, Senior)
- **FareGroup**: Categorical fare bins (Low, Medium, High, VeryHigh)

## Tips for Better Performance

1. **Feature Engineering**: Experiment with new features in `src/data_processing.py`
2. **Hyperparameter Tuning**: Adjust parameters in `src/train.py`
3. **Ensemble Methods**: Use `src/ensemble.py` for combining multiple models
4. **Cross-Validation**: Check CV scores to avoid overfitting

## Troubleshooting

### Missing data files
- Make sure you've downloaded `train.csv` and `test.csv` from Kaggle
- Place them in the `data/` directory

### Import errors
- Run `pip install -r requirements.txt` to install all dependencies
- Make sure you're in the project root directory

### Model not found error
- Train a model first using `python src/train.py` or `python src/ensemble.py`
- Check that model files exist in the `models/` directory

## Next Steps

- Try different feature engineering approaches
- Experiment with other ML algorithms (XGBoost, LightGBM)
- Perform more extensive hyperparameter tuning
- Analyze misclassified samples to improve the model

Good luck with your Kaggle submission!
