# Titanic: Machine Learning from Disaster

This repository contains a machine learning solution for the Kaggle Titanic competition - predicting survival on the Titanic using passenger data.

## Project Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. This project uses machine learning to predict which passengers survived the tragedy based on features like age, sex, passenger class, and more.

## Project Structure

```
titanic_hackaton/
├── data/                  # Data files (train.csv, test.csv)
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── src/                   # Source code for data processing and modeling
├── models/                # Saved model files
├── submissions/           # Generated submission files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup

1. Clone this repository:
```bash
git clone https://github.com/axelClement/titanic_hackaton.git
cd titanic_hackaton
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Titanic dataset from Kaggle and place the files in the `data/` directory:
   - `train.csv`
   - `test.csv`
   - `gender_submission.csv` (sample submission)

## Usage

### Training the Model

Run the training script to preprocess data and train the model:

```bash
python src/train.py
```

### Making Predictions

Generate predictions on the test set:

```bash
python src/predict.py
```

The submission file will be saved in the `submissions/` directory.

### Exploratory Data Analysis

Open the Jupyter notebook for data exploration:

```bash
jupyter notebook notebooks/eda.ipynb
```

## Features

The solution includes:

- **Data Preprocessing**: Handling missing values, encoding categorical variables
- **Feature Engineering**: Creating new features from existing ones
- **Model Training**: Multiple algorithms including Random Forest, Gradient Boosting
- **Cross-Validation**: K-fold validation for robust model evaluation
- **Hyperparameter Tuning**: Grid search for optimal parameters

## Model Performance

The model achieves competitive accuracy on the Kaggle leaderboard through careful feature engineering and ensemble methods.

## Contributing

Feel free to open issues or submit pull requests to improve the solution.

## License

This project is open source and available under the MIT License.