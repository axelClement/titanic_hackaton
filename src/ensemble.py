"""
Ensemble model combining multiple classifiers for better predictions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
import os

from data_processing import load_data, prepare_features


def create_ensemble_model():
    """
    Create an ensemble model combining multiple classifiers.
    
    Returns:
        VotingClassifier ensemble model
    """
    # Define individual models
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    
    svc = SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr),
            ('svc', svc)
        ],
        voting='soft',
        weights=[2, 2, 1, 1]  # Give more weight to RF and GB
    )
    
    return ensemble


def train_ensemble(X_train, y_train):
    """
    Train the ensemble model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained ensemble model
    """
    print("Training Ensemble Model...")
    print("This combines Random Forest, Gradient Boosting, Logistic Regression, and SVM")
    
    # Create and train ensemble
    ensemble = create_ensemble_model()
    ensemble.fit(X_train, y_train)
    
    # Cross-validation
    print("\nEvaluating with 5-fold cross-validation...")
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return ensemble


def main():
    """
    Main ensemble training pipeline.
    """
    print("=" * 50)
    print("Titanic Survival - Ensemble Model Training")
    print("=" * 50)
    
    # Load and prepare data
    print("\nLoading data...")
    train_df, test_df = load_data()
    
    print("Preprocessing data...")
    X_train, y_train, X_test, test_ids = prepare_features(train_df, test_df)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train ensemble
    print("\n" + "=" * 50)
    ensemble = train_ensemble(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/ensemble_model.pkl'
    joblib.dump(ensemble, model_path)
    print(f"\nEnsemble model saved to {model_path}")
    
    print("\n" + "=" * 50)
    print("Ensemble training completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
