"""
Model training script for Titanic survival prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

from data_processing import load_data, prepare_features


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("Training Random Forest...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("Training Gradient Boosting...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    # Initialize model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("Training Logistic Regression...")
    
    # Initialize model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    
    # Train model
    lr.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return lr


def evaluate_model(model, X_train, y_train):
    """
    Evaluate model performance on training data.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
    """
    # Predictions
    y_pred = model.predict(X_train)
    
    # Accuracy
    accuracy = accuracy_score(y_train, y_pred)
    print(f"\nTraining Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_pred))
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        print(feature_importance.head(10))


def save_model(model, model_name='best_model.pkl'):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name for saved model file
    """
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_name}'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


def main():
    """
    Main training pipeline.
    """
    print("=" * 50)
    print("Titanic Survival Prediction - Model Training")
    print("=" * 50)
    
    # Load and prepare data
    print("\nLoading data...")
    train_df, test_df = load_data()
    
    print("Preprocessing data...")
    X_train, y_train, X_test, test_ids = prepare_features(train_df, test_df)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train multiple models
    models = {}
    
    # Random Forest
    print("\n" + "=" * 50)
    models['random_forest'] = train_random_forest(X_train, y_train)
    evaluate_model(models['random_forest'], X_train, y_train)
    save_model(models['random_forest'], 'random_forest_model.pkl')
    
    # Gradient Boosting
    print("\n" + "=" * 50)
    models['gradient_boosting'] = train_gradient_boosting(X_train, y_train)
    evaluate_model(models['gradient_boosting'], X_train, y_train)
    save_model(models['gradient_boosting'], 'gradient_boosting_model.pkl')
    
    # Logistic Regression
    print("\n" + "=" * 50)
    models['logistic_regression'] = train_logistic_regression(X_train, y_train)
    evaluate_model(models['logistic_regression'], X_train, y_train)
    save_model(models['logistic_regression'], 'logistic_regression_model.pkl')
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
