"""
Compare performance of different models on the training data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import time

from data_processing import load_data, prepare_features


def compare_models(X_train, y_train):
    """
    Compare performance of multiple classification models.
    
    Args:
        X_train: Training features
        y_train: Training labels
    """
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = []
    
    print("=" * 70)
    print("Model Comparison - 5-Fold Cross-Validation")
    print("=" * 70)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        training_time = time.time() - start_time
        
        results.append({
            'Model': name,
            'Mean Accuracy': cv_scores.mean(),
            'Std Dev': cv_scores.std(),
            'Min Accuracy': cv_scores.min(),
            'Max Accuracy': cv_scores.max(),
            'Training Time (s)': training_time
        })
        
        print(f"  Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Training Time: {training_time:.2f}s")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Mean Accuracy', ascending=False)
    
    # Display results
    print("\n" + "=" * 70)
    print("Results Summary (Sorted by Mean Accuracy)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Best model
    best_model = results_df.iloc[0]
    print("\n" + "=" * 70)
    print(f"Best Model: {best_model['Model']}")
    print(f"Accuracy: {best_model['Mean Accuracy']:.4f} (+/- {best_model['Std Dev'] * 2:.4f})")
    print("=" * 70)
    
    return results_df


def main():
    """
    Main comparison pipeline.
    """
    print("=" * 70)
    print("Titanic Survival Prediction - Model Comparison")
    print("=" * 70)
    
    # Load and prepare data
    print("\nLoading data...")
    train_df, test_df = load_data()
    
    print("Preprocessing data...")
    X_train, y_train, X_test, test_ids = prepare_features(train_df, test_df)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Compare models
    print("\n")
    results = compare_models(X_train, y_train)
    
    # Save results
    results.to_csv('model_comparison_results.csv', index=False)
    print(f"\nResults saved to model_comparison_results.csv")


if __name__ == '__main__':
    main()
