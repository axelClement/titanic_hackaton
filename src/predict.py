"""
Prediction script for generating Kaggle submission files.
"""

import pandas as pd
import joblib
import os
from datetime import datetime

from data_processing import load_data, prepare_features


def load_model(model_path='models/random_forest_model.pkl'):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def generate_submission(model, X_test, test_ids, filename=None):
    """
    Generate a submission file for Kaggle.
    
    Args:
        model: Trained model
        X_test: Test features
        test_ids: Passenger IDs from test set
        filename: Output filename (optional)
        
    Returns:
        Submission DataFrame
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    
    # Save to file
    os.makedirs('submissions', exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'submission_{timestamp}.csv'
    
    filepath = f'submissions/{filename}'
    submission.to_csv(filepath, index=False)
    
    print(f"\nSubmission file saved to {filepath}")
    print(f"Total predictions: {len(submission)}")
    print(f"Survived: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
    print(f"Did not survive: {len(predictions) - predictions.sum()} ({(len(predictions) - predictions.sum())/len(predictions)*100:.2f}%)")
    
    return submission


def main():
    """
    Main prediction pipeline.
    """
    print("=" * 50)
    print("Titanic Survival Prediction - Making Predictions")
    print("=" * 50)
    
    # Load and prepare data
    print("\nLoading data...")
    train_df, test_df = load_data()
    
    print("Preprocessing data...")
    X_train, y_train, X_test, test_ids = prepare_features(train_df, test_df)
    
    print(f"Test set shape: {X_test.shape}")
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_model('models/random_forest_model.pkl')
    
    # Generate predictions
    print("\nGenerating predictions...")
    submission = generate_submission(model, X_test, test_ids)
    
    print("\n" + "=" * 50)
    print("Prediction completed successfully!")
    print("Upload the submission file to Kaggle to see your score.")
    print("=" * 50)


if __name__ == '__main__':
    main()
