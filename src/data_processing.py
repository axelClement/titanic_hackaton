"""
Utility functions for data preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    """
    Load train and test datasets.
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        
    Returns:
        train_df, test_df: Training and test DataFrames
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(df, is_train=True):
    """
    Preprocess the Titanic dataset.
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data (has 'Survived' column)
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Fill missing Age values with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Fill missing Embarked values with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Fill missing Fare values with median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Drop Cabin column (too many missing values)
    if 'Cabin' in df.columns:
        df.drop('Cabin', axis=1, inplace=True)
    
    # Create Title feature from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Rev': 'Other',
        'Dr': 'Other',
        'Col': 'Other',
        'Major': 'Other',
        'Mlle': 'Miss',
        'Countess': 'Other',
        'Ms': 'Miss',
        'Lady': 'Other',
        'Jonkheer': 'Other',
        'Don': 'Other',
        'Dona': 'Other',
        'Mme': 'Mrs',
        'Capt': 'Other',
        'Sir': 'Other'
    }
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'].fillna('Other', inplace=True)
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create IsAlone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Create Age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])
    
    # Create Fare groups
    df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
    
    # Drop unnecessary columns
    df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True, errors='ignore')
    
    return df


def encode_features(train_df, test_df):
    """
    Encode categorical features for both train and test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Encoded train and test DataFrames
    """
    # Combine datasets for consistent encoding
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Identify categorical columns
    categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
    
    # Use one-hot encoding for categorical variables
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
    
    # Ensure both datasets have the same columns
    missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
    for col in missing_cols:
        if col != 'Survived':
            test_encoded[col] = 0
    
    # Ensure test has same column order as train
    if 'Survived' in train_encoded.columns:
        feature_cols = [col for col in train_encoded.columns if col != 'Survived']
        test_encoded = test_encoded[feature_cols]
    else:
        feature_cols = train_encoded.columns.tolist()
        test_encoded = test_encoded[feature_cols]
    
    return train_encoded, test_encoded


def prepare_features(train_df, test_df):
    """
    Complete preprocessing pipeline.
    
    Args:
        train_df: Raw training DataFrame
        test_df: Raw test DataFrame
        
    Returns:
        X_train, y_train, X_test: Processed features and labels
    """
    # Store test PassengerId before dropping
    test_ids = test_df['PassengerId'].copy()
    
    # Preprocess both datasets
    train_processed = preprocess_data(train_df, is_train=True)
    test_processed = preprocess_data(test_df, is_train=False)
    
    # Separate target variable
    y_train = train_processed['Survived']
    train_processed = train_processed.drop('Survived', axis=1)
    
    # Encode features
    X_train, X_test = encode_features(train_processed, test_processed)
    
    return X_train, y_train, X_test, test_ids
