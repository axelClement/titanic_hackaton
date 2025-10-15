# Data Directory

This directory should contain the Titanic dataset files from Kaggle.

## Required Files

Download these files from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data):

1. **train.csv** - Training data with passenger information and survival labels
2. **test.csv** - Test data for making predictions
3. **gender_submission.csv** - Sample submission file (optional reference)

## File Descriptions

### train.csv
Contains 891 rows with the following columns:
- PassengerId: Unique identifier for each passenger
- Survived: Survival (0 = No, 1 = Yes) - **Target variable**
- Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Name: Passenger name
- Sex: Gender
- Age: Age in years
- SibSp: Number of siblings/spouses aboard
- Parch: Number of parents/children aboard
- Ticket: Ticket number
- Fare: Passenger fare
- Cabin: Cabin number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

### test.csv
Contains 418 rows with the same columns as train.csv except for the 'Survived' column.

## How to Get the Data

1. Go to https://www.kaggle.com/c/titanic/data
2. Click on "Download All" or download individual files
3. Extract the files to this directory

## Notes

- The actual data files are not included in the repository
- Add `data/*.csv` to `.gitignore` to avoid committing large data files
- A sample submission file format is included for reference
