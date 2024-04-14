The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

# Titanic Survival Prediction Using XGBoost

This Jupyter notebook outlines a complete application of the XGBoost algorithm to predict survival outcomes on the Titanic. The process covers several stages typical in machine learning projects: data preparation, feature engineering, model training, and evaluation, all within the setting of a Kaggle competition.

## Libraries and Data Loading
The notebook starts by importing necessary libraries:
- `numpy` and `pandas` for data manipulation,
- `xgboost` and `sklearn` for modeling,
- `matplotlib` and `category_encoders` for visualization and data encoding respectively.

The Titanic dataset is loaded directly from Kaggle's platform, including separate files for training, testing, and submission examples.

## Data Preprocessing
Data is first cleaned and preprocessed:
- Missing values are filled.
- Titles are extracted from names as a new predictive feature.
- Categorical variables are transformed into numerical format using category encoders.

## Feature Engineering
Additional attributes are created to enhance the model's predictive accuracy:
- Family size is calculated from `SibSp` and `Parch`.
- A binary indicator of cabin availability is included.

## Model Training and Evaluation
The XGBoost classifier is configured with specific hyperparameters and trained on the processed data. It is evaluated using accuracy metrics and a confusion matrix to assess its performance:
- The model is trained on a training subset.
- It is further validated on an external test set to ensure it generalizes well.

## Making Predictions and Preparing for Submission
Predictions are made on a separate test dataset from the competition. Results are formatted into a CSV file, ready for submission:
- This demonstrates the model’s practical utility in predicting real-world outcomes based on learned data patterns.

The structured approach provides a clear and instructional view into leveraging XGBoost for binary classification problems in a competitive data science environment.

