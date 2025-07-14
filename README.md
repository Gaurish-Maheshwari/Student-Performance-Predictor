# Student Performance Predictor

This is a machine learning web application developed using **Flask** that predicts a student's math exam performance based on demographic and academic-related inputs such as gender, parental education level, and reading/writing scores.

## Features

- Interactive web interface to input student information
- Machine learning backend to predict student exam performance
- Automated training and model selection from multiple regressors
- Logging and exception handling for robustness
- Modular architecture with a clear separation of concerns:
  - Model training
  - Prediction pipeline
  - Web interface

## Functionalities

- Home page with user input form
- Model selection among multiple regressors based on R² score
- Thresholding to reject poor models (R² < 0.6)
- Uses trained and serialized model for real-time prediction
- Displays predicted exam score for the given inputs

## Technologies Used

- **Python**
- **Flask** (for building the web app)
- **Pandas, NumPy** (for data handling)
- **Scikit-learn** (for ML models and preprocessing)
- **CatBoost, XGBoost** (for gradient boosting models)
- **Logging** (for tracking events and debugging)
- **Pickle** (for saving the trained model)
- **HTML/Jinja2** (for rendering templates)

## Machine Learning Models Considered

The app evaluates and compares the following regressors:

- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- XGBoost Regressor
- CatBoost Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor

The best-performing model (based on R² score) is saved and used for inference.

## Input Fields

The following inputs are required:

- `gender`: Male or Female
- `race_ethnicity`: Ethnic group classification
- `parental_level_of_education`: Highest education level of parent(s)
- `lunch`: Standard or free/reduced
- `test_preparation_course`: Completed or not completed
- `reading_score`: Integer (0–100)
- `writing_score`: Integer (0–100)

## Output

- Predicted math score based on other features
