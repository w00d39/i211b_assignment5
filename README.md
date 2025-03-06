# Breast Cancer Classification

This project aims to classify breast cancer data using various machine learning models, including Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN). The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset from the `sklearn` library.

## Project Structure

- `assignment5.py`: The main script that loads the dataset, preprocesses the data, trains the models, and evaluates their performance.

## Dependencies

- Python 3.x
- pandas
- scikit-learn

## Models
Random Forest
The Random Forest model is trained with 77 estimators. The model's performance is evaluated using accuracy, F1 score, precision, and recall.

Support Vector Machine (SVM)
Two SVM models are trained:

An initial SVM model with default parameters.
A tuned SVM model using GridSearchCV to find the best hyperparameters.
K-Nearest Neighbors (KNN)
Two KNN models are trained:

An initial KNN model with 5 neighbors.
A tuned KNN model using GridSearchCV to find the best hyperparameters.
Results
The script prints the accuracy, F1 score, precision, and recall for each model. Additionally, it prints the best hyperparameters found by GridSearchCV for the tuned SVM and KNN models.
