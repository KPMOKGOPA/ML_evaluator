import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):
    """
    Preprocess the data and split it into features and target.

    Parameters:
    - df (DataFrame): Input data.
    - target_column (str): Name of the target column.

    Returns:
    - tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical variables if necessary
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_classification(X_train, X_test, y_train, y_test):
    """
    Train and evaluate classification models.

    Parameters:
    - X_train (array-like): Features of the training set.
    - X_test (array-like): Features of the testing set.
    - y_train (array-like): Target of the training set.
    - y_test (array-like): Target of the testing set.

    Returns:
    - tuple: best_model, results
    """
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "Naive Bayes": GaussianNB()
    }
    
    best_model = None
    best_accuracy = 0
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name
    
    return best_model, results

def train_and_evaluate_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate regression models.

    Parameters:
    - X_train (array-like): Features of the training set.
    - X_test (array-like): Features of the testing set.
    - y_train (array-like): Target of the training set.
    - y_test (array-like): Target of the testing set.

    Returns:
    - tuple: best_model, results
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "SVM": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "Extra Trees": ExtraTreesRegressor(),
        "Ridge": Ridge(),
        "Lasso": Lasso()
    }
    
    best_model = None
    best_r2 = float('-inf')
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = r2
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = name
    
    return best_model, results

def plot_classification_results(results):
    """
    Plot classification model accuracy.

    Parameters:
    - results (dict): Dictionary containing model names and their corresponding accuracy.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title('Classification Model Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.show()

def plot_regression_results(results):
    """
    Plot regression model R2 score.

    Parameters:
    - results (dict): Dictionary containing model names and their corresponding R2 score.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title('Regression Model R2 Score')
    plt.xlabel('Model')
    plt.ylabel('R2 Score')
    plt.ylim(-1, 1)
    plt.xticks(rotation=45)
    plt.show()

def ml_model_evaluator(file_path, target_column):
    """
    Main function to run ML model evaluation.

    Parameters:
    - file_path (str): Path to the CSV file.
    - target_column (str): Name of the target column.

    Returns:
    - None
    """
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
    
    if df[target_column].dtype == 'object' or len(df[target_column].unique()) <= 10:
        best_model, results = train_and_evaluate_classification(X_train, X_test, y_train, y_test)
        plot_classification_results(results)
    else:
        best_model, results = train_and_evaluate_regression(X_train, X_test, y_train, y_test)
        plot_regression_results(results)
    
    print("\nModel Evaluation Results:")
    for model, score in results.items():
        print(f"{model}: {score}")
    
    print(f"\nBest Model: {best_model}")

# Example usage in a notebook or Colab
# file_path = "your_data.csv"
# target_column = "target_column"
# ml_model_evaluator(file_path, target_column)
