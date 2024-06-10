import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

def ml_model_evaluator(file_path, target_column):
    # List of required packages
    required_packages = [
        "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"
    ]

    # Install and import required packages
    for package in required_packages:
        install_and_import(package)

    # Now you can import the necessary modules from the packages
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Your ML model evaluation code here
    # Load the data
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define classification and regression models
    classification_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "Naive Bayes": GaussianNB()
    }

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "SVM": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "Extra Trees": ExtraTreesRegressor(),
        "Ridge": LinearRegression(),  # Add Ridge regression model here
        "Lasso": LinearRegression()   # Add Lasso regression model here
    }

    # Function to evaluate classification models
    def evaluate_classification_models(models, X_train, X_test, y_train, y_test):
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
        return results

    # Function to evaluate regression models
    def evaluate_regression_models(models, X_train, X_test, y_train, y_test):
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            results[name] = mse
        return results

    # Evaluate models
    if y.dtype == 'object' or len(np.unique(y)) < 10:
        results = evaluate_classification_models(classification_models, X_train, X_test, y_train, y_test)
        metric = "Accuracy"
    else:
        results = evaluate_regression_models(regression_models, X_train, X_test, y_train, y_test)
        metric = "Mean Squared Error"

    # Print results
    for model, result in results.items():
        print(f"{model}: {result:.4f} ({metric})")

    # Plot results
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title(f"Model Performance ({metric})")
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = "breast-cancer.csv"  # Replace with your actual file path
    target_column = "recurrence-events"  # Replace with your actual target column
    ml_model_evaluator(file_path, target_column)
