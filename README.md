# ML Model Evaluator

A Python program that allows users to input data and automatically tests multiple machine learning models to find the one that predicts the best.

## Features

- Load user data from a CSV file.
- Preprocess the data.
- Split the data into training and testing sets.
- Train multiple machine learning models.
- Evaluate models using common metrics.
- Report the best model.
- Visualize model performance with plots.

## Models Included

### Classification Models

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- K-Nearest Neighbors
- Gradient Boosting
- AdaBoost
- Extra Trees
- Naive Bayes

### Regression Models

- Linear Regression
- Decision Tree
- Random Forest
- SVM
- K-Nearest Neighbors
- Gradient Boosting
- AdaBoost
- Extra Trees
- Ridge
- Lasso

## Installation

1. You can install the ML Model Evaluator library via pip:

    ```bash
    pip install ML_evaluator

    ```

2. Install the required packages:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```



## Usage

 ```python
from ml_model_evaluator import ml_model_evaluator

# Replace 'file_path' and 'target_column' with appropriate values
file_path = "your_data.csv"
target_column = "target_column_name"

ml_model_evaluator(file_path, target_column)
```


## Contributing

Feel free to fork this repository, make changes, and submit pull requests. All contributions are welcome!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
