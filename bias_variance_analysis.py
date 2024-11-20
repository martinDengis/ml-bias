from data import load_wine_quality
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from tabulate import tabulate
from typing import Union
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os


# Declare global variables
random_state = np.random.RandomState(42)
X, y = load_wine_quality()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state)
timestamp = datetime.datetime.now().strftime(
    "%Y%m%d_%H_%M_%S")    # Generate a timestamp


def train_and_evaluate_models(n_samples: int, model_type: str, hyperparameter: Union[float, int] = None, n_estimators: int = None) -> tuple:
    """
    Train and evaluate different regression models to analyze bias and variance.
    Parameters:
        n_samples (int): Number of samples to use for training each model.
        model_type (str): Type of model to train. Supported types are 'lasso', 'knn', 'tree', 'tree_fully_grown', 'bagging', and 'boosting'.
        hyperparameter (Union[float, int], optional): Hyperparameter for the model. For 'lasso', it is alpha. For 'knn', it is n_neighbors. For 'tree' and 'boosting', it is max_depth.
        n_estimators (int, optional): Number of estimators for ensemble methods like 'bagging' and 'boosting'.
    Returns:
        tuple: A tuple containing average error, average bias, and average variance of the trained models.
    """
    # Model
    if model_type == 'lasso':
        model = Lasso(alpha=hyperparameter, random_state=random_state)
    elif model_type == 'knn':
        model = KNeighborsRegressor(
            n_neighbors=hyperparameter)
    elif model_type == 'tree':
        model = DecisionTreeRegressor(
            max_depth=hyperparameter, random_state=random_state)
    elif model_type == 'tree_fully_grown':
        model = DecisionTreeRegressor()
    elif model_type == 'bagging':
        model = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=hyperparameter),
            n_estimators=n_estimators,
            random_state=random_state,
        )
    elif model_type == 'boosting':
        model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=hyperparameter),
            n_estimators=n_estimators,
            random_state=random_state,
        )
    else:
        raise ValueError("Unsupported model type")

    # Train models
    errors = []
    variances = []
    bias = []

    n_models = X_train.shape[0] // n_samples    # Number of models to train
    for i in range(n_models):
        X_sample = X_train[i * n_samples:(i + 1) * n_samples]
        y_sample = y_train[i * n_samples:(i + 1) * n_samples]

        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_test)

        # Calculate bias, variance, and error
        error_i = np.mean((y_pred - y_test)**2)
        variance_i = np.mean((y_pred - np.mean(y_pred))**2)
        bias_i = error_i - variance_i   # residual error considered constant

        errors.append(error_i)
        variances.append(variance_i)
        bias.append(bias_i)

    avg_error = np.mean(errors)
    avg_bias = np.mean(bias)
    avg_variance = np.mean(variances)

    return avg_error, avg_bias, avg_variance


def output_plot(model: str, hyperparameter_name: str, results: np.ndarray) -> None:
    """
    Output a plot of the results.

    Parameters:
        model (str): Folder to save the plot in.
        hyperparameter_name (str): Name of the hyperparameter.
        results (np.ndarray): Results to plot.
    """
    hyperparameter_values = results[:, 0]
    expected_error = results[:, 1]
    bias = results[:, 2]
    variance = results[:, 3]

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(hyperparameter_values, bias, 's-',
            label="Bias + Residual Err.", color="red")  # squares
    ax.plot(hyperparameter_values, variance, '^-',
            label="Variance", color="green")  # triangles
    ax.plot(hyperparameter_values, expected_error, 'D-',
            label="Total Error", color="blue")  # diamonds

    ax.set_xlabel(f"Hyperparameter ({hyperparameter_name})")
    ax.set_ylabel("Error")
    ax.set_title(f"{model.capitalize()} Reg. Bias-Variance Decomposition")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.show()
    plt.savefig(os.path.join(model, f"plot_{timestamp}.png"))
    plt.close()

    print(f"Plot for {model} regression saved to {model}/plot_{timestamp}.png")


def run_train_eval(model: str, hyperparameters: list, hyperparameter_name: str, n_samples: int = 250) -> None:
    """
    Trains and evaluates a regression model with varying hyperparameters,
    and logs the results to a file as well as plot them.
    Args:
        model (str): The type of regression model to train (e.g., 'lasso', 'knn', etc).
        hyperparameters (list): A list of hyperparameter values to evaluate.
        hyperparameter_name (str): The name of the hyperparameter being varied.
        n_samples (int, optional): The number of samples to use for training. Defaults to 250.
    Returns:
        None
    """
    headers = [hyperparameter_name.capitalize(), "Expected Error",
               "Bias + Residual Error", "Variance"]
    results = []

    output_file = os.path.join(model, f"results_{timestamp}.txt")

    print(f"----------\n{model.capitalize()} Regression")
    for hyperparameter in hyperparameters:
        error, bias, variance = train_and_evaluate_models(
            n_samples, model_type=model, hyperparameter=hyperparameter)
        result = [hyperparameter, error, bias, variance]
        results.append(result)

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(
                f"Bootstrap Results: {hyperparameter_name}={hyperparameter}\n")
            f.write(tabulate([result], headers=headers, tablefmt="fancy_grid"))
            f.write("\n\n")

        print(
            f"Results for {hyperparameter_name}={hyperparameter} written to {output_file}")

    output_plot(model, hyperparameter_name, np.array(results))


if __name__ == "__main__":
    # Set variables and hyperparameters
    n_samples = 250  # fixed as per the statement

    alphas = []  # Lasso hyperparameters
    alphas = [10**exponent for exponent in range(-7, 0)] + \
             [2 * 10**exponent for exponent in range(-3, 0)] + \
             [5 * 10**exponent for exponent in range(-3, 0)]
    alphas.append(0.6)
    alphas.append(1)
    alphas.sort()

    ks = [1, 2, 3, 5, 7, 10, 20, 30]  # kNN hyperparameters
    max_depths = [1, 2, 3, 5, 7, 10, 20, 30]    # Decision Tree hyperparameters

    # Create models dictionary with {model_type: [hyperparameters, hyperparameter_name]}
    models = {
        "lasso": [alphas, "alpha"],
        "knn": [ks, "k"],
        "tree": [max_depths, "max_depth"]
    }

    # Run bootstrap sampling for each model type
    for model, hyperparameters in models.items():
        os.makedirs(model, exist_ok=True)
        run_train_eval(
            model, hyperparameters[0], hyperparameters[1], n_samples=n_samples)
