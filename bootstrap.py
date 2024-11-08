from data import load_wine_quality
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from tabulate import tabulate
from typing import Union
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

# Declare global variables
X, y = load_wine_quality()
random_state = np.random.RandomState(42)    # Create a RandomState instance
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")    # Generate a timestamp

"""
Procedure:

- For i = 1 to B:
	Take a bootstrap sample Bi from the data set.
	Learn a model fi on it.

- For each object, compute the expected error of all models that were built without it (about 30%). We thus use out-of-bag (OOB) samples (the ~30% of data not included in each bootstrap sample) as our test set.

- For each data point x that appears as an OOB sample across different bootstrap iterations:
	Variance:
	    > We calculate the variance of predictions made by the different models where x was OOB
	    > For each x, if f_i(x) is the prediction of model i, and f̄(x) is the mean prediction:
            Variance(x)= (1/N) * sum(fi(x)-f̄(x)**2,
                where N is the number of times x was OOB
	Bias:
    	> We compare the average prediction to the true value y(x):
            Bias²(x) = (f̄(x) - y(x))²

	Residual Error:
	    > The irreducible error can be estimated as the remaining error after accounting for bias and variance:
            Total Error = Bias² + Variance + Residual Error
"""


def bootstrap(n_samples: int, B: int, model_type: str, hyperparameter: Union[float, int]) -> tuple:
    """
    Perform bootstrap sampling and calculate bias, variance, and residual error.

    Parameters:
        n_samples (int): Number of samples to draw in each bootstrap sample.
        B (int): Number of bootstrap samples.
        model_type (str): Type of model to use.

            Available models are 'lasso', 'knn', and 'tree' (from sklearn).
            Else ValueError is raised.

        hyperparameter (float or int): Hyperparameter for the model (`alpha` for Lasso, `n_neighbors` for kNN, `max_depth` for Decision Tree).
    Returns:
        tuple: A tuple containing bias squared, variance, and residual error.
    Example:
        >>> bias_squared, variance, residual_error = bootstrap_sampling(100, 1000, 'lasso', 0.1)
        >>> print(f"Bias^2: {bias_squared}, Variance: {variance}, Residual Error: {residual_error}")
    """
    # Initialize arrays for all input samples
    y_pred = np.zeros((len(X), B))
    oob_counts = np.zeros(len(X)) # see below for explanation

    for i in range(B):
        # Get indices for bootstrap sample and perform sampling
        bootstrap_indices = resample(np.arange(len(X)), replace=True, n_samples=n_samples, random_state=random_state)

        X_resampled = X[bootstrap_indices]
        y_resampled = y[bootstrap_indices]

        # Model
        if model_type == 'lasso':
            model = Lasso(alpha=hyperparameter)
        elif model_type == 'knn':
            model = KNeighborsRegressor(n_neighbors=hyperparameter)
        elif model_type == 'tree':
            model = DecisionTreeRegressor(max_depth=hyperparameter)
        else:
            raise ValueError("Unsupported model type")

        model.fit(X_resampled, y_resampled)

        # Find OOB indices
        oob_indices = np.setdiff1d(np.arange(len(X)), bootstrap_indices)
        oob_counts[oob_indices] += 1    # tracks nb times a sample is OOB
        y_pred[oob_indices, i] = model.predict(X[oob_indices])

    # Statistics only for samples that were OOB > 0 times
    mask = oob_counts > 0
    oob_predictions = y_pred[mask]
    oob_counts = oob_counts[mask]
    true_values = y[mask]

    # Calculate metrics
    mean_predictions = np.sum(oob_predictions, axis=1) / oob_counts
        # below we use np.mean to get average variance across all samples
        # -> random var X and not individual variances of x's
    bias_squared = np.mean((mean_predictions - true_values) ** 2)
    variance = np.mean(np.var(oob_predictions, axis=1))
    residual_error = np.mean((true_values - mean_predictions) ** 2) - bias_squared - variance

    return bias_squared, variance, residual_error


def output_plot(model: str, hyperparameter_name: str, results: np.ndarray) -> None:
    """
    Output a plot of the results.

    Parameters:
        folder (str): Folder to save the plot in.
        hyperparameter_name (str): Name of the hyperparameter.
        results (np.ndarray): Results to plot.
    """
    hyperparameter_values = results[:, 0]
    bias_squared = results[:, 1]
    variance = results[:, 2]
    error = results[:, 1] + results[:, 2]

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(hyperparameter_values, bias_squared, 's-', label="Bias²", color="red")  # squares
    ax.plot(hyperparameter_values, variance, '^-', label="Variance", color="green")  # triangles
    ax.plot(hyperparameter_values, error, 'D-', label="Error", color="blue")  # diamonds

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


def run_bootstrap(model: str, hyperparameters: list, hyperparameter_name: str, B: int = 1000, n_samples: int=250) -> None:
    """
    Run bootstrap sampling for a given model type and hyperparameters, and save the results.

    Parameters:
        model (str): Type of model to use.
        hyperparameters (list): List of hyperparameter values to test.
        hyperparameter_name (str): Name of the hyperparameter.
        B (int): Number of bootstrap samples. Default is 1000.
        n_samples (int): Number of samples to draw in each bootstrap sample. Default is 250.
    """
    headers = [hyperparameter_name.capitalize(), "Bias^2", "Variance", "Residual Error"]
    results = []

    output_file = os.path.join(model, f"results_{timestamp}.txt")

    print(f"----------\n{model.capitalize()} Regression")
    for hyperparameter in hyperparameters:
        bias_squared, variance, residual_error = bootstrap(n_samples, B, model_type=model, hyperparameter=hyperparameter)
        result = [hyperparameter, bias_squared, variance, residual_error]
        results.append(result)

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"Bootstrap Results: {hyperparameter_name}={hyperparameter}\n")
            f.write(tabulate([result], headers=headers, tablefmt="fancy_grid"))
            f.write("\n\n")

        print(f"Results for {hyperparameter_name}={hyperparameter} written to {output_file}")

    output_plot(model, hyperparameter_name, np.array(results))


if __name__ == "__main__":

    # Set variables and hyperparameters
    B = 10000  # number of bootstrap samples
    n_samples = 250  # fixed as per the statement
    alphas = [0.1, 0.01, 0.001] # Lasso hyperparameters
    ks = [1, 5, 10] # kNN hyperparameters
    max_depths = [5, 10, 20]    # Decision Tree hyperparameters

    # Create models dictionary with (model_type: [hyperparameters, hyperparameter_name])
    models = {
        "lasso": [alphas, "alpha"],
        "knn": [ks, "k"],
        "tree": [max_depths, "max_depth"]
    }

    # Run bootstrap sampling for each model type
    for model, hyperparameters in models.items():
        os.makedirs(model, exist_ok=True)
        run_bootstrap(model, hyperparameters[0], hyperparameters[1], B=B, n_samples=n_samples)