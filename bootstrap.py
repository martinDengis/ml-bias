from data import load_wine_quality
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from tabulate import tabulate
import os
import numpy as np


X, y = load_wine_quality()

def bootstrap(n_samples, B, model_type, hyperparameter):
    y_pred = np.zeros((n_samples, B))
    oob_counts = np.zeros(n_samples)

    for i in range(B):
        # Generate a bootstrap sample
        X_resampled, y_resampled = resample(X, y, replace=True, n_samples=n_samples, random_state=random_state)
        model = Lasso(alpha=alpha)
        model.fit(X_resampled, y_resampled)

        oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(resample(np.arange(n_samples), replace=True, random_state=random_state)))
        oob_counts[oob_indices] += 1    # tracks nb times a sample is OOB
        y_pred[oob_indices, i] = model.predict(X[oob_indices])

    # We calculate bias, variance, and residual error
    oob_predictions = y_pred[oob_counts > 0]
    oob_counts = oob_counts[oob_counts > 0]

    # Calculate metrics
    mean_predictions = np.sum(oob_predictions, axis=1) / oob_counts
    bias_squared = np.mean((mean_predictions - true_values) ** 2)
    variance = np.mean(np.var(oob_predictions, axis=1))
    residual_error = np.mean((true_values - mean_predictions) ** 2) - bias_squared - variance

    mean_predictions = np.sum(oob_predictions, axis=1) / oob_counts
    true_values = y[oob_counts > 0]


    variance = np.mean(np.var(oob_predictions, axis=1))
    residual_error = np.mean((true_values - mean_predictions) ** 2) - bias_squared - variance

    return bias_squared, variance, residual_error


if __name__ == "__main__":
    n_samples = 250 # fixed as per the statement

    # Lasso
    B = 1000  # Number of bootstrap samples
    alpha = [0.1, 0.01, 0.001] # Hyper-parameter
    headers = ["Alpha", "Bias^2", "Variance", "Residual Error", "Total Error"]

    if not os.path.exists("lasso"):
        os.makedirs("lasso")

    print("----------\n1. Lasso Regression")
    for a in alpha:
        bias_squared, variance, residual_error = bootstrap(n_samples, B, model_type='lasso', hyperparameter=a)
        results = [a, bias_squared, variance, residual_error, bias_squared + variance + residual_error]

        output_file = os.path.join("lasso", f"results_alpha_{a}.txt")

        # Write the results to the file
        with open(output_file, "w") as f:
            f.write("Bootstrap Results: alpha=" + str(a) + "\n")
            f.write(tabulate([results], headers=headers, tablefmt="fancy_grid"))

        print(f"Results for alpha={a} written to {output_file}")

    # kNN
    B = 1000  # Number of bootstrap samples
    k = [1, 5, 10] # Hyper-parameter
    headers = ["K", "Bias^2", "Variance", "Residual Error", "Total Error"]

    print("----------\n2. kNN Regression")
    for n in k:
        bias_squared, variance, residual_error = bootstrap(n_samples, B, model_type='knn', hyperparameter=n)
        results = [n, bias_squared, variance, residual_error, bias_squared + variance + residual_error]
        print()
        print("Bootstrap Results: k=", n)
        print(tabulate([results], headers=headers, tablefmt="fancy_grid"))

    # Decision Tree
    B = 1000  # Number of bootstrap samples
    max_depth = [5, 10, 20] # Hyper-parameter
    headers = ["Max Depth", "Bias^2", "Variance", "Residual Error", "Total Error"]

    print("----------\n3. Decision Tree Regression")
    for d in max_depth:
        bias_squared, variance, residual_error = bootstrap(n_samples, B, model_type='tree', hyperparameter=d)
        results = [d, bias_squared, variance, residual_error, bias_squared + variance + residual_error]
        print()
        print("Bootstrap Results: max_depth=", d)
        print(tabulate([results], headers=headers, tablefmt="fancy_grid"))