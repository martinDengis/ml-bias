from data import load_wine_quality
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from tabulate import tabulate
import datetime
import numpy as np
import os

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


def bootstrap(n_samples, B, model_type, hyperparameter):
    # Initialize arrays for all input samples
    y_pred = np.zeros((len(X), B))
    oob_counts = np.zeros(len(X))

    for i in range(B):
        # Generate indices for bootstrap sample
        bootstrap_indices = resample(np.arange(len(X)), replace=True, n_samples=n_samples, random_state=random_state)

        # bootstrap sample
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

    # We calculate statistics only for samples that were OOB > 0 times
    mask = oob_counts > 0
    oob_predictions = y_pred[mask]
    oob_counts = oob_counts[mask]
    true_values = y[mask]

    # Calculate metrics
    mean_predictions = np.sum(oob_predictions, axis=1) / oob_counts
    bias_squared = np.mean((mean_predictions - true_values) ** 2)
    variance = np.mean(np.var(oob_predictions, axis=1))
    residual_error = np.mean((true_values - mean_predictions) ** 2) - bias_squared - variance

    return bias_squared, variance, residual_error


if __name__ == "__main__":
    folders = ["lasso", "knn", "tree"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    n_samples = 250 # fixed as per the statement

    # --------------------
    # Lasso
    # --------------------
    B = 1000  # Number of bootstrap samples
    alpha = [0.1, 0.01, 0.001] # Hyper-parameter
    headers = ["Alpha", "Bias^2", "Variance", "Residual Error", "Total Error"]

    output_file = os.path.join("lasso", f"results_{timestamp}.txt")

    print("----------\n1. Lasso Regression")
    for a in alpha:
        bias_squared, variance, residual_error = bootstrap(n_samples, B, model_type='lasso', hyperparameter=a)
        results = [a, bias_squared, variance, residual_error, bias_squared + variance + residual_error]

        with open(output_file, "a", encoding="utf-8") as f:
            f.write("Bootstrap Results: alpha=" + str(a) + "\n")
            f.write(tabulate([results], headers=headers, tablefmt="fancy_grid"))
            f.write("\n\n")

        print(f"Results for alpha={a} written to {output_file}")

    # --------------------
    # kNN
    # --------------------
    B = 1000  # Number of bootstrap samples
    k = [1, 5, 10] # Hyper-parameter
    headers = ["K", "Bias^2", "Variance", "Residual Error", "Total Error"]

    output_file = os.path.join("knn", f"results_{timestamp}.txt")

    print("----------\n2. kNN Regression")
    for n in k:
        bias_squared, variance, residual_error = bootstrap(n_samples, B, model_type='knn', hyperparameter=n)
        results = [n, bias_squared, variance, residual_error, bias_squared + variance + residual_error]

        with open(output_file, "a", encoding="utf-8") as f:
            f.write("Bootstrap Results: k=" + str(n) + "\n")
            f.write(tabulate([results], headers=headers, tablefmt="fancy_grid"))
            f.write("\n\n")

        print(f"Results for k={n} written to {output_file}")

    # --------------------
    # Decision Tree
    # --------------------
    B = 1000  # Number of bootstrap samples
    max_depth = [5, 10, 20] # Hyper-parameter
    headers = ["Max Depth", "Bias^2", "Variance", "Residual Error", "Total Error"]

    output_file = os.path.join("tree", f"results_{timestamp}.txt")

    print("----------\n3. Decision Tree Regression")
    for d in max_depth:
        bias_squared, variance, residual_error = bootstrap(n_samples, B, model_type='tree', hyperparameter=d)
        results = [d, bias_squared, variance, residual_error, bias_squared + variance + residual_error]

        with open(output_file, "a", encoding="utf-8") as f:
            f.write("Bootstrap Results: max_depth=" + str(d) + "\n")
            f.write(tabulate([results], headers=headers, tablefmt="fancy_grid"))
            f.write("\n\n")

        print(f"Results for max_depth={d} written to {output_file}")