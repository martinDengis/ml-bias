from bootstrap import run_bootstrap
from data import load_wine_quality
import datetime
import numpy as np
import os

if __name__ == "__main__":
    # Set variables and hyperparameters
    B = 1000  # number of bootstrap samples
    n_samples = 250  # fixed as per the statement

    alphas = []  # Lasso hyperparameters
    alphas = [10**exponent for exponent in range(-3, 0)] + \
             [2 * 10**exponent for exponent in range(-3, 0)] + \
             [5 * 10**exponent for exponent in range(-3, 0)]
    alphas.append(1)
    alphas.append(2)
    alphas.append(0.0000000001)
    alphas.sort()
    # alphas = [1, 5, 10, 15]

    ks = [30]  # kNN hyperparameters
    # Decision Tree hyperparameters
    max_depths = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]

    # Create models dictionary with {model_type: [hyperparameters, hyperparameter_name]}
    models = {
        "lasso": [alphas, "alpha"],
        # "knn": [ks, "k"],   # takes a very long time to run, might want to comment it out
        # "tree": [max_depths, "max_depth"]
    }

    # Run bootstrap sampling for each model type
    for model, hyperparameters in models.items():
        os.makedirs(model, exist_ok=True)
        run_bootstrap(
            model, hyperparameters[0], hyperparameters[1], B=B, n_samples=n_samples)
