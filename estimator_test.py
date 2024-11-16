import os
import numpy as np
from bootstrap import bootstrap
from tabulate import tabulate
import matplotlib.pyplot as plt
import datetime

# Global variables
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
n_estimators_range = [1, 5, 10, 20, 50, 100, 200]  # Number of estimators for ensemble methods
depth_range = [1, 2, 3, 5, 7, 10, 20, 30, None]  # Depth of the base learner (None = Fully Grown)
B = 1000  # Number of bootstrap samples


def output_plot_ensemble_results(method: str, depth: str, results: np.ndarray) -> None:
    """
    Generates and saves a plot of the results for Bagging and Boosting with different max_depth values.

    Parameters:
        method (str): Ensemble method ("bagging" or "boosting").
        depth (str): Depth of the base learner (e.g., "2", "5", "fully_grown").
        results (np.ndarray): Results to plot containing n_estimators, bias², variance, and total error.
    """
    n_estimators = results[:, 0]
    bias_squared = results[:, 1]
    variance = results[:, 2]
    total_error = results[:, 3]

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(n_estimators, bias_squared, 'o-', label="Bias²", color="red")
    ax.plot(n_estimators, variance, '^-', label="Variance", color="green")
    ax.plot(n_estimators, total_error, 's-', label="Total Error", color="blue")

    ax.set_xlabel("Number of Estimators (n_estimators)")
    ax.set_ylabel("Error")
    ax.set_title(f"{method.capitalize()} Reg. Bias-Variance Decomposition")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    folder = f"ensemble_analysis/{method}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"plot_{method}_depth_{depth}_{timestamp}.png"))
    plt.close()

    print(f"Plot for {method} regression saved in {folder}/")


def run_ensemble_analysis(method: str, n_estimators_range: list, depth_range: list) -> None:
    """
    Runs Bagging or Boosting bias-variance analysis for multiple max_depth values and saves the results.

    Parameters:
        method (str): Ensemble method ("bagging" or "boosting").
        n_estimators_range (list): List of n_estimators values to test.
        depth_range (list): List of max_depth values for the base learner.
    """
    headers = ["n_estimators", "Bias^2 + Residual Error", "Variance", "Expected Error"]

    for max_depth in depth_range:
        depth_label = "fully_grown" if max_depth is None else str(max_depth)
        output_file = os.path.join(f"ensemble_analysis/{method}", f"{method}_depth_{depth_label}_{timestamp}.txt")
        results = []

        print(f"----------\n{method.capitalize()} (Base Learner Depth: {depth_label})")
        for n_estimators in n_estimators_range:
            # Compute bias, variance, and total error
            bias_squared, variance, expected_error = bootstrap(
                n_samples=250,
                B=B,
                model_type=method,
                hyperparameter=max_depth,
                n_estimators=n_estimators,
            )
            result = [n_estimators, bias_squared, variance, expected_error]
            results.append(result)

            # Write results to a text file
            os.makedirs(f"ensemble_analysis/{method}", exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{method.capitalize()} Results: Depth={depth_label}, n_estimators={n_estimators}\n")
                f.write(tabulate([result], headers=headers, tablefmt="fancy_grid"))
                f.write("\n\n")

        # Convert results to NumPy array and save plot
        results = np.array(results)
        output_plot_ensemble_results(method, depth_label, results)


if __name__ == "__main__":
    os.makedirs("ensemble_analysis", exist_ok=True)

    # Run analysis for Bagging
    run_ensemble_analysis("bagging", n_estimators_range, depth_range)

    # Run analysis for Boosting
    run_ensemble_analysis("boosting", n_estimators_range, depth_range)
