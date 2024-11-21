import os
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import datetime
from bias_variance_analysis import train_and_evaluate_models

# Global variables
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
n_estimators_range = [1, 5, 10, 20, 50, 100, 200]  # Number of estimators for ensemble methods
depth_range = [1, 2, 3, 5, 7, 10, 20, 30, None]  # Depth of the base learner (None = Fully Grown)


def run_ensemble_analysis(method: str, n_estimators_range: list, depth_range: list) -> None:
    """
    Runs Bagging or Boosting bias-variance analysis for multiple max_depth values and saves the results.

    Parameters:
        method (str): Ensemble method ("bagging" or "boosting").
        n_estimators_range (list): List of n_estimators values to test.
        depth_range (list): List of max_depth values for the base learner.
    """
    headers = ["n_estimators","Expected Error" , "Bias + Residual Error", "Variance"]
    depth_results = {}
    output_file = os.path.join(f"ensemble_analysis", f"combined_analysis_{timestamp}.txt")

    # Create output directory if it doesn't exist
    os.makedirs("ensemble_analysis", exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n===== {method.upper()} Analysis =====\n")

        for max_depth in depth_range:
            depth_label = "fully_grown" if max_depth is None else str(max_depth)
            results = []

            f.write(f"\n--- {method.capitalize()} (Base Learner Depth: {depth_label}) ---\n")
            print(f"----------\n{method.capitalize()} (Base Learner Depth: {depth_label})")

            for n_estimators in n_estimators_range:
                # Compute bias, variance, and total error using train_and_evaluate_models
                expected_error, avg_bias, avg_variance = train_and_evaluate_models(
                    n_samples=250,
                    model_type=method,
                    hyperparameter=max_depth,
                    n_estimators=n_estimators,
                )
                result = [n_estimators, expected_error, avg_bias, avg_variance]
                results.append(result)

            # Write results for the current depth
            f.write(tabulate(results, headers=headers, tablefmt="fancy_grid"))
            f.write("\n\n")

        print(f"Results saved to {output_file}")

        # Convert results to NumPy array and save plot
        results = np.array(results)
        # Store results for this depth
        depth_results[depth_label] = np.array(results)

        plot_depth_comparison_simple(method, depth_results)


def plot_depth_comparison_simple(method: str, depth_results: dict) -> None:
    """
    Plots comparisons of Bias + residual error, Variance, and Total Error across different depths,
    with a separate plot for each metric.

    Parameters:
        method (str): Ensemble method ("bagging" or "boosting").
        depth_results (dict): Dictionary containing results for each depth.
                              Keys are depths (str), values are NumPy arrays of results.
    """
    # Prepare figure
    metrics = ["Bias + residual error", "Variance", "Total Error"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 3 side-by-side plots

    for depth_label, results in depth_results.items():
        n_estimators = results[:, 0]
        total_error = results[:, 1]
        bias = results[:, 2]
        variance = results[:, 3]

        # Plot each metric on its respective subplot
        axes[0].plot(n_estimators, bias, label=f"Depth {depth_label}", marker='o')
        axes[1].plot(n_estimators, variance, label=f"Depth {depth_label}", marker='^')
        axes[2].plot(n_estimators, total_error, label=f"Depth {depth_label}", marker='s')

    # Configure plots
    for i, ax in enumerate(axes):
        ax.set_xlabel("Number of Estimators")
        ax.set_ylabel(metrics[i])
        ax.set_title(f"{metrics[i]} Comparison")
        ax.legend(loc="best")

    # Save the plots
    folder = f"ensemble_analysis/{method}"
    os.makedirs(folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"depth_comparison_{timestamp}.png"))
    plt.close()
    print(f"Comparison plot for {method} saved in {folder}/depth_comparison_{timestamp}.png")


if __name__ == "__main__":
    os.makedirs("ensemble_analysis", exist_ok=True)

    # Run analysis for Bagging
    run_ensemble_analysis("bagging", n_estimators_range, depth_range)

    # Run analysis for Boosting
    run_ensemble_analysis("boosting", n_estimators_range, depth_range)
