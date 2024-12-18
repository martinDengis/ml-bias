import os
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import datetime
from bias_variance_analysis import train_and_evaluate_models

# Define sample sizes for the experiment
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")  # Generate a timestamp
sample_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 1500, 2000]
models = {
    "lasso": {"hyperparameter": 0.02, "label": "Alpha"},
    "knn": {"hyperparameter": 20, "label": "k"},
    "tree": {"hyperparameter": 2, "label": "Max Depth (Fixed)"},
    "tree_fully_grown": {"hyperparameter": None, "label": "Fully Grown"},
}

def output_plot_sample_size(model: str, results: np.ndarray) -> None:
    """
    Generates and saves a plot of the results for sample size testing.

    Parameters:
        model (str): Type of model ("lasso", "knn", "tree").
        results (np.ndarray): Results to plot containing sample sizes, bias + residual error, variance, and expected error.
    """
    sample_sizes = results[:, 0]
    expected_error = results[:, 1]
    bias = results[:, 2]
    variance = results[:, 3]

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(sample_sizes, bias, 'o-', label="Bias + residual error", color="red")  # Circles for bias + residual error
    ax.plot(sample_sizes, variance, '^-', label="Variance", color="green")  # Triangles for variance
    ax.plot(sample_sizes, expected_error, 's-', label="Total Error", color="blue")  # Squares for total error

    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Error")
    ax.set_title(f"{model.capitalize()} Reg. Bias-Variance Decomposition")
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the plot
    folder = "sample_size"
    plt.savefig(os.path.join(folder, f"plot_{model}_sample_size_{timestamp}.png"))
    plt.close()

    print(f"Plot for {model} regression saved in {folder}/")

def run_sample_size_tests(model: str, sample_sizes: list, hyperparameter: float, hyperparameter_name: str) -> None:
    """
    Runs sample size testing for a given model type and saves the results.

    Parameters:
        model (str): Type of model ("lasso", "knn", "tree").
        sample_sizes (list): List of sample sizes to test.
        hyperparameter (float or int): Hyperparameter for the model (e.g., alpha for Lasso, k for kNN, max_depth for Decision Tree).
        hyperparameter_name (str): Name of the hyperparameter.
    """
    headers = ["Sample Size", "Expected Error" , "Bias + Residual Error", "Variance"]
    results = []

    output_file = os.path.join("sample_size", f"{model}_results_{timestamp}.txt")
    os.makedirs("sample_size", exist_ok=True)

    print(f"----------\n{model.capitalize()} Regression Sample Size")
    for n_samples in sample_sizes:
        # Use train_and_evaluate_models to calculate bias, variance, and total error
        expected_error, avg_bias, avg_variance = train_and_evaluate_models(
            n_samples=n_samples,
            model_type=model,
            hyperparameter=hyperparameter
        )
        result = [n_samples,expected_error, avg_bias, avg_variance]
        results.append(result)

    # Write results to a text file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"Sample Size Results: \n")
        f.write(tabulate(results, headers=headers, tablefmt="fancy_grid"))
        f.write("\n\n")

    print(f"Results for sample size={n_samples} written to {output_file}")

    # Convert results to a NumPy array for plotting
    results = np.array(results)
    output_plot_sample_size(model, results)

if __name__ == "__main__":
    # Run sample size tests for each model
    for model, params in models.items():
        run_sample_size_tests(
            model=model,
            sample_sizes=sample_sizes,
            hyperparameter=params["hyperparameter"],
            hyperparameter_name=params["label"]
        )
