from known_opt_bo.test_functions import functions
from bo import BOKnownOpt
from bo_viz import *
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import norm

print(f"Current working directory: {Path.cwd()}")
def save_prep(bo, experiment_name=None, grid_size=1000, save_dir = "known_opt_bo/experiments/bo_progress_res"):
    """
    Prepare and save BO results for later visualization without needing the BO/GP objects.

    Args:
        bo: Bayesian optimization object with history
        experiment_name: Name for this experiment (default: function_name_timestamp)
        grid_size: Number of points for the prediction grid
        save_dir: Base directory for saving results

    Returns:
        Path to the saved experiment directory
    """

    # Create experiment directory
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{bo.function.name}_{timestamp}"

    exp_dir = Path(save_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving BO results to: {exp_dir}")
    print(f"Processing {len(bo.history)} iterations...")

    # Create grid
    x_grid = np.linspace(bo.search_space[0][0], bo.search_space[0][1], grid_size)
    x_points = x_grid.reshape(-1, 1)

    # Compute true function values once (same for all iterations)
    true_values = np.array([bo.function.func(point) * bo.function.ismax for point in x_points]).flatten()

    # Process each iteration
    for iter_idx in range(len(bo.history)):
        print(f"  Processing iteration {iter_idx}...")

        # Create iteration directory
        iter_dir = exp_dir / f"iteration_{iter_idx}"
        iter_dir.mkdir(exist_ok=True)

        # Get iteration data
        iteration_data = bo.history[iter_idx]
        gp = iteration_data['gp_chains'][-1]  # Use last chain

        # Get observed points
        observed_x = iteration_data['X'].flatten()
        observed_y = iteration_data['y'].flatten()

        # ==================== GP Predictions ====================
        # Get predictions in NORMALIZED space
        predictions_normalized = gp.predict(x_points, normalized_out=True)

        # ==================== Acquisition Function ====================
        # Compute acquisition using normalized predictions
        acq_name = bo.acq

        if acq_name == "emp_rb":
            mean_normalized = np.mean(predictions_normalized, axis=1)
            lower_normalized = np.percentile(predictions_normalized, 5, axis=1)
            upper_normalized = np.percentile(predictions_normalized, 95, axis=1)
            range_normalized = upper_normalized - lower_normalized
            beta = 0.1 * np.log(len(gp.y_obs))
            acquisition = mean_normalized + np.sqrt(beta) * range_normalized

        elif acq_name == "emp_MES":
            mean_normalized = np.mean(predictions_normalized, axis=1)
            lower_normalized = np.percentile(predictions_normalized, 5, axis=1)
            upper_normalized = np.percentile(predictions_normalized, 95, axis=1)
            range_normalized = upper_normalized - lower_normalized
            range_safe = np.maximum(range_normalized, 1e-8)
            sqrt_beta = np.min((gp.y_max - mean_normalized) / range_safe)
            acquisition = mean_normalized + sqrt_beta * range_normalized

        elif acq_name == "emp_MES_quant":
            q50_normalized = np.percentile(predictions_normalized, 50, axis=1)
            lower_normalized = np.percentile(predictions_normalized, 5, axis=1)
            upper_normalized = np.percentile(predictions_normalized, 95, axis=1)
            range_normalized = upper_normalized - lower_normalized
            range_safe = np.maximum(range_normalized, 1e-8)
            sqrt_beta = np.min((gp.y_max - q50_normalized) / range_safe)
            acquisition = q50_normalized + sqrt_beta * range_normalized

        elif acq_name == "TrueMES":
            mu_h = np.percentile(predictions_normalized, 50, axis=1)
            q75 = np.percentile(predictions_normalized, 75, axis=1)
            q25 = np.percentile(predictions_normalized, 25, axis=1)
            sigma_h = np.clip((q75 - q25) / 1.349, 1e-9, None)
            gamma = (gp.y_max - mu_h) / sigma_h
            pdf_g = norm.pdf(gamma)
            cdf_g = np.clip(norm.cdf(gamma), 1e-12, 1.0)
            acquisition = (gamma * pdf_g) / (2 * cdf_g) - np.log(cdf_g)

        elif acq_name == "TrueMES75":
            mu_h = np.percentile(predictions_normalized, 50, axis=1)
            q75 = np.percentile(predictions_normalized, 87.5, axis=1)
            q25 = np.percentile(predictions_normalized, 12.5, axis=1)
            sigma_h = np.clip((q75 - q25) / 2.3006, 1e-9, None)
            gamma = (gp.y_max - mu_h) / sigma_h
            pdf_g = norm.pdf(gamma)
            cdf_g = np.clip(norm.cdf(gamma), 1e-12, 1.0)
            acquisition = (gamma * pdf_g) / (2 * cdf_g) - np.log(cdf_g)

        else:
            raise ValueError(f"Unknown acquisition function: {acq_name}")

        # ==================== Compute and Unnormalize Quantiles ====================
        # Compute quantiles on normalized predictions
        q50_norm = np.percentile(predictions_normalized, 50, axis=1)
        q2_5_norm = np.percentile(predictions_normalized, 0.5, axis=1)
        q97_5_norm = np.percentile(predictions_normalized, 99.5, axis=1)

        # Unnormalize only these 3 quantiles
        gp_median = q50_norm * gp.y_std + gp.y_mean
        gp_lower = q2_5_norm * gp.y_std + gp.y_mean
        gp_upper = q97_5_norm * gp.y_std + gp.y_mean

        # ==================== Save Data ====================
        # Save grid and predictions
        np.save(iter_dir / "x_grid.npy", x_grid)
        np.save(iter_dir / "gp_median.npy", gp_median)
        np.save(iter_dir / "gp_lower.npy", gp_lower)
        np.save(iter_dir / "gp_upper.npy", gp_upper)
        np.save(iter_dir / "true_y.npy", true_values)
        np.save(iter_dir / "acq_values.npy", acquisition)

        # Save observations
        np.save(iter_dir / "observed_X.npy", observed_x)
        np.save(iter_dir / "observed_y.npy", observed_y)

        # Save metadata
        metadata = {
            "iteration": iter_idx,
            "acq_name": acq_name,
            "function_name": bo.function.name,
            "y_max_original": float(gp.y_max_original),
            "search_space": [[float(bo.search_space[0][0]), float(bo.search_space[0][1])]],
            "n_observations": len(observed_x),
            "grid_size": grid_size,
        }

        with open(iter_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\n✓ Successfully saved all {len(bo.history)} iterations to {exp_dir}")
    return exp_dir


### Function to model/optimize
myfunction = functions.Forrester()

### BO parameters
iterations = 4
acq = 'TrueMES'         # optimize using which acq function ('emp_ucb', 'emp_rb)?
iter_size = 5000       # how many posterior samples needed to construct each GP?
chains = 1             # choose 2 for convergence check via plots of the GP parameters posterior

### GP parameters
gp_kernel = 'RBF'           # which kernel should the GP use ('RBF', 'Matern', 'exp')
kernel_jitter = 1e-6
init_MALA_step = 1e-7      # 0.00005 # MALA scales

sigma_priors = [100, 100]   # [alpha, beta] prior scales for the kernel parameters
bounds_a = [1e-3, 40000]
bounds_b = [1e-3, 400]

p = 50                      # will be at least (2*observations + 1)
# R = [
#     [1, 0],
#     [0, 1],
#     [1, 1]
# ]

### experiment settings
n_init_points = 4
np.random.seed(556)


lower_bounds = [b[0] for b in myfunction.bounds_dict.values()]
upper_bounds = [b[1] for b in myfunction.bounds_dict.values()]
# init_x = np.random.uniform(low=lower_bounds, high=upper_bounds,
#                                          size=(n_init_points, myfunction.input_dim))

init_x = np.array([0.02, 0.92, 0.08, 0.8]).reshape(-1, 1)
# init_x = np.array([[3], [5], [10]])
bo = BOKnownOpt(init_x, myfunction, acq, iter_size, chains, gp_kernel, kernel_jitter, init_MALA_step, sigma_priors, p, bounds_a, bounds_b)

# Then in your main code:
for iter in range(iterations):
    print(f"\n{'=' * 60}")
    print(f"Starting iteration {iter}")
    print(f"{'=' * 60}")

    bo.select_next_point()

    # Visualize after iteration completes
    print(f"\nVisualizing iteration {iter}...")
    plot_parameter_evolution(bo, iteration_idx=iter)
    visualize_gp_1d(bo, iteration_idx=iter)

visualize_gp_1d(bo)


answer = input("\nDo you want to save these results? (y/n): ")
if answer.lower() == 'y':
    exp_path = save_prep(bo)
    print(f"Results saved!")





