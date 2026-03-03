import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from scipy.stats import norm

def plot_parameter_evolution(bo, iteration_idx):
    param_names = ['a', 'b', 'mu']

    iter_data = bo.history[iteration_idx]
    gp_chains = iter_data['gp_chains']

    colors = ['blue', 'orange']  # one color per chain

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for param_idx, param_name in enumerate(param_names):
        ax = axes[param_idx]

        for chain_idx, gp in enumerate(gp_chains):
            color = colors[chain_idx]

            samples = np.asarray(gp.gp_par[param_name]).ravel()

            sns.histplot(
                samples,
                bins=30,
                stat='density',
                alpha=0.4,
                color=color,
                ax=ax
            )

            sns.kdeplot(
                samples,
                color=color,
                linewidth=2,
                ax=ax,
                label=f'Chain {chain_idx + 1}'
            )

        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f'Parameter Distributions – Iteration {iteration_idx}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig, axes


def visualize_gp_1d(bo, iteration_idx=None, grid_size=1000):
    """Plot GP predictions and acquisition function for one or all iterations.

    Args:
        bo: Bayesian optimization object
        iteration_idx: Specific iteration to plot (int), or None to plot all iterations
        grid_size: Number of points for the grid
    """

    # If iteration_idx is None, plot all iterations in a 2x2 grid
    if iteration_idx is None:
        n_iterations = len(bo.history)

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        # Plot each iteration
        for idx in range(min(n_iterations, 4)):
            _plot_single_iteration(bo, idx, grid_size, axes[idx])

        # Hide unused subplots if fewer than 4 iterations
        for idx in range(n_iterations, 4):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()
        return fig, axes

    # Otherwise, plot single iteration
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        _plot_single_iteration(bo, iteration_idx, grid_size, ax)
        plt.tight_layout()
        plt.show()
        return fig, ax


def _plot_single_iteration(bo, iteration_idx, grid_size, ax):
    """Helper function to plot a single iteration on a given axis."""

    # Resolve iteration index
    true_idx = iteration_idx if iteration_idx >= 0 else len(bo.history) - 1

    # Get iteration data
    iteration_data = bo.history[true_idx]
    gp = iteration_data['gp_chains'][-1]  # Use last chain
    y_max = gp.y_max_original

    # Create grid
    x_grid = np.linspace(bo.search_space[0][0], bo.search_space[0][1], grid_size)
    x_points = x_grid.reshape(-1, 1)

    # Get predictions in NORMALIZED space
    predictions_normalized = gp.predict(x_points, normalized_out=True)

    # ==================== ACQUISITION FUNCTION ====================
    # Compute acquisition using normalized predictions
    acq_name = bo.acq

    if acq_name == "emp_rb":
        mean_norm = np.mean(predictions_normalized, axis=1)
        lower_norm = np.percentile(predictions_normalized, 5, axis=1)
        upper_norm = np.percentile(predictions_normalized, 95, axis=1)
        range_norm = upper_norm - lower_norm
        beta = 0.1 * np.log(len(gp.y_obs))
        acquisition = mean_norm + np.sqrt(beta) * range_norm

    elif acq_name == "emp_MES":
        mean_norm = np.mean(predictions_normalized, axis=1)
        lower_norm = np.percentile(predictions_normalized, 5, axis=1)
        upper_norm = np.percentile(predictions_normalized, 95, axis=1)
        range_norm = upper_norm - lower_norm
        range_safe = np.maximum(range_norm, 1e-8)
        sqrt_beta = np.min((gp.y_max - mean_norm) / range_safe)
        acquisition = mean_norm + sqrt_beta * range_norm

    elif acq_name == "emp_MES_quant":
        q50_norm = np.percentile(predictions_normalized, 50, axis=1)
        lower_norm = np.percentile(predictions_normalized, 5, axis=1)
        upper_norm = np.percentile(predictions_normalized, 95, axis=1)
        range_norm = upper_norm - lower_norm
        range_safe = np.maximum(range_norm, 1e-8)
        sqrt_beta = np.min((gp.y_max - q50_norm) / range_safe)
        acquisition = q50_norm + sqrt_beta * range_norm

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

    # ==================== PLOTTING STATISTICS ====================
    # Always plot median and 95% credible interval (2.5-97.5 percentiles)

    # Compute quantiles on normalized predictions
    median_norm = np.percentile(predictions_normalized, 50, axis=1)
    lower_norm = np.percentile(predictions_normalized, 2.5, axis=1)
    upper_norm = np.percentile(predictions_normalized, 97.5, axis=1)

    # Unnormalize for plotting
    gp_median = median_norm * gp.y_std + gp.y_mean
    gp_lower = lower_norm * gp.y_std + gp.y_mean
    gp_upper = upper_norm * gp.y_std + gp.y_mean

    # ==================== TRUE FUNCTION & OBSERVATIONS ====================
    # True function values
    true_values = np.array([bo.function.func(point) * bo.function.ismax for point in x_points]).flatten()

    # Observed points
    observed_x = iteration_data['X'].flatten()
    observed_y = iteration_data['y'].flatten()

    # ==================== PLOTTING SETUP ====================
    # Determine plotting range
    y_min_main = min(np.min(true_values), np.min(gp_lower))
    y_max_main = y_max
    y_range_main = y_max_main - y_min_main

    # Scale acquisition function for "mountain view"
    acq_scaled = (acquisition - np.min(acquisition)) / (np.max(acquisition) - np.min(acquisition))
    acq_height = 0.2 * y_range_main
    acq_baseline = y_min_main - 0.35 * y_range_main
    acq_shifted = acq_baseline + acq_scaled * acq_height

    # ==================== PLOTTING ====================
    # True function
    ax.plot(x_grid, true_values, 'k-', linewidth=2, alpha=0.7, label='True Function')

    # GP predictions: median and 95% credible interval
    ax.plot(x_grid, gp_median, 'b-', linewidth=2, label='GP Median')
    ax.fill_between(x_grid, gp_lower, gp_upper, color='blue', alpha=0.2, label='95% Credible Interval')

    # Observed points
    ax.scatter(observed_x, observed_y, c='red', s=100, alpha=1.0,
               label='Observed Points', edgecolors='black', linewidths=1.5, zorder=5)

    # y_max line
    ax.axhline(y_max, color='red', linestyle='--', linewidth=2, label=f'y_max = {y_max:.3f}')

    # Acquisition function
    ax.fill_between(x_grid, acq_baseline, acq_shifted, color='green', alpha=0.15, label='Acquisition Function')
    ax.plot(x_grid, acq_shifted, 'g-', linewidth=1.5)

    # Maximum of acquisition function
    max_acq_idx = np.argmax(acquisition)
    max_acq_x = x_grid[max_acq_idx]
    max_acq_shifted_val = acq_shifted[max_acq_idx]
    ax.scatter([max_acq_x], [max_acq_shifted_val], c='darkgreen', s=200, marker='*',
               edgecolors='black', linewidths=2, label=f'Max AF at x={max_acq_x:.3f}', zorder=5)

    # Last observed point (if not first iteration)
    if true_idx > 0:
        new_point_x = observed_x[-1]
        new_point_y = observed_y[-1]
        ax.scatter([new_point_x], [new_point_y], c='orange', s=150, marker='D',
                   edgecolors='black', linewidths=2, label=f'Last Observed (x={new_point_x:.3f})', zorder=6)
        ax.plot([new_point_x, new_point_x], [new_point_y, acq_baseline], 'orange', linestyle='--', linewidth=2,
                alpha=0.7)

    # Vertical lines for previous observed points
    for obs_x in observed_x[:-1] if true_idx > 0 else observed_x:
        ax.axvline(obs_x, color='gray', linestyle=':', alpha=0.3)

    # Formatting
    ax.set_ylim(acq_baseline - acq_height * 0.1, y_max_main + 0.05 * y_range_main)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'GP Predictions & Acquisition Function - Iteration {true_idx}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)



