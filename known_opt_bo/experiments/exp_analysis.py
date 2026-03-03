import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import seaborn as sns
import json
import math
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from known_opt_bo.test_functions import functions


def plot_legend(save_fig=False, results_dir="result"):

    fixed_style = dict(linestyle="-", linewidth=2, marker="o", markersize=5)

    distinct_colors = [
        "#e6194b",  # red
        "#3cb44b",  # green
        "#4363d8",  # blue
        "#f58231",  # orange
        "#911eb4",  # purple
        "#42d4f4",  # cyan
        "#f032e6",  # magenta
        "#bfef45",  # lime
        "#fabed4",  # pink
        "#469990",  # teal
        "#dcbeff",  # lavender
        "#9A6324",  # brown
    ]

    algo_style_map = {
        "TGQP+MES50": dict(**fixed_style, color=distinct_colors[0]),
        "TGQP+MES75": dict(**fixed_style, color=distinct_colors[1]),
        "GP+TEI": dict(**fixed_style, color=distinct_colors[2]),
        "Random": dict(**fixed_style, color=distinct_colors[3]),
        "SlogGP(boundary)+SlogEI": dict(**fixed_style, color=distinct_colors[4]),
        "SlogGP(boundary)+SlogTEI": dict(**fixed_style, color=distinct_colors[5]),
        "SlogGP(fixedbound)+SlogEI": dict(**fixed_style, color=distinct_colors[6]),
        "SlogGP+SlogEI": dict(**fixed_style, color=distinct_colors[7]),
        "SlogGP+SlogTEI": dict(**fixed_style, color=distinct_colors[8]),
        "transformedGP+ERM": dict(**fixed_style, color=distinct_colors[9]),
        "GP+EI": dict(**fixed_style, color=distinct_colors[10]),
        "GP+MES": dict(**fixed_style, color=distinct_colors[11]),
    }

    ncol = 6  # 4 per row → 3 rows for 12 algos, adjust to taste

    handles = [
        plt.Line2D([0], [0], label=algo, **style)
        for algo, style in algo_style_map.items()
    ]

    n_rows = math.ceil(len(handles) / ncol)
    fig, ax = plt.subplots(figsize=(ncol * 3.5, n_rows * 0.6))
    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        fontsize=16,
        frameon=False,
        handlelength=2.3,
        handleheight=1.5,
        columnspacing=2.0,
        labelspacing=0.2,
    )

    plt.tight_layout()

    if save_fig:
        plots_dir = Path(results_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig_path = plots_dir / "legend.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Legend saved to {fig_path}")

    plt.show()

def mean_regret(
    information,
    results_dir="result",
    save_fig=True,
    fig_name=None,
    exclude_suffixes=None,
):
    # -----------------------------
    # Setup
    # -----------------------------
    results_path = Path(results_dir)
    func_name = information["name"]
    fstar = -information["fstar"]

    if exclude_suffixes is None:
        exclude_suffixes = ["_boundaryValue", "_varianceValue"]

    fixed_style = dict(linestyle="-", linewidth=2, marker="o", markersize=5)

    distinct_colors = [
        "#e6194b",  # red
        "#3cb44b",  # green
        "#4363d8",  # blue
        "#f58231",  # orange
        "#911eb4",  # purple
        "#42d4f4",  # cyan
        "#f032e6",  # magenta
        "#bfef45",  # lime
        "#fabed4",  # pink
        "#469990",  # teal
        "#dcbeff",  # lavender
        "#9A6324",  # brown
    ]

    algo_style_map = {
        "TGQP+MES50":  dict(**fixed_style, color=distinct_colors[0]),
        "TGQP+MES75":  dict(**fixed_style, color=distinct_colors[1]),
        "GP+TEI":  dict(**fixed_style, color=distinct_colors[2]),
        "Random":  dict(**fixed_style, color=distinct_colors[3]),
        "SlogGP(boundary)+SlogEI":  dict(**fixed_style, color=distinct_colors[4]),
        "SlogGP(boundary)+SlogTEI":  dict(**fixed_style, color=distinct_colors[5]),
        "SlogGP(fixedbound)+SlogEI":  dict(**fixed_style, color=distinct_colors[6]),
        "SlogGP+SlogEI":  dict(**fixed_style, color=distinct_colors[7]),
        "SlogGP+SlogTEI":  dict(**fixed_style, color=distinct_colors[8]),
        "transformedGP+ERM": dict(**fixed_style, color=distinct_colors[9]),
        "GP+EI": dict(**fixed_style, color=distinct_colors[10]),
        "GP+MES": dict(**fixed_style, color=distinct_colors[11]),
    }

    # -----------------------------
    # Find and filter result files
    # -----------------------------
    pattern = f"{func_name}_*"
    result_files = list(results_path.glob(pattern))

    result_files = [
        f for f in result_files
        if not any(suffix in f.name for suffix in exclude_suffixes)
    ]

    if not result_files:
        print(f"No valid result files found for '{func_name}'")
        return

    print(f"Found {len(result_files)} algorithm(s) for '{func_name}'")

    # -----------------------------
    # Load data & compute regret
    # -----------------------------
    regret_stats = {}

    for file_path in result_files:
        algo_name = file_path.stem[len(func_name) + 1:]

        try:
            data = np.loadtxt(file_path, delimiter=",")
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        regrets = data - fstar

        regret_stats[algo_name] = {
            "mean": np.mean(regrets, axis=0),
            "n_experiments": regrets.shape[0],
            "n_iterations": regrets.shape[1],
        }

        print(
            f"  - {algo_name}: "
            f"{regrets.shape[0]} experiments, "
            f"{regrets.shape[1]} iterations"
        )

    if not regret_stats:
        print("No usable regret data.")
        return

    # -----------------------------
    # Plot (no legend)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, stats in regret_stats.items():
        iters = np.arange(1, stats["n_iterations"] + 1)

        if algo in algo_style_map:
            style = algo_style_map[algo]
        else:
            print(f"  Warning: '{algo}' not found in algo_style_map, skipping.")
            continue

        ax.plot(iters, stats["mean"], **style)  # no label

    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Mean Regret", fontsize=22)
    ax.set_title(func_name, fontsize=25)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, alpha=0.6)

    plt.tight_layout()

    # -----------------------------
    # Save figure & metadata
    # -----------------------------
    if save_fig:
        plots_dir = results_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        if fig_name is None:
            fig_name = f"{func_name}_mean_regret.png"

        fig_path = plots_dir / fig_name
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {fig_path}")

        meta_path = fig_path.with_suffix(".txt")
        with open(meta_path, "w") as f:
            f.write("Mean Regret Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Function: {func_name}\n")
            f.write(f"Optimal value (f*): {fstar}\n")
            f.write(f"Experiments per algorithm: {regret_stats[next(iter(regret_stats))]['n_experiments']}\n")
            f.write(f"Iterations: {regret_stats[next(iter(regret_stats))]['n_iterations']}\n\n")
            f.write("Algorithms:\n")
            for algo in regret_stats:
                f.write(f"  - {algo}\n")
            f.write("\nExcluded diagnostic suffixes:\n")
            for s in exclude_suffixes:
                f.write(f"  - {s}\n")
            f.write(f"\nGenerated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Metadata saved to {meta_path}")

    plt.show()


def u_comparison(
        res_num,
        results_dir="u_comparison_results",
        save_fig=True,
        fig_name=None,
):

    # -----------------------------
    # Setup
    # -----------------------------
    results_path = Path(results_dir) / f"res_{res_num}"

    if not results_path.exists():
        print(f"Result folder '{results_path}' does not exist!")
        return

    # -----------------------------
    # Find parameter files
    # -----------------------------
    pattern = "*_u*_*.csv"
    result_files = list(results_path.glob(pattern))

    if not result_files:
        print(f"No parameter files found in res_{res_num}")
        return

    # Detect function name from first file
    func_name = result_files[0].stem.split('_u')[0]

    # Group files by u value
    u_params = {}  # {u_value: {'a': data, 'b': data, 'mu': data}}

    for file_path in result_files:
        parts = file_path.stem.split('_')
        u_part = [p for p in parts if p.startswith('u')][0]
        u_value = int(u_part[1:])
        param_name = parts[-1]  # 'a', 'b', or 'mu'

        if u_value not in u_params:
            u_params[u_value] = {}

        try:
            data = np.loadtxt(file_path, delimiter=",")
            u_params[u_value][param_name] = data.ravel()
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
            continue

    if not u_params:
        print("No usable parameter data.")
        return

    # -----------------------------
    # Plot
    # -----------------------------
    param_names = ['a', 'b', 'mu']
    param_labels = [r'$\sigma^2$', r'$l^2$', r'$\mu$']

    u_values = sorted(u_params.keys())
    if len(u_values) > 2:
        print(f"Warning: found {len(u_values)} n values, using first 2.")
        u_values = u_values[:2]

    colors = sns.color_palette("husl", 2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 16))

    for param_idx, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
        ax = axes[param_idx]

        for u_idx, u_value in enumerate(u_values):
            if param_name not in u_params[u_value]:
                continue

            color = colors[u_idx]
            samples = u_params[u_value][param_name]

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
                linewidth=2.5,
                ax=ax,
                label=f'$n={u_value}$'
            )

        ax.set_xlabel(param_label, fontsize=20)
        ax.set_ylabel('Density', fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=18, frameon=True)

    fig.suptitle(
        'Parameter Posterior Distributions',
        fontsize=22
    )
    plt.tight_layout(h_pad=4.0, rect=[0, 0.02, 1, 0.97])

    # -----------------------------
    # Save figure & metadata
    # -----------------------------
    if save_fig:
        plots_dir = results_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        if fig_name is None:
            fig_name = f"{func_name}_u_comparison.png"

        fig_path = plots_dir / fig_name
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {fig_path}")

        meta_path = fig_path.with_suffix(".txt")
        with open(meta_path, "w") as f:
            f.write("N-Value Comparison Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Function: {func_name}\n")
            f.write(f"Result set: res_{res_num}\n")
            f.write(f"N values compared: {u_values}\n\n")
            f.write("Parameters compared:\n")
            for label in param_labels:
                f.write(f"  - {label}\n")
            f.write(
                f"\nGenerated on: "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

        print(f"Metadata saved to {meta_path}")

    plt.show()


def plot_functions(
    functions_list,
    results_dir="../test_functions",
    save_fig=True,
    fig_name=None,
    n_points=1000,
):
    import math
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    n_functions = len(functions_list)

    if n_functions == 0:
        print("No functions provided!")
        return

    # -----------------------------
    # Determine grid layout (max 3 per row)
    # -----------------------------
    n_cols = min(3, n_functions)
    n_rows = math.ceil(n_functions / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 6 * n_rows)
    )

    # Flatten axes for uniform indexing
    axes = np.array(axes).reshape(-1)

    # -----------------------------
    # Plot functions
    # -----------------------------
    for idx, func in enumerate(functions_list):
        ax = axes[idx]

        func_name = func.name
        fstar = func.fstar * func.ismax

        # Handle bounds
        if hasattr(func, 'bounds_dict') and isinstance(func.bounds_dict, dict):
            bounds_keys = list(func.bounds_dict.keys())
            bounds_values = list(func.bounds_dict.values())
        else:
            bounds_values = [
                (func.bounds[0, i].item(), func.bounds[1, i].item())
                for i in range(func.dim)
            ]
            bounds_keys = [f'x{i}' for i in range(func.dim)]

        # Only plot 1D functions
        if func.input_dim == 1 or func.dim == 1:
            x_min, x_max = bounds_values[0]
            x = np.linspace(x_min, x_max, n_points).reshape(-1, 1)

            y = func.func(x) * func.ismax

            ax.plot(x, y, linewidth=2, color='blue')
            ax.axhline(
                y=fstar,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Optimum: {fstar:.4f}'
            )

            if idx >= n_functions - n_cols:
                ax.set_xlabel(bounds_keys[0], fontsize=24)
            if idx % n_cols == 0:
                ax.set_ylabel('f(x)', fontsize=24)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_title(
                f'{func_name.capitalize()} | '
                f'{bounds_keys[0]} ∈ [{x_min}, {x_max}]',
                fontsize=22
            )

            ax.legend(fontsize=22, loc='lower left')
            ax.grid(True, alpha=0.3)

        else:
            ax.text(
                0.5,
                0.5,
                f'{func_name.capitalize()}\n'
                f'{func.input_dim}D function\n'
                f'Bounds: {bounds_values}\n'
                f'Optimum: {fstar:.4f}',
                ha='center',
                va='center',
                fontsize=11,
                transform=ax.transAxes
            )
            ax.set_title(
                f'{func_name.capitalize()} ({func.input_dim}D)',
                fontsize=12
            )
            ax.axis('off')

    # Hide unused subplots
    for ax in axes[n_functions:]:
        ax.axis('off')

    plt.tight_layout(h_pad=7.0, w_pad=3.0, rect=[0.02, 0.02, 0.98, 0.98])

    # -----------------------------
    # Save figure & metadata
    # -----------------------------
    if save_fig:
        results_path = Path(results_dir)
        plots_dir = results_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if fig_name is None:
            func_names = '_'.join([f.name for f in functions_list])
            fig_name = f"functions_{func_names}.png"

        fig_path = plots_dir / fig_name
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {fig_path}")

        meta_path = fig_path.with_suffix(".txt")
        with open(meta_path, "w") as f:
            f.write("Test Functions Visualization\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of functions: {n_functions}\n\n")

            for func in functions_list:
                f.write(f"Function: {func.name}\n")
                f.write(f"  Dimension: {func.input_dim}\n")
                f.write(f"  Optimum value: {func.fstar * func.ismax:.6f}\n")
                if hasattr(func, 'bounds_dict'):
                    f.write(f"  Bounds: {func.bounds_dict}\n")
                else:
                    f.write(f"  Bounds: {func.bounds.tolist()}\n")
                f.write("\n")

            f.write(
                f"Generated on: "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

        print(f"Metadata saved to {meta_path}")

    plt.show()


def plot_bo_progress(exp_num, results_dir="experiments/bo_progress_res"):

    results_path = Path(results_dir) / exp_num

    # Find all iteration directories
    iter_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("iteration_")],
                       key=lambda x: int(x.name.split("_")[1]))

    n_iterations = len(iter_dirs)

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    # Plot each iteration
    for idx, iter_dir in enumerate(iter_dirs[:4]):  # Only plot first 4
        ax = axes[idx]

        # Load data
        x_grid = np.load(iter_dir / "x_grid.npy")
        gp_median = np.load(iter_dir / "gp_median.npy")
        gp_lower = np.load(iter_dir / "gp_lower.npy")
        gp_upper = np.load(iter_dir / "gp_upper.npy")
        true_y = np.load(iter_dir / "true_y.npy")
        acq_values = np.load(iter_dir / "acq_values.npy")
        observed_x = np.load(iter_dir / "observed_X.npy")
        observed_y = np.load(iter_dir / "observed_y.npy")

        with open(iter_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        y_max = metadata["y_max_original"]

        # Plotting setup
        y_min_main = min(np.min(true_y), np.min(gp_lower))
        y_max_main = y_max
        y_range_main = y_max_main - y_min_main

        # Scale acquisition function
        acq_scaled = (acq_values - np.min(acq_values)) / (np.max(acq_values) - np.min(acq_values))
        acq_height = 0.2 * y_range_main
        acq_baseline = y_min_main - 0.35 * y_range_main
        acq_shifted = acq_baseline + acq_scaled * acq_height

        # Plot
        ax.plot(x_grid, true_y, 'k-', linewidth=2, alpha=0.7, label='True Function')
        ax.plot(x_grid, gp_median, 'b-', linewidth=2, label='GP Median')
        ax.fill_between(x_grid, gp_lower, gp_upper, color='blue', alpha=0.2, label='99% Credible Interval')
        ax.scatter(observed_x, observed_y, c='red', s=100, alpha=1.0,
                   label='Observed Points', edgecolors='black', linewidths=1.5, zorder=5)
        ax.axhline(y_max, color='red', linestyle='--', linewidth=2, label=f'y_max = {y_max:.3f}')
        ax.fill_between(x_grid, acq_baseline, acq_shifted, color='green', alpha=0.15, label='Acquisition Function')
        ax.plot(x_grid, acq_shifted, 'g-', linewidth=1.5)

        # Maximum of acquisition function
        max_acq_idx = np.argmax(acq_values)
        max_acq_x = x_grid[max_acq_idx]
        max_acq_shifted_val = acq_shifted[max_acq_idx]
        ax.scatter([max_acq_x], [max_acq_shifted_val], c='darkgreen', s=200, marker='*',
                   edgecolors='black', linewidths=2, label=f'Max AF at x={max_acq_x:.3f}', zorder=5)

        # Last observed point (if not first iteration)
        if idx > 0:
            new_point_x = observed_x[-1]
            new_point_y = observed_y[-1]
            ax.scatter([new_point_x], [new_point_y], c='orange', s=150, marker='D',
                       edgecolors='black', linewidths=2, label=f'Last Observed', zorder=6)
            ax.plot([new_point_x, new_point_x], [new_point_y, acq_baseline], 'orange', linestyle='--', linewidth=2,
                    alpha=0.7)

        # Vertical lines for previous observed points
        for obs_x in observed_x[:-1] if idx > 0 else observed_x:
            ax.axvline(obs_x, color='gray', linestyle=':', alpha=0.3)

        # Formatting
        ax.set_ylim(acq_baseline - acq_height * 0.1, y_max_main + 0.05 * y_range_main)
        if idx >= 2:  # only bottom row gets x label
            ax.set_xlabel('x', fontsize=25)
        if idx % 2 == 0:  # only left column gets y label
            ax.set_ylabel('y', fontsize=25)
        ax.tick_params(axis='both', labelsize=22)
        ax.set_title(f'Iteration {idx}', fontsize=25)
        # ax.legend(fontsize=12, loc='lower left')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_iterations, 4):
        axes[idx].axis('off')

    # Manual spacing — no tight_layout so it doesn't override these values
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.08, hspace=0.28, wspace=0.1)

    # # Create standalone legend figure
    # fig_legend, ax_legend = plt.subplots(figsize=(16, 3.5))
    # ax_legend.axis('off')
    #
    # legend_elements = [
    #     Line2D([0], [0], color='black', linewidth=3, alpha=0.7, label='True Function'),
    #     Line2D([0], [0], color='blue', linewidth=3, label='GP Median'),
    #     Patch(facecolor='blue', alpha=0.2, label='99% Credible Interval'),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=18,
    #            markeredgecolor='black', label='Observed Points'),
    #     Line2D([0], [0], color='red', linestyle='--', linewidth=3, label='$y_{max}$'),
    #     Patch(facecolor='green', alpha=0.15, label='Acquisition Function'),
    #     Line2D([0], [0], marker='*', color='w', markerfacecolor='darkgreen', markersize=24,
    #            markeredgecolor='black', label='Max Acquisition Function'),
    #     Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', markersize=18,
    #            markeredgecolor='black', label='Last Observed Point'),
    # ]
    #
    # ax_legend.legend(handles=legend_elements, loc='center', ncol=4,
    #                  fontsize=20, frameon=True, framealpha=0.5,
    #                  handlelength=2.5, handleheight=2, handletextpad=0.8, labelspacing=1.2)
    # fig_legend.tight_layout()

    plt.show()

    return fig, axes


if __name__ == "__main__":

    # plot_legend(save_fig=False)

    # ### Plot mean regret for a given experiment ###
    # information = {
    #     'name': 'Forrester',  # Function name
    #     'fstar': 6.0207
    # }
    # mean_regret(information,
    #             results_dir=r'C:\Users\Amitl\PycharmProjects\tgqp_bo\known_opt_bo\experiments\result',
    #             save_fig=True)

    ### Plot parameter distributions per u for a given experiment ###
    u_comparison(res_num=72,
                 results_dir=r'C:\Users\Amitl\PycharmProjects\tgqp_bo\known_opt_bo\experiments\u_comparison_results',
                 save_fig=True)
    #
    # ### Plot BO progression
    # plot_bo_progress(
    #     exp_num="forrester_20260207_111205",
    #     results_dir=r'C:\Users\Amitl\PycharmProjects\tgqp_bo\known_opt_bo\experiments\bo_progress_res'
    # )

    # functions_list = [
    #     functions.Forrester(negate=False),
    #     functions.fourier(negate=False),
    #     functions.Levy(negate=False),
    #     functions.MultiModal2(negate=False),
    #     functions.MultiModal7(negate=False),
    #     functions.MultiModal14(negate=False),
    #     functions.MultiModal15(negate=False),
    #     # add more as needed
    # ]
    #
    # plot_functions(
    #     functions_list=functions_list,
    #     save_fig=False,  # set True if you want to save
    # )
