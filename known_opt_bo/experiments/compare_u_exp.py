# experiments/u_comparison.py
import os
from pathlib import Path
import numpy as np
from known_opt_bo.test_functions import functions
from known_opt_gp.truncated_gp import TGQP
from known_bound.utlis import get_initial_points
import torch

repo_name = "u_comparison_results"

if not os.path.exists(repo_name):
    os.makedirs(repo_name)
    print(f"Directory '{repo_name}' created.")
else:
    print(f"Directory '{repo_name}' already exists.")

# Find next available result folder number
existing_folders = [d for d in os.listdir(repo_name) if
                    d.startswith('res_') and os.path.isdir(os.path.join(repo_name, d))]
if existing_folders:
    numbers = [int(f.split('_')[1]) for f in existing_folders if f.split('_')[1].isdigit()]
    next_num = max(numbers) + 1 if numbers else 1
else:
    next_num = 1

# Create new result folder
result_folder = os.path.join(repo_name, f'res_{next_num}')
os.makedirs(result_folder)
print(f"Created result folder: {result_folder}")

my_function = functions.MultiModal14(negate=False)
bounds = my_function.bounds
n_init = 5
seed = 3814

iter_size = 30000
gp_kernel = 'RBF'
kernel_jitter = 1e-6
init_MALA_step = 1e-6

sigma_priors = [3, 100]
bounds_a = [1e-3, 40000]
bounds_b = [1e-6, 400]
p1 = 40
p2 = 100
p_to_compare = [p1, p2]

# x = get_initial_points(bounds, n_init, "cpu", torch.double, seed=seed)
x = torch.tensor([[0.2], [0.24], [0.45], [3.5], [3.1]])
print('X: ', x)
y = my_function.func(x) * my_function.ismax
print('y: ', y)
y_max = my_function.fstar * my_function.ismax
if isinstance(my_function.bounds_dict, dict):
    search_space = []
    for key in list(my_function.bounds_dict.keys()):
        search_space.append(my_function.bounds_dict[key])
    search_space = np.asarray(search_space)
else:
    search_space = np.asarray(my_function.bounds)

for p in p_to_compare:
    np.random.seed(1234)
    gp = TGQP(x, y, y_max, search_space, gp_kernel, kernel_jitter, iter_size, init_MALA_step, sigma_priors, p,
                   bounds_a, bounds_b)
    gp.fit()

    # Save parameters right after fitting
    func_name = my_function.name
    np.savetxt(f'{result_folder}/{func_name}_u{p}_a.csv', gp.gp_par['a'], delimiter=',')
    np.savetxt(f'{result_folder}/{func_name}_u{p}_b.csv', gp.gp_par['b'], delimiter=',')
    np.savetxt(f'{result_folder}/{func_name}_u{p}_mu.csv', gp.gp_par['mu'], delimiter=',')

    print(f"Saved parameters for u={p}")

print("\nAll results saved!")