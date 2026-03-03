import numpy as np
from . import epm as epf
import torch
from torch.autograd.functional import hessian
import scipy.optimize
# from harmonic_hmc import run_harmonic_hmc
import rpy2.robjects as robjects
from .harmonic_hmc import run_harmonic_hmc
# import GPy
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt



def save_get_h_inputs(y_obs, y_max, Sigma_inv_par, Sigma_par, T1, T2, A, mu_dims,
                      filepath='debug_data/get_h_inputs.npz'):
    """
    Save all inputs to get_h() function for debugging.

    Call this at the beginning of get_h() like:
    save_get_h_inputs(y_obs, y_max, Sigma_inv_par, Sigma_par, T1, T2, A, mu_dims)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save all arrays
    np.savez(filepath,
             y_obs=y_obs,
             y_max=y_max,
             Sigma_inv_par=Sigma_inv_par,
             Sigma_par=Sigma_par,
             T1=T1,
             T2=T2,
             A=A,
             mu_dims=mu_dims)

    print(f"Debug inputs saved to {filepath}")


def get_u_1d(search_space, p, init_X):
    L, U = search_space[0]
    init_X = np.array(init_X).flatten()
    n1 = len(init_X)
    n2 = p - n1

    sort_order = np.argsort(init_X)
    sorted_X = init_X[sort_order]

    points_list = list(sorted_X)
    n2_remaining = n2

    L_observed = np.any(np.isclose(sorted_X, L))
    U_observed = np.any(np.isclose(sorted_X, U))

    if not L_observed:
        points_list.append(L)
        n2_remaining -= 1

    if not U_observed:
        points_list.append(U)
        n2_remaining -= 1

    points_list.sort()
    fixed_points = np.array(points_list)

    segments = []

    if not L_observed:
        segments.append({
            'start': L,
            'end': sorted_X[0],
            'length': sorted_X[0] - L,
            'min_points': 0
        })

    for i in range(len(sorted_X) - 1):
        segments.append({
            'start': sorted_X[i],
            'end': sorted_X[i + 1],
            'length': sorted_X[i + 1] - sorted_X[i],
            'min_points': 1
        })

    if not U_observed:
        segments.append({
            'start': sorted_X[-1],
            'end': U,
            'length': U - sorted_X[-1],
            'min_points': 0
        })

    total_length = sum(seg['length'] for seg in segments)

    min_required = sum(seg['min_points'] for seg in segments)
    remaining_after_min = n2_remaining - min_required

    for seg in segments:
        if total_length > 0:
            seg['allocated'] = seg['min_points'] + int(np.round(remaining_after_min * seg['length'] / total_length))
        else:
            seg['allocated'] = seg['min_points']

    total_allocated = sum(seg['allocated'] for seg in segments)
    diff = n2_remaining - total_allocated

    if diff != 0 and len(segments) > 0:
        largest_seg = max(segments, key=lambda s: s['length'])
        largest_seg['allocated'] += diff

    new_points = []
    for seg in segments:
        if seg['allocated'] > 0:
            pts = np.linspace(seg['start'], seg['end'], seg['allocated'] + 2)[1:-1]
            new_points.extend(pts)

    u = np.concatenate([fixed_points, new_points])
    u = np.sort(u)

    obs_idx = []
    for orig_idx in range(n1):
        val = init_X[orig_idx]
        idx = np.where(np.isclose(u, val))[0][0]
        obs_idx.append(idx)

    return u.reshape(-1, 1), obs_idx









class TGQP:
    def __init__(self, init_x, init_y, y_max, search_space, gp_kernel, kernel_jitter, iter_size, MALA_step, sigma_priors, p, bounds_a, bounds_b):
        self.x_obs_original = init_x
        x_scaled = (init_x - search_space[:, 0]) / (search_space[:, 1] - search_space[:, 0])
        self.x_obs = x_scaled - 0.5
        self.search_space = search_space
        self.search_space_centered = np.array([[-0.5, 0.5]] * search_space.shape[0])

        # self.y_obs = init_y
        self.y_max_original = y_max

        self.y_mean = np.mean(init_y)
        self.y_std = np.std(init_y)
        self.y_obs = (init_y - self.y_mean) / self.y_std
        self.y_max = (y_max - self.y_mean) / self.y_std

        self.gp_kernel = gp_kernel
        self.kernel_jitter = kernel_jitter
        self.iter_size = iter_size
        self.MALA_step = MALA_step
        self.sigma_priors = np.array([100, 100]) if sigma_priors is None else np.array(sigma_priors)
        self.p = max(2 * len(self.x_obs) + 1, p)
        self.gp_par = {
            "a": np.zeros((self.iter_size, 1)),
            "b": np.zeros((self.iter_size, 1)),
            "mu": np.zeros((self.iter_size, 1))
        }
        self.Xi_samples = np.zeros((self.iter_size, self.p))
        self.u, self.x_obs_idx_in_u = get_u_1d(self.search_space_centered, self.p, self.x_obs)

        self.ordered_u = self.u.copy()
        self.c_SIGMA = None
        self.xdiff2_torch = None
        self.A = None
        self.acceptance_rate = None
        F_eta = -1 * np.eye(self.p)
        g_eta = np.full(self.p, self.y_max)
        self.F_eta_r = robjects.r['matrix'](
            robjects.FloatVector(F_eta.flatten(order='F')),
            nrow=F_eta.shape[0],
            ncol=F_eta.shape[1]
        )
        self.g_eta_r = robjects.FloatVector(g_eta)

        F_Xi2 = -1 * np.eye(self.p - len(self.y_obs))
        g_Xi2 = np.full(self.p - len(self.y_obs), self.y_max)
        self.F_Xi2_r = robjects.r['matrix'](
            robjects.FloatVector(F_Xi2.flatten(order='F')),
            nrow=F_Xi2.shape[0],
            ncol=F_Xi2.shape[1]
        )
        self.g_Xi2_r = robjects.FloatVector(g_Xi2)
        self.lb_a, self.ub_a = bounds_a
        self.lb_b, self.ub_b = bounds_b

    # @profile
    def fit(self):
        burn_in = 5000
        adapt_mode = True
        W = 200
        self.reset_var()
        iter_counter = 0
        adaptor_counter = 0
        accepted_counter = 0
        self.unorder_u()
        if self.gp_kernel == 'RBF':
            self.c_SIGMA = epf.getM_RBF(self.u, self.kernel_jitter)
        elif self.gp_kernel == 'Matern':
            self.c_SIGMA = epf.getM_Matern(self.u, self.directions, 2)
        self.xdiff2_torch = torch.tensor(self.c_SIGMA.xdiff2, dtype=torch.float64)
        init_par = self.get_init_par()
        par = self.find_map(init_par)
        self.A = self.compute_precond_matrix(par)
        print('initial par is: ', par)
        SIGMA_inv_par = self.c_SIGMA.M(par[0], par[1])
        init_eta = np.full(self.p, par[2])

        init_Xi2 = init_eta[len(self.y_obs):]
        Xi2 = self.sample_Xi2(SIGMA_inv_par, par[2], init_Xi2).reshape(-1, 1)
        init_Xi2 = Xi2
        Xi_com = np.concatenate((self.y_obs, Xi2), axis=0)
        # self.debug_mu_curvature(par[2], Xi_com, SIGMA_inv_par)

        un_ll_par = self.get_unnormalized_loglik(Xi_com, par[2], SIGMA_inv_par)
        k = 0
        while iter_counter < self.iter_size + burn_in:
            par_prime, logq_par, logq_par_prime, trans_par_prime, trans_par = self.get_MALA_prop(Xi_com, par,
                                                                                                 self.MALA_step)
            SIGMA_inv_prime = self.c_SIGMA.M(par_prime[0], par_prime[1])
            un_ll_par_prime = self.get_unnormalized_loglik(Xi_com, par_prime[2], SIGMA_inv_prime)
            eta = self.sample_eta(SIGMA_inv_prime, par_prime[2], init_eta).reshape(-1, 1)
            init_eta = eta

            un_ll_eta_par = self.get_unnormalized_loglik(eta, par[2], SIGMA_inv_par)
            un_ll_eta_prime = self.get_unnormalized_loglik(eta, par_prime[2], SIGMA_inv_prime)

            log_prior_par_prime = self.get_logprior(trans_par_prime)
            log_prior_par = self.get_logprior(trans_par)

            log_r = (un_ll_par_prime + log_prior_par_prime + logq_par + un_ll_eta_par
                     - un_ll_par - log_prior_par - logq_par_prime - un_ll_eta_prime)

            if np.log(np.random.rand()) < min(0, log_r):
                accepted_counter += 1
                par = par_prime
                SIGMA_inv_par = SIGMA_inv_prime
            iter_counter += 1

            if iter_counter <= burn_in:
                adaptor_counter += 1
                if iter_counter % W == 0:
                    k += 1
                    self.update_step(accepted_counter/adaptor_counter, k)
                    # print('Acceptance rate was: ', accepted_counter/adaptor_counter)
                    # print('New step is: ', self.MALA_step)
                    accepted_counter = 0
                    adaptor_counter = 0

            Xi2 = self.sample_Xi2(SIGMA_inv_par, par[2], init_Xi2).reshape(-1, 1)
            init_Xi2 = Xi2
            Xi_com = np.concatenate((self.y_obs, Xi2))

            if iter_counter > burn_in:
                self.gp_par["a"][iter_counter - burn_in - 1, :] = par[0]
                self.gp_par["b"][iter_counter - burn_in - 1, :] = par[1]
                self.gp_par["mu"][iter_counter - burn_in - 1, :] = par[2]
                self.Xi_samples[iter_counter - burn_in - 1, :] = Xi_com.ravel()
                # iter_counter += 1

            # self.communicate(iter_counter, adaptor_counter, adapt_mode, 1000)

            un_ll_par = self.get_unnormalized_loglik(Xi_com, par[2], SIGMA_inv_par)

        self.ar = accepted_counter / (iter_counter - burn_in)
        print('############################ Acceptance rate was: ', self.ar)
        self.order_Xi()


    def predict(self, locations, kind='linear', normalized_out=False):
        locations = np.array(locations)
        single_input = locations.ndim == 1
        if single_input:
            locations = locations.reshape(1, -1)

        x_scaled = (locations - self.search_space[:, 0]) / (self.search_space[:, 1] - self.search_space[:, 0])
        x_scaled_centered = x_scaled - 0.5

        n_locations = x_scaled_centered.shape[0]
        k = self.Xi_samples.shape[0]  # number of MCMC samples
        predictions = np.zeros((n_locations, k))

        for i in range(n_locations):
            interpolator = interp1d(self.ordered_u[:, 0], self.Xi_samples, kind=kind, axis=1,
                                    bounds_error=False, fill_value="extrapolate")
            interpolated_values = interpolator(x_scaled_centered[i]).T  # shape (k,)
            predictions[i] = interpolated_values

        if not normalized_out:
            predictions = predictions * self.y_std + self.y_mean
        return predictions[0] if single_input else predictions

    def get_unnormalized_loglik(self, y, mu, SIGMA_inv_par):
        diff = y - mu
        un_ll = -0.5 * diff.T @ SIGMA_inv_par @ diff
        return un_ll

    def get_logprior(self, trans_par):
        log_prior_par = (
                - (trans_par[0] ** 2) / (2 * self.sigma_priors[0])  # α: sum over dims
                - (trans_par[1] ** 2) / (2 * self.sigma_priors[1])  # β:
        )
        return log_prior_par

    def sample_eta(self, SIGMA_inv_prime, mu, init_val):
        mu_full = np.repeat(mu, self.p)
        M_chol = scipy.linalg.cholesky(SIGMA_inv_prime, lower=False)
        init_r = robjects.FloatVector(init_val)
        mean_r = robjects.FloatVector(mu_full)
        cholesky_r = robjects.r['matrix'](robjects.FloatVector(M_chol.flatten(order='F')),
                                          nrow=M_chol.shape[0], ncol=M_chol.shape[1])

        try:
            y = run_harmonic_hmc(nSample=100, mean=mean_r, choleskyFactor=cholesky_r,
                                 constrainDirec=self.F_eta_r, constrainBound=self.g_eta_r,
                                 init=init_r, precFlg=True)
        except:
            print('mean: ', mu_full)
            print('init: ', init_val)
            print('g_Xi2_r: ', self.g_eta_r)
            print('SIGMA_inv_prime: ', SIGMA_inv_prime)
            print('M_chol: ', M_chol)

        return np.array(y)[-1, :]


    def sample_Xi2(self, M, mu, init_val):
        n1 = self.y_obs.shape[0]
        mu_full = np.repeat(mu, self.p).reshape(-1, 1)
        M22 = M[n1:, n1:]
        M21 = M[n1:, :n1]
        r = M22 @ mu_full[n1:] - M21 @ (self.y_obs - mu_full[:n1])
        mean = np.linalg.solve(M22, r)
        M_y_hat_chol = scipy.linalg.cholesky(M22, lower=False)
        cholesky_r = robjects.r['matrix'](
            robjects.FloatVector(M_y_hat_chol.flatten(order='F')), nrow=M_y_hat_chol.shape[0],
            ncol=M_y_hat_chol.shape[1]
        )
        mean_r = robjects.FloatVector(mean)
        init_r = robjects.FloatVector(init_val)

        try:
            Xi2 = run_harmonic_hmc(nSample=1, mean=mean_r, choleskyFactor=cholesky_r, constrainDirec=self.F_Xi2_r, constrainBound=self.g_Xi2_r,
                                   init=init_r, precFlg=True)
        except:
            print('mean: ', mean)
            print('init: ', init_val)
            print('g_Xi2_r: ', self.g_Xi2_r)
            print('M22: ', M22)
            print('M_y_hat_chol: ', M_y_hat_chol)
        return np.array(Xi2)

    def find_map(self, init_par):
        a_init = torch.tensor(init_par[0], dtype=torch.float64)
        b_init = torch.tensor(init_par[1], dtype=torch.float64)
        mu_init = torch.tensor(init_par[2], dtype=torch.float64)

        # Stack parameters into array format
        par_init = torch.stack([a_init, b_init, mu_init])
        trans_init = self.par_to_trans(par_init)

        alpha = torch.nn.Parameter(trans_init[0].clone().detach().requires_grad_(True))
        beta = torch.nn.Parameter(trans_init[1].clone().detach().requires_grad_(True))
        mu = torch.nn.Parameter(trans_init[2].clone().detach().requires_grad_(True))

        optimizer = torch.optim.Adam([alpha, beta, mu], lr=0.1)
        num_iterations = 20000
        final_loss_value = None
        tol = 1e-3
        patience = 500
        best_loss = float('inf')
        best_iter = 0

        y = torch.tensor(self.y_obs, dtype=torch.float64).flatten()

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Stack transformed parameters and convert back
            trans_current = torch.stack([alpha, beta, mu])
            par_current = self.trans_to_par(trans_current)
            a = par_current[0]
            b = par_current[1]
            mu_par = par_current[2]

            K = a * torch.exp(-0.5 * self.xdiff2_torch / b)
            covariance = K + self.kernel_jitter * torch.eye(K.shape[0], dtype=torch.float64)
            n1 = len(self.y_obs)
            covariance_obs = covariance[:n1, :n1]
            mu_vec = mu_par.expand(n1)
            likelihood = torch.distributions.MultivariateNormal(mu_vec, covariance_matrix=covariance_obs)
            log_like = likelihood.log_prob(y)

            log_prior_alpha = -0.5 * (alpha ** 2 / self.sigma_priors[0] ** 2)
            log_prior_beta = -0.5 * (beta ** 2 / self.sigma_priors[1] ** 2)

            loss = -(log_like + log_prior_alpha + log_prior_beta)
            loss_value = loss.item()
            final_loss_value = loss_value

            loss.backward()
            optimizer.step()

            if loss_value + tol < best_loss:
                best_loss = loss_value
                best_iter = iteration
            elif iteration - best_iter > patience:
                print(f"Early stopping at iteration {iteration}, best loss {best_loss:.6f}")
                break

            if (iteration + 1) % 2000 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss_value:.6f}")
        print(f"Final loss: {final_loss_value:.6f}")

        with torch.no_grad():
            trans_final = torch.stack([alpha, beta, mu])
            par_final = self.trans_to_par(trans_final)
            a_opt = par_final[0]
            b_opt = par_final[1]
            mu_opt = par_final[2]

        res = np.array([a_opt, b_opt, mu_opt])

        return res

    def compute_precond_matrix(self, par):
        a = torch.tensor(par[0], dtype=torch.float64)
        b = torch.tensor(par[1], dtype=torch.float64)
        mu = torch.tensor(par[2], dtype=torch.float64)

        # Stack into [3] tensor for par_to_trans
        par_stack = torch.stack([a, b, mu])
        trans_stack = self.par_to_trans(par_stack)

        # Use transformed parameters directly for Hessian
        params_flat = trans_stack.requires_grad_(True)

        def neg_log_post_func(params):
            # Extract transformed parameters
            alpha = params[0]
            beta = params[1]
            mu_param = params[2]

            # Stack back into [3] tensor for trans_to_par
            trans = torch.stack([alpha, beta, mu_param])
            par_current = self.trans_to_par(trans)
            a_param = par_current[0]
            b_param = par_current[1]
            mu_param = par_current[2]

            # Build kernel
            K = a_param * torch.exp(-0.5 * self.xdiff2_torch / b_param)
            covariance = K + self.kernel_jitter * torch.eye(K.shape[0], dtype=torch.float64)

            n1 = len(self.y_obs)
            covariance_obs = covariance[:n1, :n1]
            mu_vec = mu_param.expand(n1)

            y = torch.tensor(self.y_obs, dtype=torch.float64).flatten()
            likelihood = torch.distributions.MultivariateNormal(mu_vec, covariance_matrix=covariance_obs)
            log_like = likelihood.log_prob(y)

            log_prior_alpha = -0.5 * (alpha ** 2 / self.sigma_priors[0] ** 2)
            log_prior_beta = -0.5 * (beta ** 2 / self.sigma_priors[1] ** 2)

            neg_log_post = -(log_like + log_prior_alpha + log_prior_beta)
            return neg_log_post

        H = torch.autograd.functional.hessian(neg_log_post_func, params_flat)
        H = H.detach().numpy()
        H = (H + H.T) / 2
        jitter = 1e-6
        max_jitter = 1e-2
        while jitter <= max_jitter:
            jitter_matrix = jitter * np.eye(H.shape[0])
            A = np.linalg.inv(H + jitter_matrix)
            A = (A + A.T) / 2
            try:
                L = scipy.linalg.cholesky(A, lower=True)
                break  # Success, exit the loop
            except np.linalg.LinAlgError:
                if jitter >= max_jitter:
                    print(
                        "Warning: Covariance matrix A is not positive definite. Cholesky decomposition failed even with maximum jitter.")
                    break
                else:
                    jitter *= 10
                    print(f"Warning: Cholesky decomposition failed. Increasing jitter to {jitter}")
        return A

    def get_init_par(self):
        epsilon = 1e-12
        a_range = np.max(self.y_obs) - np.min(self.y_obs)
        a = a_range
        a = np.clip(a, self.lb_a + epsilon, None)
        mu = np.mean(self.y_obs)
        under_diag_sum = np.sum(np.tril(self.c_SIGMA.xdiff2, k=-1))
        pairs = (self.p * (self.p - 1)) // 2
        b = under_diag_sum / pairs
        b = np.clip(b, self.lb_b + epsilon, None)

        # Return scalar values instead of arrays
        par = np.array([a, b, mu])
        return par


    def reset_var(self):
        self.gp_par = {
            "a": np.zeros((self.iter_size, 1)),
            "b": np.zeros((self.iter_size, 1)),
            "mu": np.zeros((self.iter_size, 1))
        }
        self.Xi_samples = np.zeros((self.iter_size, self.p))
        self.acceptance_rate = None

    def unorder_u(self):
        mask = np.ones(len(self.u), dtype=bool)
        mask[self.x_obs_idx_in_u] = False
        self.u = np.concatenate([self.x_obs.reshape(-1, 1), self.u[mask]])

    # @profile
    def get_MALA_prop(self, Xi_com, par, step):
        transformed_par = self.par_to_trans(par)
        par = np.copy(par)
        grad_trans_par = self.pytorch_grads_mvn(par, Xi_com, self.kernel_jitter)
        transformed_par_prime = self.compute_prop(transformed_par, grad_trans_par, step)
        par_prime = self.trans_to_par(transformed_par_prime)
        logq_par_prime = self.log_q_MALA(transformed_par_prime, transformed_par, grad_trans_par, step)
        grad_trans_par_prime = self.pytorch_grads_mvn(par_prime, Xi_com, self.kernel_jitter)
        logq_par = self.log_q_MALA(transformed_par, transformed_par_prime, grad_trans_par_prime, step)

        return par_prime, logq_par, logq_par_prime, transformed_par_prime, transformed_par

    def pytorch_grads_mvn(self, par, Xi_com, jitter):
        par_grads = np.zeros_like(par)
        Xi_torch = torch.tensor(Xi_com.flatten(), dtype=torch.float64)

        # par is now a 1D array of shape (3,)
        par_tensor = torch.as_tensor(par, dtype=torch.float64)

        # Transform to unconstrained space using par_to_trans
        trans = self.par_to_trans(par_tensor)
        alpha_val = trans[0]
        beta_val = trans[1]
        mu_val = trans[2]

        # Create torch tensors with gradients
        alpha = alpha_val.clone().detach().requires_grad_(True)
        beta = beta_val.clone().detach().requires_grad_(True)
        mu = mu_val.clone().detach().requires_grad_(True)

        # Stack back into [3] tensor for trans_to_par
        trans_stack = torch.stack([alpha, beta, mu])
        par_current = self.trans_to_par(trans_stack)
        a = par_current[0]
        b = par_current[1]
        mu_param = par_current[2]

        # Kernel
        if self.gp_kernel == "RBF":
            K = a * torch.exp(-self.xdiff2_torch / (2 * b))
        else:
            raise ValueError(f"Unsupported kernel: {self.gp_kernel}")
        K = K + jitter * torch.eye(self.p, dtype=torch.float64)

        # Likelihood + priors
        mu_vec = mu_param * torch.ones(self.p, dtype=torch.float64)
        mvn = torch.distributions.MultivariateNormal(mu_vec, covariance_matrix=K)

        log_prior_alpha = -0.5 * (alpha ** 2) / (self.sigma_priors[0] ** 2)
        log_prior_beta = -0.5 * (beta ** 2) / (self.sigma_priors[1] ** 2)

        log_posterior = mvn.log_prob(Xi_torch) + log_prior_alpha + log_prior_beta

        # Backpropagate gradients
        log_posterior.backward()

        # Store results
        par_grads[0] = alpha.grad.item()
        par_grads[1] = beta.grad.item()
        par_grads[2] = mu.grad.item()

        return par_grads

    def compute_prop(self, transformed_par, grad_trans_par, step):
        Xi = np.random.multivariate_normal(
            mean=np.zeros_like(transformed_par),
            cov=np.eye(len(transformed_par)),
            size=1
        )[0]

        B = 2 * step * self.A
        try:
            L = scipy.linalg.cholesky(B, lower=True)
        except:
            print('A: ', self.A)
        noise_term = L @ Xi

        transformed_par_prime = transformed_par + step * (self.A @ grad_trans_par) + noise_term

        return transformed_par_prime

    def log_q_MALA(self, transformed_par_prime, transformed_par, grad_trans_par, step):
        diff = transformed_par_prime - transformed_par - step * grad_trans_par
        squared_norms = np.sum(diff ** 2, axis=0)
        # Compute log_q for each dimension
        log_q = -squared_norms / (4 * step)

        return log_q  # Returns array of shape (dim,)

    def is_adapted(self, accepted, counter):
        if 0.55 > accepted / counter > 0.3:
            return True
        return False

    def adapt_step(self, accepted, counter):
        rate = accepted / counter
        if rate < 0.3:
            if rate < 0.05:
                print('Acceptance rate extremely low:')
                self.MALA_step = self.MALA_step / 12
            elif rate < 0.2:
                print('Acceptance rate relatively low:')
                self.MALA_step = self.MALA_step / 6
            else:
                print('Acceptance rate a bit low:')
                self.MALA_step = self.MALA_step / 2.4
        else:
            if rate < 0.75:
                print('Acceptance rate a bit high:')
                self.MALA_step = self.MALA_step * 2
            elif rate < 0.95:
                print('Acceptance rate relatively high:')
                self.MALA_step = self.MALA_step * 5
            else:
                print('Acceptance rate extremely high:')
                self.MALA_step = self.MALA_step * 10
        print(rate)

    def update_step(self, ar, k, a=10, t0=10, kappa=0.6, alpha_target=0.57):
        gamma_k = a / ((k + t0) ** kappa)
        log_step = np.log(self.MALA_step)
        new_log_step = log_step + gamma_k * (ar - alpha_target)
        new_step = np.exp(new_log_step)
        self.MALA_step = np.clip(new_step, 1e-12, 0.1)

    def order_Xi(self):
        n1 = len(self.x_obs_idx_in_u)
        Xi_reordered = np.zeros_like(self.Xi_samples)

        for i, target_idx in enumerate(self.x_obs_idx_in_u):
            Xi_reordered[:, target_idx] = self.Xi_samples[:, i]

        all_indices = np.arange(self.p)
        remaining_indices = all_indices[~np.isin(all_indices, self.x_obs_idx_in_u)]
        Xi_reordered[:, remaining_indices] = self.Xi_samples[:, n1:]

        self.Xi_samples = Xi_reordered

    def communicate(self, iter_counter, adaptor_counter, is_adapting, iter_limit):
        if is_adapting:
            if adaptor_counter % iter_limit == 0:
                print(adaptor_counter, 'iterations of step size adaptation done')
        else:
            if iter_counter % iter_limit == 0:
                print(iter_counter, 'iterations done')

    def par_to_trans(self, par):
        a, b, mu = par[0], par[1], par[2]

        if isinstance(a, torch.Tensor):
            alpha = torch.log(a - self.lb_a)
            beta = torch.log(b - self.lb_b)
            return torch.stack([alpha, beta, mu])
        else:
            alpha = np.log(a - self.lb_a)
            beta = np.log(b - self.lb_b)
            return np.array([alpha, beta, mu], dtype=par.dtype)

    def trans_to_par(self, trans):
        alpha, beta, mu = trans[0], trans[1], trans[2]

        if isinstance(alpha, torch.Tensor):
            a = self.lb_a + torch.exp(alpha)
            b = self.lb_b + torch.exp(beta)
            return torch.stack([a, b, mu])
        else:
            a = self.lb_a + np.exp(alpha)
            b = self.lb_b + np.exp(beta)
            return np.array([a, b, mu], dtype=trans.dtype)

    def debug_mu_curvature(self, mu, Xi, SIGMA_inv_par,
                           width=2,  # sweep ± width around mu
                           n_points=401):  # resolution of the sweep

        # Prepare sweep
        mus = np.linspace(mu - width, mu + width, n_points)
        loglik_vals = np.zeros_like(mus)

        # Evaluate log-likelihood for each mu value
        for i, m in enumerate(mus):
            loglik_vals[i] = float(self.get_unnormalized_loglik(Xi, m, SIGMA_inv_par))

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(mus, loglik_vals, '-k', lw=2)
        plt.axvline(mu, color='red', linestyle='--', label=f'mu={mu:.4g}')
        plt.xlabel("mu", fontsize=12)
        plt.ylabel("log-likelihood", fontsize=12)
        plt.title("Log-likelihood curvature around mu")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return mus, loglik_vals










