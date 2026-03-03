import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm

class emp_ucb:
    def __init__(self, gp):
        self.gp = gp
        self.beta = 1*np.log(len(self.gp.y_obs))
        self.acq_val = None

    def optimize(self):
        mean = np.mean(self.gp.y_predictions, axis=0)
        sigma = np.std(self.gp.y_predictions, ddof=1, axis=0)
        # print('sigma is: ',sigma)
        self.acq_val = mean + sigma*np.sqrt(self.beta) ###########
        max_idx = np.argmax(self.acq_val)
        max_location = np.array(self.gp.grid_to_predict[max_idx]).reshape(1, -1)
        return max_location, max_idx


class emp_rb:
    def __init__(self, gp, novelty_threshold=1e-4):  # MODIFIED: Added novelty_threshold parameter
        self.gp = gp
        self.beta = 0.1 * np.log(len(self.gp.y_obs))
        self.acq_val = None
        self.novelty_threshold = novelty_threshold  # NEW: Store threshold

    def acquisition_function(self, x, debug=False):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # x is now (s, dim) where s is number of points
        samples = self.gp.predict(x, normalized_out=True)  # Returns (s, k)

        # Compute statistics for each point (along axis=1 for each row)
        mean_pred = np.mean(samples, axis=1)  # Shape: (s,)
        lower_bound_pred = np.percentile(samples, 5, axis=1)  # Shape: (s,)
        upper_bound_pred = np.percentile(samples, 95, axis=1)  # Shape: (s,)

        range_size = upper_bound_pred - lower_bound_pred  # Shape: (s,)
        acq_values = mean_pred + range_size * np.sqrt(self.beta)  # Shape: (s,)

        # Return scalar if input was single point, otherwise return array
        return acq_values[0] if len(acq_values) == 1 else acq_values

    def optimize(self, bounds, method="L-BFGS-B"):
        bounds = np.array(bounds)
        dim = len(bounds)
        opts = {'maxiter': 500 * dim, 'maxfun': 500 * dim, 'disp': False}

        restart_num = 3 * dim
        X_candidate = []
        AF_candidate = []

        for i in range(restart_num):
            init_X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(30 * dim, dim))
            value_holder = self.acquisition_function(init_X, debug=False)  # Returns array of shape (30*dim,)
            x0 = init_X[np.argmax(value_holder)]
            res = minimize(lambda x: -self.acquisition_function(x), x0,
                           bounds=bounds, method=method, options=opts)
            print(f"Function evaluations: {res.nfev}")
            print(f"Iterations: {res.nit}")
            print(f"Message: {res.message}")
            print(f"Success: {res.success}")
            X_temp = res.x
            AF_temp = self.acquisition_function(X_temp)

            # NEW: Check if candidate is novel (not too close to observed points)
            distances = np.linalg.norm(self.gp.x_obs_original - X_temp, axis=1)
            min_distance = np.min(distances)
            is_novel = min_distance > self.novelty_threshold

            if is_novel:  # NEW: Only add to candidates if novel
                X_candidate.append(X_temp)
                AF_candidate.append(AF_temp)
                # print(f"Novel candidate accepted (min distance: {min_distance:.6f})")
            else:
                print(f"Candidate rejected - too close to observed point (min distance: {min_distance:.6f})")

        # NEW: Fallback if no valid candidates found
        if len(X_candidate) == 0:
            print("Warning: No novel candidates found. Returning random point from bounds.")
            X_next = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, dim))
        else:
            X_next = X_candidate[np.argmax(AF_candidate)]

        return X_next.reshape(1, -1)



class emp_MES:
    def __init__(self, gp, novelty_threshold=1e-4, eps=1e-8):
        self.gp = gp
        self.novelty_threshold = novelty_threshold
        self.eps = eps
        self.sqrt_beta = None   # frozen per BO iteration

    # --------------------------------------------------
    # Compute beta ONCE per BO iteration
    # --------------------------------------------------
    def update_beta(self, bounds, n_ref=1000):
        bounds = np.array(bounds)
        dim = bounds.shape[0]

        # Reference set for beta calibration
        X_ref = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(n_ref, dim)
        )

        samples = self.gp.predict(X_ref, normalized_out=True)
        mu = np.mean(samples, axis=1)

        q5 = np.percentile(samples, 5, axis=1)
        q95 = np.percentile(samples, 95, axis=1)
        range90 = np.maximum(q95 - q5, self.eps)

        ratios = (self.gp.y_max - mu) / range90
        self.sqrt_beta = np.min(ratios)

    # --------------------------------------------------
    # Acquisition (beta is frozen!)
    # --------------------------------------------------
    def acquisition_function(self, x, debug=False):
        if self.sqrt_beta is None:
            raise RuntimeError("update_beta() must be called before optimize().")

        x = np.atleast_2d(x)

        samples = self.gp.predict(x, normalized_out=True)
        mu = np.mean(samples, axis=1)

        q5 = np.percentile(samples, 5, axis=1)
        q95 = np.percentile(samples, 95, axis=1)
        range90 = np.maximum(q95 - q5, self.eps)

        acq = mu + self.sqrt_beta * range90

        if debug:
            print("mu:", mu)
            print("range90:", range90)
            print("sqrt_beta:", self.sqrt_beta)
            print("acq:", acq)

        return acq if len(acq) > 1 else acq[0]

    # --------------------------------------------------
    # Optimization (unchanged logic)
    # --------------------------------------------------
    def optimize(self, bounds, method="L-BFGS-B"):
        bounds = np.array(bounds)
        dim = len(bounds)

        opts = {'maxiter': 500 * dim, 'maxfun': 500 * dim, 'disp': False}
        restart_num = 3 * dim

        X_candidate = []
        AF_candidate = []

        for _ in range(restart_num):
            init_X = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(30 * dim, dim)
            )

            values = self.acquisition_function(init_X)
            x0 = init_X[np.argmax(values)]

            res = minimize(
                lambda x: -self.acquisition_function(x),
                x0,
                bounds=bounds,
                method=method,
                options=opts
            )
            print(f"Function evaluations: {res.nfev}")
            print(f"Iterations: {res.nit}")
            print(f"Message: {res.message}")
            print(f"Success: {res.success}")

            X_temp = res.x
            AF_temp = self.acquisition_function(X_temp)

            # Novelty check
            distances = np.linalg.norm(self.gp.x_obs_original - X_temp, axis=1)
            if np.min(distances) > self.novelty_threshold:
                X_candidate.append(X_temp)
                AF_candidate.append(AF_temp)

        if len(X_candidate) == 0:
            X_next = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(1, dim)
            )
        else:
            X_next = X_candidate[np.argmax(AF_candidate)]

        return X_next.reshape(1, -1)


class emp_MES_quant:
    def __init__(self, gp, novelty_threshold=1e-4, eps=1e-8):
        self.gp = gp
        self.novelty_threshold = novelty_threshold
        self.eps = eps
        self.sqrt_beta = None   # frozen per BO iteration

    # --------------------------------------------------
    # Compute beta ONCE per BO iteration
    # --------------------------------------------------
    def update_beta(self, bounds, n_ref=1000):
        bounds = np.array(bounds)
        dim = bounds.shape[0]

        # Reference set for beta calibration
        X_ref = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(n_ref, dim)
        )

        samples = self.gp.predict(X_ref, normalized_out=True)
        q50 = np.percentile(samples, 50, axis=1)

        q5 = np.percentile(samples, 5, axis=1)
        q95 = np.percentile(samples, 95, axis=1)
        range90 = np.maximum(q95 - q5, self.eps)

        ratios = (self.gp.y_max - q50) / range90
        self.sqrt_beta = np.min(ratios)

    # --------------------------------------------------
    # Acquisition (beta is frozen!)
    # --------------------------------------------------
    def acquisition_function(self, x, debug=False):
        if self.sqrt_beta is None:
            raise RuntimeError("update_beta() must be called before optimize().")

        x = np.atleast_2d(x)

        samples = self.gp.predict(x, normalized_out=True)
        q50 = np.percentile(samples, 50, axis=1)

        q5 = np.percentile(samples, 5, axis=1)
        q95 = np.percentile(samples, 95, axis=1)
        range90 = np.maximum(q95 - q5, self.eps)

        acq = q50 + self.sqrt_beta * range90

        if debug:
            print("q50:", q50)
            print("range90:", range90)
            print("sqrt_beta:", self.sqrt_beta)
            print("acq:", acq)

        return acq if len(acq) > 1 else acq[0]

    # --------------------------------------------------
    # Optimization (unchanged logic)
    # --------------------------------------------------
    def optimize(self, bounds, method="L-BFGS-B"):
        bounds = np.array(bounds)
        dim = len(bounds)

        opts = {'maxiter': 500 * dim, 'maxfun': 500 * dim, 'disp': False}
        restart_num = 3 * dim

        X_candidate = []
        AF_candidate = []

        for _ in range(restart_num):
            init_X = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(30 * dim, dim)
            )

            values = self.acquisition_function(init_X)
            x0 = init_X[np.argmax(values)]

            res = minimize(
                lambda x: -self.acquisition_function(x),
                x0,
                bounds=bounds,
                method=method,
                options=opts
            )
            print(f"Function evaluations: {res.nfev}")
            print(f"Iterations: {res.nit}")
            print(f"Message: {res.message}")
            print(f"Success: {res.success}")

            X_temp = res.x
            AF_temp = self.acquisition_function(X_temp)

            # Novelty check
            distances = np.linalg.norm(self.gp.x_obs_original - X_temp, axis=1)
            if np.min(distances) > self.novelty_threshold:
                X_candidate.append(X_temp)
                AF_candidate.append(AF_temp)

        if len(X_candidate) == 0:
            X_next = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(1, dim)
            )
        else:
            X_next = X_candidate[np.argmax(AF_candidate)]

        return X_next.reshape(1, -1)


class TrueMES:
    def __init__(self, gp, novelty_threshold=1e-4):
        self.gp = gp
        self.novelty_threshold = novelty_threshold
        self.acq_val = None

    def acquisition_function(self, x, debug=False):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        samples = self.gp.predict(x, normalized_out=True)

        mu_h = np.percentile(samples, 50, axis=1)

        q75 = np.percentile(samples, 75, axis=1)
        q25 = np.percentile(samples, 25, axis=1)
        sigma_h = (q75 - q25) / 1.349

        sigma_h = np.clip(sigma_h, 1e-9, None)

        gamma = (self.gp.y_max - mu_h) / sigma_h

        pdf_g = norm.pdf(gamma)
        cdf_g = norm.cdf(gamma)

        cdf_g = np.clip(cdf_g, 1e-12, 1.0)

        acq_values = (gamma * pdf_g) / (2 * cdf_g) - np.log(cdf_g)

        return acq_values[0] if len(acq_values) == 1 else acq_values

    def optimize(self, bounds, method="L-BFGS-B"):
        bounds = np.array(bounds)
        dim = len(bounds)

        opts = {'maxiter': 500 * dim, 'maxfun': 500 * dim, 'disp': False}
        restart_num = 3 * dim

        X_candidate = []
        AF_candidate = []

        for _ in range(restart_num):
            init_X = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(30 * dim, dim)
            )

            values = self.acquisition_function(init_X)
            x0 = init_X[np.argmax(values)]

            res = minimize(
                lambda x: -self.acquisition_function(x),
                x0,
                bounds=bounds,
                method=method,
                options=opts
            )
            # print(f"Function evaluations: {res.nfev}")
            # print(f"Iterations: {res.nit}")
            # print(f"Message: {res.message}")
            # print(f"Success: {res.success}")

            X_temp = res.x
            AF_temp = self.acquisition_function(X_temp)

            # Novelty check
            distances = np.linalg.norm(self.gp.x_obs_original - X_temp, axis=1)
            if np.min(distances) > self.novelty_threshold:
                X_candidate.append(X_temp)
                AF_candidate.append(AF_temp)

        if len(X_candidate) == 0:
            X_next = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(1, dim)
            )
        else:
            X_next = X_candidate[np.argmax(AF_candidate)]

        return X_next.reshape(1, -1)


class TrueMES75:
    def __init__(self, gp, novelty_threshold=1e-4):
        self.gp = gp
        self.novelty_threshold = novelty_threshold
        self.acq_val = None

    def acquisition_function(self, x, debug=False):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        samples = self.gp.predict(x, normalized_out=True)

        mu_h = np.percentile(samples, 50, axis=1)

        q875 = np.percentile(samples, 87.5, axis=1)
        q125 = np.percentile(samples, 12.5, axis=1)
        sigma_h = (q875 - q125) / 2.3006

        sigma_h = np.clip(sigma_h, 1e-9, None)

        gamma = (self.gp.y_max - mu_h) / sigma_h

        pdf_g = norm.pdf(gamma)
        cdf_g = norm.cdf(gamma)

        cdf_g = np.clip(cdf_g, 1e-12, 1.0)

        acq_values = (gamma * pdf_g) / (2 * cdf_g) - np.log(cdf_g)

        return acq_values[0] if len(acq_values) == 1 else acq_values

    def optimize(self, bounds, method="L-BFGS-B"):
        bounds = np.array(bounds)
        dim = len(bounds)

        opts = {'maxiter': 500 * dim, 'maxfun': 500 * dim, 'disp': False}
        restart_num = 3 * dim

        X_candidate = []
        AF_candidate = []

        for _ in range(restart_num):
            init_X = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(30 * dim, dim)
            )

            values = self.acquisition_function(init_X)
            x0 = init_X[np.argmax(values)]

            res = minimize(
                lambda x: -self.acquisition_function(x),
                x0,
                bounds=bounds,
                method=method,
                options=opts
            )
            # print(f"Function evaluations: {res.nfev}")
            # print(f"Iterations: {res.nit}")
            # print(f"Message: {res.message}")
            # print(f"Success: {res.success}")

            X_temp = res.x
            AF_temp = self.acquisition_function(X_temp)

            # Novelty check
            distances = np.linalg.norm(self.gp.x_obs_original - X_temp, axis=1)
            if np.min(distances) > self.novelty_threshold:
                X_candidate.append(X_temp)
                AF_candidate.append(AF_temp)

        if len(X_candidate) == 0:
            X_next = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(1, dim)
            )
        else:
            X_next = X_candidate[np.argmax(AF_candidate)]

        return X_next.reshape(1, -1)
