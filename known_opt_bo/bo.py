import numpy as np
from known_opt_gp.truncated_gp import TGQP
from known_opt_bo.acquisition import acq_functions


class BOKnownOpt:
    def __init__(self, init_x, function, acq='emp_rb', gp_iter_size=2000, chains=1, gp_kernel='RBF',
                 kernel_jitter=1e-6, MALA_step=0.01, sigma_priors=None, p=None, bounds_a=None, bounds_b=None):
        self.X = np.asarray(init_x)
        self.function = function
        self.y = self.function.func(self.X) * self.function.ismax
        print("Initial observations:")
        print(f"X coordinates:\n{self.X}")
        print(f"y values:\n{self.y}")
        self.acq = acq
        self.gp_iter_size = gp_iter_size
        self.chains = chains
        self.gp_kernel = gp_kernel
        self.kernel_jitter = kernel_jitter
        self.MALA_step = MALA_step
        self.sigma_priors = sigma_priors
        self.bounds_a = bounds_a
        self.bounds_b = bounds_b
        self.p = p
        self.f_max = self.function.fstar * self.function.ismax
        if isinstance(function.bounds_dict, dict):
            self.search_space = []
            for key in list(function.bounds_dict.keys()):
                self.search_space.append(function.bounds_dict[key])
            self.search_space = np.asarray(self.search_space)
        else:
            self.search_space = np.asarray(function.bounds)
        self.history = []

    def select_next_point(self):
        gps = []
        for chains in range(self.chains):
            gp = TGQP(self.X, self.y, self.f_max, self.search_space, self.gp_kernel, self.kernel_jitter,
                           self.gp_iter_size, self.MALA_step, self.sigma_priors, self.p, self.bounds_a, self.bounds_b)
            gp.fit()
            gps.append(gp)

        if self.acq == 'emp_MES':
            acq = acq_functions.emp_MES(gp)
            acq.update_beta(self.search_space)
        elif self.acq == 'emp_MES_quant':
            acq = acq_functions.emp_MES_quant(gp)
            acq.update_beta(self.search_space)
        elif self.acq == 'TrueMES':
            acq = acq_functions.TrueMES(gp)
        elif self.acq == 'TrueMES75':
            acq = acq_functions.TrueMES75(gp)
        elif self.acq == 'emp_rb':
            acq = acq_functions.emp_rb(gp)
        next_loc = acq.optimize(self.search_space)

        iteration_data = {
            "X": self.X.copy(),
            'y': self.y.copy(),
            "gp_chains": gps,
            "next_loc": next_loc
        }
        self.history.append(iteration_data)

        self.X = np.vstack((self.X, next_loc))
        self.y = np.vstack((self.y, self.function.func(next_loc) * self.function.ismax))
        # self.p = self.p + 2
