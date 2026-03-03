# -*- coding: utf-8 -*-

import torch
from typing import Optional
import numpy as np
from collections import OrderedDict
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def reshape(x, input_dim):
    '''
    Reshapes x into a matrix with input_dim columns
    '''

    x = np.array(x)
    if x.size == input_dim:
        x = x.reshape((1, input_dim))
    return x


class functions:
    def plot(self):
        bounds = self.bounds
        if isinstance(bounds, dict):
            # Get the name of the parameters
            keys = bounds.keys()
            arr_bounds = []
            for key in keys:
                arr_bounds.append(bounds[key])
                arr_bounds = np.asarray(arr_bounds)
        else:
            arr_bounds = np.asarray(bounds)
        X = np.array([np.arange(x[0], x[1], 0.01) for x in arr_bounds])
        X = X.reshape(-1, 2)
        X1 = np.array([X[:, 0]])
        X2 = np.array([X[:, 1]])
        X1, X2 = np.meshgrid(X1, X2)
        y = np.zeros([X1.shape[1], X2.shape[1]])
        # print(y.shape)
        # print(X1.shape)
        # print(X2.shape)
        for ii in range(0, X1.shape[1]):
            for jj in range(0, X2.shape[1]):
                Xij = np.array([X1[ii, ii], X2[jj, jj]])
                # print(Xij)
                y[ii, jj] = self.func(Xij)
                #        f1=plt.figure(1)
                #        ax=plt.axes(projection='3d')
                #        ax.plot_surface(X1,X2,y)
                plt.contourf(X1, X2, y, levels=np.arange(0, 35, 1))
                plt.colorbar()

    def findSdev(self):
        num_points_per_dim = 100
        bounds = self.bounds
        if isinstance(bounds, dict):
            # Get the name of the parameters
            keys = bounds.keys()
            arr_bounds = []
            for key in keys:
                arr_bounds.append(bounds[key])
            arr_bounds = np.asarray(arr_bounds)
        else:
            arr_bounds = np.asarray(bounds)
        X = np.array([np.random.uniform(x[0], x[1], size=num_points_per_dim) for x in arr_bounds])
        X = X.reshape(num_points_per_dim, -1)
        y = self.func(X)
        sdv = np.std(y)
        # maxima=np.max(y)
        # minima=np.min(y)
        return sdv



class sin(functions):
    def __init__(self, sd=None):
        self.input_dim = 1
        self.bounds = {'x': (-1, 15)}
        # self.bounds={'x':(0,1)}

        self.fstar = 11
        self.min = 0
        self.ismax = 1
        self.name = 'sin'
        if sd == None or sd == 0:
            self.sd = 0
        else:
            # self.sd=self.findSdev()
            self.sd = sd

    def func(self, x):
        x = np.asarray(x)

        fval = np.sin(x)
        return fval * self.ismax


class sincos(functions):
    def __init__(self, sd=None):
        self.input_dim = 1
        self.bounds = {'x': (-1, 2)}
        # self.bounds={'x':(0,1)}

        self.fstar = 11
        self.min = 0
        self.ismax = 1
        self.name = 'sincos'
        if sd == None or sd == 0:
            self.sd = 0
        else:
            # self.sd=self.findSdev()
            self.sd = sd

    def func(self, x):
        x = np.asarray(x)

        fval = x * np.sin(x) + x * np.cos(2 * x)
        return fval * self.ismax


class fourier:

    def __init__(self, negate: bool = False, sd: Optional[float] = None):
        # Interface for custom algorithm
        self.bounds_dict = {'x': (0, 10)}
        self.input_dim = 1
        self.ismax = -1 if not negate else 1
        self.min = 4.795  ## approx location of minimum
        self.fstar = -9.5083483926941064  ## approx optimal value (not multiplied by ismax)
        self.name = 'fourier'

        # Interface for botorch
        self.dim = 1
        self.bounds = torch.tensor([[0.0], [10.0]], dtype=torch.float64)

        # Noise parameter
        if sd is None or sd == 0:
            self.sd = 0
        else:
            self.sd = sd

    def func(self, X):
        """Evaluation method for custom algorithm (numpy input)"""
        X = np.asarray(X)
        X = X.reshape((len(X), 1))
        n = X.shape[0]

        fval = X * np.sin(X) + X * np.cos(2 * X)

        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, 0.1 * self.sd, n).reshape(n, 1)

        return fval.reshape(n, 1) + noise

    def __call__(self, X):
        """Evaluation method for botorch (torch.Tensor input)"""
        # Check if input is tensor
        is_tensor = torch.is_tensor(X)

        # Convert to numpy if torch tensor
        if is_tensor:
            X_np = X.detach().cpu().numpy()
            device = X.device
            dtype = X.dtype
        else:
            X_np = np.asarray(X)

        # Ensure correct shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # Evaluate
        fval = X_np * np.sin(X_np) + X_np * np.cos(2 * X_np)

        # Add noise if needed
        if self.sd == 0:
            noise = 0
        else:
            noise = np.random.normal(0, 0.1 * self.sd, X_np.shape[0])

        result = fval.flatten() + noise

        # Return in same format as input
        if is_tensor:
            result_tensor = torch.tensor(result, device=device, dtype=dtype)
            # if result_tensor.shape[0] == 1:
            #     return result_tensor[0]
            return result_tensor
        else:
            # if result.shape[0] == 1:
            #     return result[0]
            return result


class Forrester:

    def __init__(self, negate: bool = False, sd: Optional[float] = None):
        # Interface for custom algorithm
        self.bounds_dict = {'x': (0, 1)}
        self.input_dim = 1
        self.ismax = -1 if not negate else 1
        self.min = 0.757249  ## approx location of minimum
        self.fstar = -6.02074  ## approx optimal value (not multiplied by ismax)
        self.name = 'forrester'

        # Interface for botorch
        self.dim = 1
        self.bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

        # Noise parameter
        if sd is None or sd == 0:
            self.sd = 0
        else:
            self.sd = sd

    def func(self, X):
        """Evaluation method for custom algorithm (numpy input)"""
        X = np.asarray(X)
        X = X.reshape((len(X), 1))
        n = X.shape[0]

        fval = ((6 * X - 2)**2) * np.sin(12 * X - 4)

        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, 0.1 * self.sd, n).reshape(n, 1)

        return fval.reshape(n, 1) + noise

    def __call__(self, X):
        """Evaluation method for botorch (torch.Tensor input)"""
        # Check if input is tensor
        is_tensor = torch.is_tensor(X)

        # Convert to numpy if torch tensor
        if is_tensor:
            X_np = X.detach().cpu().numpy()
            device = X.device
            dtype = X.dtype
        else:
            X_np = np.asarray(X)

        # Ensure correct shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # Evaluate
        fval = ((6 * X_np - 2)**2) * np.sin(12 * X_np - 4)

        # Add noise if needed
        if self.sd == 0:
            noise = 0
        else:
            noise = np.random.normal(0, 0.1 * self.sd, X_np.shape[0])

        result = fval.flatten() + noise

        # Return in same format as input
        if is_tensor:
            result_tensor = torch.tensor(result, device=device, dtype=dtype)
            # if result_tensor.shape[0] == 1:
            #     return result_tensor[0]
            return result_tensor
        else:
            # if result.shape[0] == 1:
            #     return result[0]
            return result


class Levy:
    def __init__(self, negate: bool = False, sd: Optional[float] = None):
        # Interface for custom algorithm
        self.bounds_dict = {'x': (-10, 10)}
        self.input_dim = 1
        self.ismax = -1 if not negate else 1
        self.min = 1.0  ## approx location of minimum
        self.fstar = 0.0  ## approx optimal value (not multiplied by ismax)
        self.name = 'levy'

        # Interface for botorch
        self.dim = 1
        self.bounds = torch.tensor([[-10.0], [10.0]], dtype=torch.float64)

        # Noise parameter
        if sd is None or sd == 0:
            self.sd = 0
        else:
            self.sd = sd

    def func(self, X):
        """Evaluation method for custom algorithm (numpy input)"""
        X = np.asarray(X)
        X = X.reshape((len(X), 1))
        n = X.shape[0]

        fval = (np.sin(np.pi * (X + 3) / 4)**2 +
                ((X - 1) / 4)**2 * (1 + np.sin(np.pi * (X + 3) / 2)**2))

        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, 0.1 * self.sd, n).reshape(n, 1)

        return fval.reshape(n, 1) + noise

    def __call__(self, X):
        """Evaluation method for botorch (torch.Tensor input)"""
        # Check if input is tensor
        is_tensor = torch.is_tensor(X)

        # Convert to numpy if torch tensor
        if is_tensor:
            X_np = X.detach().cpu().numpy()
            device = X.device
            dtype = X.dtype
        else:
            X_np = np.asarray(X)

        # Ensure correct shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # Evaluate
        fval = (np.sin(np.pi * (X_np + 3) / 4)**2 +
                ((X_np - 1) / 4)**2 * (1 + np.sin(np.pi * (X_np + 3) / 2)**2))

        # Add noise if needed
        if self.sd == 0:
            noise = 0
        else:
            noise = np.random.normal(0, 0.1 * self.sd, X_np.shape[0])

        result = fval.flatten() + noise

        # Return in same format as input
        if is_tensor:
            result_tensor = torch.tensor(result, device=device, dtype=dtype)
            # if result_tensor.shape[0] == 1:
            #     return result_tensor[0]
            return result_tensor
        else:
            # if result.shape[0] == 1:
            #     return result[0]
            return result



class GL:
    def __init__(self, negate: bool = False, sd: Optional[float] = None):
        # Interface for custom algorithm
        self.bounds_dict = {'x': (0.5, 2.5)}
        self.input_dim = 1
        self.ismax = -1 if not negate else 1
        self.min = 0.54856405  ## approx location of minimum
        self.fstar = -0.86901113
        self.name = 'gl'

        # Interface for botorch
        self.dim = 1
        self.bounds = torch.tensor([[0.5], [2.5]], dtype=torch.float64)

        # Noise parameter
        if sd is None or sd == 0:
            self.sd = 0
        else:
            self.sd = sd

    def func(self, X):
        """Evaluation method for custom algorithm (numpy input)"""
        X = np.asarray(X)
        X = X.reshape((len(X), 1))
        n = X.shape[0]

        fval = np.sin(10 * np.pi * X) / (2 * X) + (X - 1)**4

        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, 0.1 * self.sd, n).reshape(n, 1)

        return fval.reshape(n, 1) + noise

    def __call__(self, X):
        """Evaluation method for botorch (torch.Tensor input)"""
        # Check if input is tensor
        is_tensor = torch.is_tensor(X)

        # Convert to numpy if torch tensor
        if is_tensor:
            X_np = X.detach().cpu().numpy()
            device = X.device
            dtype = X.dtype
        else:
            X_np = np.asarray(X)

        # Ensure correct shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # Evaluate
        fval = np.sin(10 * np.pi * X_np) / (2 * X_np) + (X_np - 1)**4

        # Add noise if needed
        if self.sd == 0:
            noise = 0
        else:
            noise = np.random.normal(0, 0.1 * self.sd, X_np.shape[0])

        result = fval.flatten() + noise

        # Return in same format as input
        if is_tensor:
            result_tensor = torch.tensor(result, device=device, dtype=dtype)
            # if result_tensor.shape[0] == 1:
            #     return result_tensor[0]
            return result_tensor
        else:
            # if result.shape[0] == 1:
            #     return result[0]
            return result


class MultiModal2:
    def __init__(self, negate: bool = False, sd: Optional[float] = None):
        # Interface for custom algorithm
        self.bounds_dict = {'x': (0.0, 1.2)}
        self.input_dim = 1
        self.ismax = -1 if not negate else 1
        self.min = 0.966086  ## approx location of minimum
        self.fstar = -1.489073  ## approx optimal value (not multiplied by ismax)
        self.name = 'multimodal2'

        # Interface for botorch
        self.dim = 1
        self.bounds = torch.tensor([[0.0], [1.2]], dtype=torch.float64)

        # Noise parameter
        if sd is None or sd == 0:
            self.sd = 0
        else:
            self.sd = sd

    def func(self, X):
        """Evaluation method for custom algorithm (numpy input)"""
        X = np.asarray(X)
        X = X.reshape((len(X), 1))
        n = X.shape[0]

        fval = -(1.4 - 3 * X) * np.sin(18 * X)

        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, 0.1 * self.sd, n).reshape(n, 1)

        return fval.reshape(n, 1) + noise

    def __call__(self, X):
        """Evaluation method for botorch (torch.Tensor input)"""
        # Check if input is tensor
        is_tensor = torch.is_tensor(X)

        # Convert to numpy if torch tensor
        if is_tensor:
            X_np = X.detach().cpu().numpy()
            device = X.device
            dtype = X.dtype
        else:
            X_np = np.asarray(X)

        # Ensure correct shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # Evaluate
        fval = -(1.4 - 3 * X_np) * np.sin(18 * X_np)

        # Add noise if needed
        if self.sd == 0:
            noise = 0
        else:
            noise = np.random.normal(0, 0.1 * self.sd, X_np.shape[0])

        result = fval.flatten() + noise

        # Return in same format as input
        if is_tensor:
            result_tensor = torch.tensor(result, device=device, dtype=dtype)
            # if result_tensor.shape[0] == 1:
            #     return result_tensor[0]
            return result_tensor
        else:
            # if result.shape[0] == 1:
            #     return result[0]
            return result


class MultiModal7:
    def __init__(self, negate: bool = False, sd: Optional[float] = None):
        # Interface for custom algorithm
        self.bounds_dict = {'x': (2.7, 7.5)}
        self.input_dim = 1
        self.ismax = -1 if not negate else 1
        self.min = 5.19978  ## approx location of minimum
        self.fstar = -1.6013  ## approx optimal value (not multiplied by ismax)
        self.name = 'multimodal7'

        # Interface for botorch
        self.dim = 1
        self.bounds = torch.tensor([[2.7], [7.5]], dtype=torch.float64)

        # Noise parameter
        if sd is None or sd == 0:
            self.sd = 0
        else:
            self.sd = sd

    def func(self, X):
        """Evaluation method for custom algorithm (numpy input)"""
        X = np.asarray(X)
        X = X.reshape((len(X), 1))
        n = X.shape[0]

        fval = np.sin(X) + np.sin(10 * X / 3) + np.log(X) - 0.84 * X + 3

        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, 0.1 * self.sd, n).reshape(n, 1)

        return fval.reshape(n, 1) + noise

    def __call__(self, X):
        """Evaluation method for botorch (torch.Tensor input)"""
        # Check if input is tensor
        is_tensor = torch.is_tensor(X)

        # Convert to numpy if torch tensor
        if is_tensor:
            X_np = X.detach().cpu().numpy()
            device = X.device
            dtype = X.dtype
        else:
            X_np = np.asarray(X)

        # Ensure correct shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # Evaluate
        fval = np.sin(X_np) + np.sin(10 * X_np / 3) + np.log(X_np) - 0.84 * X_np + 3

        # Add noise if needed
        if self.sd == 0:
            noise = 0
        else:
            noise = np.random.normal(0, 0.1 * self.sd, X_np.shape[0])

        result = fval.flatten() + noise

        # Return in same format as input
        if is_tensor:
            result_tensor = torch.tensor(result, device=device, dtype=dtype)
            # if result_tensor.shape[0] == 1:
            #     return result_tensor[0]
            return result_tensor
        else:
            # if result.shape[0] == 1:
            #     return result[0]
            return result


class MultiModal14:

    def __init__(self, negate: bool = False, sd: Optional[float] = None):
        # Interface for custom algorithm
        self.bounds_dict = {'x': (0.0, 4.0)}
        self.input_dim = 1
        self.ismax = -1 if not negate else 1
        self.min = 0.224885  ## approx location of minimum
        self.fstar = -0.788685  ## approx optimal value (not multiplied by ismax)
        self.name = 'multimodal14'

        # Interface for botorch
        self.dim = 1
        self.bounds = torch.tensor([[0.0], [4.0]], dtype=torch.float64)

        # Noise parameter
        if sd is None or sd == 0:
            self.sd = 0
        else:
            self.sd = sd

    def func(self, X):
        """Evaluation method for custom algorithm (numpy input)"""
        X = np.asarray(X)
        X = X.reshape((len(X), 1))
        n = X.shape[0]

        fval = -np.exp(-X) * np.sin(2 * np.pi * X)

        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, 0.1 * self.sd, n).reshape(n, 1)

        return fval.reshape(n, 1) + noise

    def __call__(self, X):
        """Evaluation method for botorch (torch.Tensor input)"""
        # Check if input is tensor
        is_tensor = torch.is_tensor(X)

        # Convert to numpy if torch tensor
        if is_tensor:
            X_np = X.detach().cpu().numpy()
            device = X.device
            dtype = X.dtype
        else:
            X_np = np.asarray(X)

        # Ensure correct shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # Evaluate
        fval = -np.exp(-X_np) * np.sin(2 * np.pi * X_np)

        # Add noise if needed
        if self.sd == 0:
            noise = 0
        else:
            noise = np.random.normal(0, 0.1 * self.sd, X_np.shape[0])

        result = fval.flatten() + noise

        # Return in same format as input
        if is_tensor:
            result_tensor = torch.tensor(result, device=device, dtype=dtype)
            # if result_tensor.shape[0] == 1:
            #     return result_tensor[0]
            return result_tensor
        else:
            # if result.shape[0] == 1:
            #     return result[0]
            return result


class MultiModal15:

    def __init__(self, negate: bool = False, sd: Optional[float] = None):
        # Interface for custom algorithm
        self.bounds_dict = {'x': (-5.0, 5.0)}
        self.input_dim = 1
        self.ismax = -1 if not negate else 1
        self.min = 2.41422  ## approx location of minimum
        self.fstar = -0.03553  ## approx optimal value (not multiplied by ismax)
        self.name = 'multimodal15'

        # Interface for botorch
        self.dim = 1
        self.bounds = torch.tensor([[-5.0], [5.0]], dtype=torch.float64)

        # Noise parameter
        if sd is None or sd == 0:
            self.sd = 0
        else:
            self.sd = sd

    def func(self, X):
        """Evaluation method for custom algorithm (numpy input)"""
        X = np.asarray(X)
        X = X.reshape((len(X), 1))
        n = X.shape[0]

        fval = (X**2 - 5 * X + 6) / (X**2 + 1)

        if self.sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, 0.1 * self.sd, n).reshape(n, 1)

        return fval.reshape(n, 1) + noise

    def __call__(self, X):
        """Evaluation method for botorch (torch.Tensor input)"""
        # Check if input is tensor
        is_tensor = torch.is_tensor(X)

        # Convert to numpy if torch tensor
        if is_tensor:
            X_np = X.detach().cpu().numpy()
            device = X.device
            dtype = X.dtype
        else:
            X_np = np.asarray(X)

        # Ensure correct shape
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # Evaluate
        fval = (X_np**2 - 5 * X_np + 6) / (X_np**2 + 1)

        # Add noise if needed
        if self.sd == 0:
            noise = 0
        else:
            noise = np.random.normal(0, 0.1 * self.sd, X_np.shape[0])

        result = fval.flatten() + noise

        # Return in same format as input
        if is_tensor:
            result_tensor = torch.tensor(result, device=device, dtype=dtype)
            # if result_tensor.shape[0] == 1:
            #     return result_tensor[0]
            return result_tensor
        else:
            # if result.shape[0] == 1:
            #     return result[0]
            return result