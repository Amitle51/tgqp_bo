#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:28:50 2024

@author: aripakman
"""


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
# from numba import njit
from scipy.linalg import block_diag, cho_solve, cholesky



def compute_M(xdiff2, jitter, p, a, b,):
    K = a * np.exp(-0.5 * xdiff2 / b) + jitter * np.eye(p)
    L = cholesky(K, lower=True)
    K_inv = cho_solve((L, True), np.eye(K.shape[0]))
    return K_inv


class getM_RBF:
    def __init__(self, u, jitter):
        self.xdiff2 = np.square(euclidean_distances(u))
        self.jitter = jitter
        self.p = u.shape[0]  # size of each block

    # @profile
    def M(self, a=1, b=.01, is_cov=False):
        K_inv = compute_M(self.xdiff2, self.jitter, self.p, a, b)
        return K_inv


class getM_Matern:
    def __init__(self, xs, dim, p):
        # Split xs into dim equal parts
        xs_blocks = np.array_split(xs, dim, axis=0)
        self.p = p
        self.xdiff2 = []
        for xs_block in xs_blocks:
            xdiff2_block = euclidean_distances(xs_block)
            self.xdiff2.append(xdiff2_block)

        self.jitter = 1e-3
        self.d = xs_blocks[0].shape[0]  # size of each block
        self.dim = dim

    def M(self, a=1, b=.01, is_cov=False):
        if np.isscalar(a):
            a = np.full(self.dim, a)
        if np.isscalar(b):
            b = np.full(self.dim, b)

        K_blocks = []
        for i, xdiff2 in enumerate(self.xdiff2):
            if self.p == 1:
                K_block = a[i] * (1 + (np.sqrt(3)*xdiff2)/b[i]) * np.exp(-(np.sqrt(3)*xdiff2)/b[i]) + self.jitter * np.eye(self.d)
            elif self.p == 2:
                K_block = (a[i] * (1 + (np.sqrt(5) * xdiff2)/b[i] + (5*np.square(xdiff2))/(3*np.square(b[i]))) * np.exp(-(np.sqrt(5)*xdiff2)/b[i]) +
                           self.jitter * np.eye(self.d))
            K_blocks.append(K_block)
        K = block_diag(*K_blocks)

        if is_cov:
            return K
        return np.linalg.inv(K)


class getM_exp:
    def __init__(self, xs):
        if len(xs.shape) == 1:
            xs = xs.reshape(-1, 1)
        self.xdiff = euclidean_distances(xs)
        self.jitter = 0.000000001
        self.d = xs.shape[0]

    def M(self, a=1, b=.01, is_cov=False):
        K = a * np.exp(-self.xdiff / b) + self.jitter * np.eye(self.d)
        if is_cov:
            return K
        return np.linalg.inv(K)


