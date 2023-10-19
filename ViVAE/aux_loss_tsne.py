"""
Re-implementation of the t-SNE algorithm: van der Maaten, L. and Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008.

Copyright 2023 David Novak

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from numba import jit

MAX = np.log(sys.float_info.max) / 2.0

@jit(nopython=True)
def cond_entropy_and_cond_prob(x=np.array([]), beta=1.) -> np.float64:
    ## x is row of pairwise distance matrix without i-th entry
    ## beta is precision parameter (\beta = 1/(2*\sigma^{2}) where \sigma is variance)
    p = -x * beta
    offset = MAX - max(p)

    ## Numerator of conditional probabilities:
    p = np.exp(p + offset)
    ## Denominator of conditional probabilities:
    sum_p = np.sum(p)

    ## S_{i} = \sum_{k\neq i} exp(-||x_{i} - x_{k}||^{2} / 2 \sigma_{i}^{2})
    ## h = log(S_{i}) + \beta_{i} \sum_{j} p_{j|i} ||x_{i} - x_{j}||^{2}
    h = np.log(sum_p) - offset + beta * np.sum(np.multiply(x, p)) / sum_p

    ## Normalise p to probabilities
    p /= sum_p

    return h, p

@jit(nopython=True)
def cond_probs(x, dist_x, perp=30., tolerance=1e-4, max_iter=50) -> np.float64:
    n = x.shape[0]

    ## Initialise conditional probabilities and entropies
    p = np.zeros((n,n), dtype=np.float64)
    log_perp = np.log(perp)

    d_i = np.zeros(n-1, dtype=np.float64)

    ## Binary search for \beta_{i} (precision) values:
    ## \beta_{i} = 1/(2*\sigma_{i}^{2}) where \sigma_{i} is variance
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        beta = 1.

        ## Non-zero distances to vantage-point
        k = 0
        for j in range(n):
            if j!=i:
                d_i[k] = dist_x[i,j]
                k += 1
        ## Could use this in a non-numba setting:
        # d_i = dist_x[i, np.concatentate((np.r_[0:i], np.r_[(i+1):n]))]
        h_i, p_i = cond_entropy_and_cond_prob(d_i, beta)

        h_diff = h_i - log_perp

        search_iter = 0
        while np.abs(h_diff) > tolerance and search_iter < max_iter:
            if h_diff > 0:
                betamin = beta
                if np.isfinite(betamax):
                    beta = (beta+betamax)/2.
                else: # not np.isfinite(beta_max)
                    beta *= 2.
            else: # h_diff <= 0
                betamax = beta
                if np.isfinite(betamin):
                    beta = (beta+betamin)/2.
                else: # not np.isfinite(sigma_min)
                    beta /= 2.
            ## Recompute entropy and probability values for this beta
            h_i, p_i = cond_entropy_and_cond_prob(d_i, beta)
            h_diff = h_i - log_perp
            search_iter += 1

        k = 0
        for j in range(n):
            if j!=i:
                p[i, j] = p_i[k]
                k += 1
        # p[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = p_i

    return p

def pairwise_distances(x):
    """
    Compute pairwise distances between points

    - x: coordinate matrix (tf.Tensor/np.ndarray)
    """
    sum_x = tf.reduce_sum(tf.square(x), 1)
    return tf.constant(-2.0, tf.float64) * tf.matmul(x, x, transpose_b=True) + tf.reshape(sum_x, [-1, 1]) + sum_x