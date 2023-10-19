"""
Code from TriMap: Amid, E. and Warmuthm, M. K. (2019). TriMap: Large-scale Dimensionality Reduction Using Triplets. arXiv preprint arXiv:1910.00204.

Copyright 2018 Ehsan Amid

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

from numba import njit, prange
import numpy as np

def tempered_log(x, t):
    """Tempered log with temperature t"""
    if np.abs(t - 1.0) < 1e-5:
        return np.log(x)
    else:
        return 1.0 / (1.0 - t) * (np.power(x, 1.0 - t) - 1.0)

@njit("f4(f4[:])")
def l2_norm(x):
    """L2 norm of a vector."""
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i] ** 2
    return np.sqrt(result)

@njit("f4(f4[:],f4[:])")
def euclid_dist(x1, x2):
    """Euclidean distance between two vectors."""
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)

@njit()
def calculate_dist(x1, x2, distance_index):
    if distance_index == 0:
        return euclid_dist(x1, x2)

@njit()
def rejection_sample(n_samples, max_int, rejects):
    """Rejection sampling.
    Samples "n_samples" integers from a given interval [0,max_int] while
    rejecting the values that are in the "rejects".
    """
    result = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(max_int)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(rejects.shape[0]):
                if j == rejects[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result

@njit("i4[:,:](f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True)
def sample_knn_triplets(P, nbrs, n_inliers, n_outliers):
    """Sample nearest neighbors triplets based on the similarity values in P.
    Input
    ------
    nbrs: Nearest neighbors indices for each point. The similarity values
        are given in matrix P. Row i corresponds to the i-th point.
    P: Matrix of pairwise similarities between each point and its neighbors
        given in matrix nbrs
    n_inliers: Number of inlier points
    n_outliers: Number of outlier points
    Output
    ------
    triplets: Sampled triplets
    """
    n, n_neighbors = nbrs.shape
    triplets = np.empty((n * n_inliers * n_outliers, 3), dtype=np.int32)
    for i in prange(n):
        sort_indices = np.argsort(-P[i])
        for j in prange(n_inliers):
            sim = nbrs[i][sort_indices[j + 1]]
            samples = rejection_sample(n_outliers, n, nbrs[i][sort_indices[: j + 2]])
            for k in prange(n_outliers):
                index = i * n_inliers * n_outliers + j * n_outliers + k
                out = samples[k]
                triplets[index][0] = i
                triplets[index][1] = sim
                triplets[index][2] = out
    return triplets


@njit("f4[:,:](f4[:,:],i4,f4[:],i4)", parallel=True, nogil=True)
def sample_random_triplets(X, n_random, sig, distance_index):
    """Sample uniformly random triplets.
    Input
    ------
    X: Instance matrix or pairwise distances
    n_random: Number of random triplets per point
    sig: Scaling factor for the distances
    distance_index: index of the distance measure
    Output
    ------
    rand_triplets: Sampled triplets
    """
    n = X.shape[0]
    rand_triplets = np.empty((n * n_random, 4), dtype=np.float32)
    for i in prange(n):
        for j in prange(n_random):
            sim = np.random.choice(n)
            while sim == i:
                sim = np.random.choice(n)
            out = np.random.choice(n)
            while out == i or out == sim:
                out = np.random.choice(n)
            if distance_index == -1:
                d_sim = X[i, sim]
            else:
                d_sim = calculate_dist(X[i], X[sim], distance_index)
            p_sim = -(d_sim**2) / (sig[i] * sig[sim])
            if distance_index == -1:
                d_out = X[i, out]
            else:
                d_out = calculate_dist(X[i], X[out], distance_index)
            p_out = -(d_out**2) / (sig[i] * sig[out])
            if p_sim < p_out:
                sim, out = out, sim
                p_sim, p_out = p_out, p_sim
            rand_triplets[i * n_random + j][0] = i
            rand_triplets[i * n_random + j][1] = sim
            rand_triplets[i * n_random + j][2] = out
            rand_triplets[i * n_random + j][3] = p_sim - p_out
    return rand_triplets

@njit("f4[:,:](f4[:,:],f4[:],i4[:,:])", parallel=True, nogil=True)
def find_p(knn_distances, sig, nbrs):
    """Calculates the similarity matrix P.
    Input
    ------
    knn_distances: Matrix of pairwise knn distances
    sig: Scaling factor for the distances
    nbrs: Nearest neighbors
    Output
    ------
    P: Pairwise similarity matrix
    """
    n, n_neighbors = knn_distances.shape
    P = np.zeros((n, n_neighbors), dtype=np.float32)
    for i in prange(n):
        for j in prange(n_neighbors):
            P[i][j] = -knn_distances[i][j] ** 2 / (sig[i] * sig[nbrs[i][j]])
    return P

@njit("f4[:](i4[:,:],f4[:,:],i4[:,:],f4[:],f4[:])", parallel=True, nogil=True)
def find_weights(triplets, P, nbrs, outlier_distances, sig):
    """Calculates the weights for the sampled nearest neighbors triplets.
    Input
    ------
    triplets: Sampled triplets
    P: Pairwise similarity matrix
    nbrs: Nearest neighbors
    outlier_distances: Matrix of pairwise outlier distances
    sig: Scaling factor for the distances
    Output
    ------
    weights: Weights for the triplets
    """
    n_triplets = triplets.shape[0]
    weights = np.empty(n_triplets, dtype=np.float32)
    for t in prange(n_triplets):
        i = triplets[t][0]
        sim = 0
        while nbrs[i][sim] != triplets[t][1]:
            sim += 1
        p_sim = P[i][sim]
        p_out = -outlier_distances[t] ** 2 / (sig[i] * sig[triplets[t][2]])
        weights[t] = p_sim - p_out
    return weights

def generate_triplets_known_knn(
    X,
    knn_nbrs,
    knn_distances,
    n_inliers,
    n_outliers,
    n_random,
    pairwise_dist_matrix=None,
    distance="euclidean",
    verbose=False,
    weight_temp=0.5,
    seed = None # random seed support added by David Novak for cyen
):
    all_distances = pairwise_dist_matrix is not None
    if all_distances:
        distance = "other"
    distance_dict = {
        "euclidean": 0,
        "manhattan": 1,
        "angular": 2,
        "hamming": 3,
        "other": -1,
    }
    distance_index = distance_dict[distance]
    if knn_nbrs[0, 0] != 0:
        knn_nbrs = np.hstack(
            (np.array(range(knn_nbrs.shape[0]))[:, np.newaxis], knn_nbrs)
        ).astype(np.int32)
        knn_distances = np.hstack(
            (np.zeros((knn_distances.shape[0], 1)), knn_distances)
        ).astype(np.float32)
    sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)  # scale parameter
    P = find_p(knn_distances, sig, knn_nbrs)
    triplets = sample_knn_triplets(P, knn_nbrs, n_inliers, n_outliers)
    n_triplets = triplets.shape[0]
    outlier_distances = np.empty(n_triplets, dtype=np.float32)
    for t in range(n_triplets):
        if all_distances:
            outlier_distances[t] = pairwise_dist_matrix[triplets[t, 0], triplets[t, 2]]
        else:
            outlier_distances[t] = calculate_dist(
                X[triplets[t, 0], :], X[triplets[t, 2], :], distance_index
            )
    weights = find_weights(triplets, P, knn_nbrs, outlier_distances, sig)
    if n_random > 0:
        if seed is not None:
            np.random.seed(seed)
        if all_distances:
            rand_triplets = sample_random_triplets(
                pairwise_dist_matrix, n_random, sig, distance_index
            )
        else:
            rand_triplets = sample_random_triplets(X, n_random, sig, distance_index)
        rand_weights = rand_triplets[:, -1]
        rand_triplets = rand_triplets[:, :-1].astype(np.int32)
        triplets = np.vstack((triplets, rand_triplets))
        weights = np.hstack((weights, rand_weights))
    weights[np.isnan(weights)] = 0.0
    weights -= np.min(weights)
    weights = tempered_log(1.0 + weights, weight_temp)
    return (triplets, weights)





