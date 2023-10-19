"""
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

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.util import deprecation
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import datetime
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from typing import Optional,Union

### Imports ----

## k-NN graph building and loading (for smoothing, triplet and TriMap loss)
from .aux_knn import make_knn

## Network building
from .aux_nn import Encoder, Sampler, Decoder, eps_sq

## t-SNE similarity loss helpers
from .aux_loss_tsne import cond_probs, pairwise_distances

## TriMap loss helper
from .aux_loss_trimap import generate_triplets_known_knn

LOSS_TERMS = ['reconstruction', 'kldiv', 'triplet', 'trimap', 'quartet', 'quintet', 'sextet', 'tsne']
PALETTE = [
            '#000000', '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059',
            '#7A4900', '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87',
            '#5A0007', '#809693', '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80',
            '#61615A', '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9', '#B903AA', '#D16100',
            '#DDEFFF', '#000035', '#7B4F4B', '#A1C299', '#300018', '#0AA6D8', '#013349', '#00846F',
            '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744', '#C0B9B2', '#C2FF99', '#001E09',
            '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1', '#788D66',
            '#885578', '#FAD09F', '#FF8A9A', '#D157A0', '#BEC459', '#456648', '#0086ED', '#886F4C',
            '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', '#FF913F', '#938A81',
            '#575329', '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00',
            '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C', '#772600', '#D790FF', '#9B9700',
            '#549E79', '#FFF69F', '#201625', '#72418F', '#BC23FF', '#99ADC0', '#3A2465', '#922329',
            '#5B4534', '#FDE8DC', '#404E55', '#0089A3', '#CB7E98', '#A4E804', '#324E72', '#6A3A4C'
        ]


### Distance function ----

def euclidean_distance(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Compute Euclidean distance between tensors

    - x, y: tensor (coordinates of a point or a row-wise batch of points) (tf.Tensor)
    """
    return K.sqrt(K.maximum(K.sum(K.square(x-y), axis=1, keepdims=False), 1e-9)) # prevent NaN

### TriMap loss helpers ----

def trimap_similarity(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Compute TriMap similarity between tensors

    - x, y: tensor (coordinates of a point or a row-wise batch of points) (tf.Tensor)
    """

    dist = euclidean_distance(x, y)
    res = tf.math.pow(dist, tf.constant(2.0, dtype=tf.float64))
    res = tf.math.add(res, tf.constant(1.0, dtype=tf.float64))
    res = tf.math.pow(res, tf.constant(-1.0, dtype=tf.float64))
    return res
    
### Scale-agnostic loss term helpers ----

def quartet_norm_factor(x: list, f=euclidean_distance) -> tf.Tensor:
    """
    Compute quartet loss normalisation factor

    Coputes denominator of a quartet-normalised distance. This is the sum of all pairwise distances within the quartet

    - x: list of 4 `tf.Tensor`s (a quartet composed of 4 points or row-wise batches of points) (list)
    - f: differentiable distance function for tensors (function)
    """

    return f(x[0],x[1])+f(x[1],x[2])+f(x[2],x[3])+f(x[0],x[2])+f(x[0],x[3])+f(x[1],x[3])

def quintet_norm_factor(x: list, f=euclidean_distance) -> tf.Tensor:
    """
    Compute quintet loss normalisation factor

    Coputes denominator of a quintet-normalised distance. This is the sum of all pairwise distances within the quintet

    - x: list of 5 `tf.Tensor`s (a quintet composed of 5 points or row-wise batches of points) (list)
    - f: differentiable distance function for tensors (function)
    """
    
    return f(x[0],x[1])+f(x[1],x[2])+f(x[2],x[3])+f(x[3],x[4])+f(x[0],x[4])+f(x[0],x[2])+f(x[0],x[3])+f(x[1],x[4])+f(x[1],x[3])+f(x[2],x[4])

def sextet_norm_factor(x: list, f=euclidean_distance) -> tf.Tensor:
    """
    Compute sextet loss normalisation factor

    Coputes denominator of a sectet-normalised distance. This is the sum of all pairwise distances within the sextet

    - x: list of 6 `tf.Tensor`s (a sextet composed of 6 points or row-wise batches of points) (list)
    - f: differentiable distance function for tensors (function)
    """
    return f(x[0],x[1])+f(x[1],x[2])+f(x[2],x[3])+f(x[3],x[4])+f(x[4],x[5])+f(x[0],x[5])+f(x[0],x[2])+f(x[0],x[3])+f(x[0],x[4])+f(x[1],x[5])+f(x[1],x[3])+f(x[1],x[4])+f(x[2],x[4])+f(x[2],x[5])+f(x[3],x[5])

def quartet_normalised_distance(
    x:       tf.Tensor,
    z:       tf.Tensor,
    i:       int,
    j:       int,
    x_denom: tf.Tensor,
    z_denom: tf.Tensor,
    f = euclidean_distance
) -> tf.Tensor:
    """
    Compute quartet-normalised distance

    Computes value of special distance function between points for purposes of the quartet loss.

    - x: row-wise coordinates of a batch of high-dimensional points (tf.Tensor)
    - z: row-wise coordinates of a batch of low-dimensional point (tf.Tensor)
    - i, j: indices of points in `x` and `z` (int)
    - x_denom: quartet-normalisation factor for high-dimensional points (tf.Tensor)
    - z_denom: quartet-normalisation factor for low-dimensional points (tf.Tensor)
    - f: differentiable distance function for tensors (function)
    """
    
    d_x = f(x[i], x[j]) / x_denom
    d_z = f(z[i], z[j]) / z_denom
    D = tf.math.pow(d_x - d_z, tf.constant(2.0, dtype=tf.float64))
    return D

def quintet_normalised_distance(
    x:       tf.Tensor,
    z:       tf.Tensor,
    i:       int,
    j:       int,
    x_denom: tf.Tensor,
    z_denom: tf.Tensor,
    f = euclidean_distance
) -> tf.Tensor:
    """
    Compute quintet-normalised distance

    Computes value of special distance function between points for purposes of the quintet loss.

    - x: row-wise coordinates of a batch of high-dimensional points (tf.Tensor)
    - z: row-wise coordinates of a batch of low-dimensional point (tf.Tensor)
    - i, j: indices of points in `x` and `z` (int)
    - x_denom: quintet-normalisation factor for high-dimensional points (tf.Tensor)
    - z_denom: quintet-normalisation factor for low-dimensional points (tf.Tensor)
    - f: differentiable distance function for tensors (function)
    """
    return quartet_normalised_distance(x, z, i, j, x_denom, z_denom)

def sextet_normalised_distance(
    x:       tf.Tensor,
    z:       tf.Tensor,
    i:       int,
    j:       int,
    x_denom: tf.Tensor,
    z_denom: tf.Tensor,
    f = euclidean_distance
) -> tf.Tensor:
    """
    Compute sextet-normalised distance

    Computes value of special distance function between points for purposes of the sextet loss.

    - x: row-wise coordinates of a batch of high-dimensional points (tf.Tensor)
    - z: row-wise coordinates of a batch of low-dimensional point (tf.Tensor)
    - i, j: indices of points in `x` and `z` (int)
    - x_denom: sextet-normalisation factor for high-dimensional points (tf.Tensor)
    - z_denom: sextet-normalisation factor for low-dimensional points (tf.Tensor)
    - f: differentiable distance function for tensors (function)
    """
    return quartet_normalised_distance(x, z, i, j, x_denom, z_denom)

def quartet_cost(
    x_q0: tf.Tensor,
    x_q1: tf.Tensor,
    x_q2: tf.Tensor,
    x_q3: tf.Tensor,
    z_q0: tf.Tensor,
    z_q1: tf.Tensor,
    z_q2: tf.Tensor,
    z_q3: tf.Tensor,
    f = euclidean_distance
) -> tf.Tensor:
    """
    Compute batch-wise cost value for point quartets in a batch

    - x_q0, x_q1, x_q2, x_q3: row-wise coordinates of batches of high-dimensional points with indices 0, 1, 2 and 3 within their respective quartets (tf.Tensor)
    - z_q0, z_q1, z_q2, z_q3: row-wise coordinates of batches of low-dimensional points with indices 0, 1, 2 and 3 within their respective quartets (tf.Tensor)
    - f:                      differentiable distance function for tensors (function)
    """

    x = [x_q0, x_q1, x_q2, x_q3]
    z = [z_q0, z_q1, z_q2, z_q3]
    x_denom = quartet_norm_factor(x)
    z_denom = quartet_norm_factor(z)
    def d(i, j):
        return quartet_normalised_distance(x=x, z=z, i=i, j=j, x_denom=x_denom, z_denom=z_denom, f=f)
    return tf.reduce_mean(d(0,1)+d(1,2)+d(2,3)+d(0,2)+d(0,3)+d(1,3))

def quintet_cost(
    x_q0: tf.Tensor,
    x_q1: tf.Tensor,
    x_q2: tf.Tensor,
    x_q3: tf.Tensor,
    x_q4: tf.Tensor,
    z_q0: tf.Tensor,
    z_q1: tf.Tensor,
    z_q2: tf.Tensor,
    z_q3: tf.Tensor,
    z_q4: tf.Tensor,
    f = euclidean_distance
) -> tf.Tensor:
    """
    Compute batch-wise cost value for point quintets in a batch

    - x_q0, x_q1, x_q2, x_q3, x_q4: row-wise coordinates of batches of high-dimensional points with indices 0, 1, 2, 3 and 4 within their respective quintets (tf.Tensor)
    - z_q0, z_q1, z_q2, z_q3, z_q4: row-wise coordinates of batches of low-dimensional points with indices 0, 1, 2, 3 and 4 within their respective quintets (tf.Tensor)
    - f:                            differentiable distance function for tensors (function)
    """
    
    x = [x_q0, x_q1, x_q2, x_q3, x_q4]
    z = [z_q0, z_q1, z_q2, z_q3, z_q4]
    x_denom = quintet_norm_factor(x)
    z_denom = quintet_norm_factor(z)
    def d(i, j):
        return quintet_normalised_distance(x=x, z=z, i=i, j=j, x_denom=x_denom, z_denom=z_denom)
    return tf.reduce_mean(d(0,1)+d(1,2)+d(2,3)+d(3,4)+d(0,4)+d(0,2)+d(0,3)+d(1,4)+d(1,3)+d(2,4))

def sextet_cost(
    x_q0: tf.Tensor,
    x_q1: tf.Tensor,
    x_q2: tf.Tensor,
    x_q3: tf.Tensor,
    x_q4: tf.Tensor,
    x_q5: tf.Tensor,
    z_q0: tf.Tensor,
    z_q1: tf.Tensor,
    z_q2: tf.Tensor,
    z_q3: tf.Tensor,
    z_q4: tf.Tensor,
    z_q5: tf.Tensor,
    f = euclidean_distance
) -> tf.Tensor:
    """
    Compute batch-wise cost value for point sextets in a batch

    - x_q0, x_q1, x_q2, x_q3, x_q4, x_q5: row-wise coordinates of batches of high-dimensional points with indices 0, 1, 2, 3, 4 and 5 within their respective sextets (tf.Tensor)
    - z_q0, z_q1, z_q2, z_q3, z_q4, z_q5: row-wise coordinates of batches of low-dimensional points with indices 0, 1, 2, 3, 4 and 5 within their respective sextets (tf.Tensor)
    - f:                                  differentiable distance function for tensors (function)
    """
    x = [x_q0, x_q1, x_q2, x_q3, x_q4, x_q5]
    z = [z_q0, z_q1, z_q2, z_q3, z_q4, z_q5]
    x_denom = quintet_norm_factor(x)
    z_denom = quintet_norm_factor(z)
    def d(i, j):
        return sextet_normalised_distance(x=x, z=z, i=i, j=j, x_denom=x_denom, z_denom=z_denom)
    return tf.reduce_mean(d(0,1)+d(1,2)+d(2,3)+d(3,4)+d(4,5)+d(0,5)+d(0,2)+d(0,3)+d(0,4)+d(1,5)+d(1,3)+d(1,4)+d(2,4)+d(2,5)+d(3,5))

### Model definition ----

class ViVAE_network(tf.keras.Model):
    """
    ViVAE neural network model class

    Parametric dimension-reduction model with a combined loss function.
    """
    def __init__(
        self,
        data:         Optional[list] = np.ndarray,
        full_dim:     Optional[int] = int,
        enc_shape:    list = [32,64,128,32],
        dec_shape:    list = [32,128,64,32],
        latent_dim:   int = 2,
        dropout_rate: float = 0.00,
        activation:   str = 'selu',
        gm_prior:     bool = True,
        verbose:      bool = False,
        **kwargs
    ):
        """
        Instantiate ViVAE network model

        Constructor for a `cyen_network` object. Do not use this manually, use the `ViVAE` model constructor instead.

        - data:         optional high-dimensional data coordinate matrix (`full_dim` can be specified instead) (nd.nparray)
        - full_dim:     optional `data` dimensionality `data.shape[1]` (`data` can be specified instead) (int)
        - enc_shape:    list of consecutive node counts defining the size of each layer of the encoder (list of ints)
        - dec_shape:    list of consecutive node counts defining the size of each layer of the decoder (list of ints)
        - latent_dim:   dimensionality of latent projection of `data` (int)
        - dropout_rate: rate of dropout for regularisation (float)
        - activation:   activation function in each node of the encoder and decoder networks: eg. 'selu', 'relu', 'sigmoid' (str)
        - gm_prior:     use Gaussian mixture instead of isotropic Gaussian? (bool)
        - verbose:      print progress messages during instantiation? (bool)
        """
        super(ViVAE_network, self).__init__(name='ViVAE', **kwargs)

        if data is None and full_dim is None:
            raise ValueError('Either `data` or `full_dim` must be specified to build network')
        
        if full_dim is None:
            full_dim = data.shape[1]
        if latent_dim < 1 or latent_dim >= full_dim:
            raise ValueError('Invalid latent representation dimensionality')

        self.knn_required = False
        self.knn_idcs = None
        self.knn_dist = None

        self.verbose = verbose
        self.attach_data(data)
        self.full_dim = full_dim
        self.enc_shape = enc_shape
        self.dec_shape = dec_shape
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.gm_prior = gm_prior
        self.encoder = Encoder(latent_dim=self.latent_dim, shape=self.enc_shape, dropout=self.dropout_rate, activation=self.activation)
        self.sampler = Sampler(gm_prior=gm_prior)

    def set_loss_weights(self, **kwargs):
        """
        Setter for loss function term weights
        """
        w = dict()
        for l in LOSS_TERMS:
            w[l] = 0.
        if kwargs is None:
            w['reconstruction'] = 1.
            w['kldiv'] = 1.
            self.w = w
        else:
            for l in LOSS_TERMS:
                if l in kwargs:
                    w[l] = kwargs[l]
        self.w = w

    def attach_data(self, X):
        """
        Attach high-dimensional data
        """
        self.data = copy.deepcopy(X)

    def attach_knn(
        self,
        knn:     Optional[list] = None,
        x:       Optional[np.ndarray] = None,
        fname:   Optional[str] = None,
        k:       int = 100
    ):
        """
        Attach or construct a k-nearest-neighbour graph

        - knn:   optional k-NNG object, as generated with `make_knn` (list)
        - x:     optional coordinate matrix of high-dimensional data to construct k-NNG for (int)
        - fname: optional name of .npy file with save k-NNG object (str)
        - k:     nearest neighbour count if a new k-NNG is to be constructed (int)
        """
        verbose = self.verbose
        
        if self.knn_idcs is None:
            if knn is None:
                if verbose:
                    print('Creating new k-NNG')
                if x is None:
                    x = self.data
                ncol = x.shape[1]
                if k > (ncol-1):
                    k = ncol-1
                knn = make_knn(x=x, fname=fname, k=k, verbose=verbose)
            else:
                if verbose:
                    print('Using pre-computed k-NNG')
            self.knn_idcs = np.array(knn[0], dtype=np.int64)
            self.knn_dist = np.array(knn[1], dtype=np.int64)
            self.k = self.knn_idcs.shape[1]

    def get_triplet_data(
        self,
        method:      str = 'simple',
        k:           int = 50,
        l:           Optional[int] = None,
        n_inliers:   int = 3,
        n_outliers:  int = 3,
        n_random:    int = 1,
        weight_temp: float = 0.5,
        max_points:  Union[Optional[int], str] = None,
        shuffle:     bool = True,
        seed:        Optional[int] = None
    ) -> list:
        """
        Sample indices and points triplets for Siamese network training

        Sampling can be done using the 'simple' method or the 'trimap' method.
        
        The 'simple' method picks a random positive reference from inside the k-ary neighbourhood of each vantage point.
        The negative reference point is either picked from anywhere outside the k-ary neighbourhood or from the neighbourhood
        of points with rank between k and l (l > k).

        The 'trimap' method uses the methodology from the TriMap dimension-reduction algorithm, resulting in multiple
        triplets per anchor (vantage point) and a vector of weights per triplet. This enlarges the input dataset, and a maximum
        number of points can be set (`max_points`) to limit this.

        - method:      triplet sampling method, either 'simple' or 'trimap' (str)
        - k:           positive reference neighbour rank for 'simple' sampling (int)
        - l:           optional negative reference neighbour rank for 'simple' sampling (int)
        - n_inliers:   nearest-neighbour count for 'trimap' sampling (int)
        - n_outliers:  triplet count per nearest neighbour for 'trimap' sampling (int)
        - n_random:    random triplet count per point for 'trimap' sampling (int)
        - weight_temp: tempered logarithm temperature parameter for 'trimap' sampling (float)
        - max_points:  maximum number of triplets generated with 'trimap' sampling. Can be 'auto' for dataset size times 3 or None for unlimited (int/str)
        - shuffle:     shuffle triplets in 'trimap' sampling? (Strongly encouraged.) (bool)
        - seed:        optional random seed for reproducibility (int)

        Returns:
        List containing 7 objects:
            - matrix of anchor point coordinates (np.ndarray)
            - matrix of positive reference point coordinates (np.ndarray)
            - matrix of negative reference point coordinates (np.ndarray)
            - weight vector per triplet (np.array)
            - vector of anchor point indices (np.array)
            - vector of positive reference point indices (np.array)
            - vector of negative reference point indices (np.array)
        """

        if method not in ['simple', 'trimap']:
            raise ValueError('Triplet sampling `method` must be either "simple" or "trimap"')
        if self.knn_idcs is None or self.knn_dist is None:
            raise AttributeError('k-nearest neighbour graph needs to be constructed before triplet sampling')
        if seed is not None:
            np.random.seed(seed)

        ## Generate point indices and weights

        if method == 'simple':
            n = self.knn_idcs.shape[0]
            pts = np.arange(n)
            if self.verbose:
                print('Sampling positive reference points using "simple" triplet sampling')
            idcs_pos = np.array([np.random.choice(self.knn_idcs[idx_pt][range(1,k+1)]) for idx_pt in pts]).astype(int)
            if self.verbose:
                print('Sampling negative reference points using "simple" triplet sampling')
            if l is None:
                idcs_neg = np.array([np.random.choice(np.delete(pts, self.knn_idcs[idx_pt][range(k+1)].astype(int))) for idx_pt in pts]).astype(int)
            else:
                idcs_neg = np.array([np.random.choice(self.knn_idcs[idx_pt][range(k+1, l+1)].astype(int)) for idx_pt in pts]).astype(int)
            idcs_anch = pts
            weights = np.repeat(1., repeats=idcs_neg.shape[0])
        elif method == 'trimap':
            triplets, weights = generate_triplets_known_knn(
                X=self.data.astype(np.float32),
                knn_nbrs=self.knn_idcs[:,range(k+1)].astype(np.int32),
                knn_distances=self.knn_dist[:,range(k+1)].astype(np.float32),
                n_inliers=n_inliers,
                n_outliers=n_outliers,
                n_random=n_random,
                pairwise_dist_matrix=None,
                distance='euclidean',
                verbose=False,
                weight_temp=weight_temp,
                seed=seed
            )
            mu = np.mean(weights)
            sig = np.std(weights)
            weights = (weights-mu)/sig
            if shuffle:
                idcs = np.arange(triplets.shape[0])
                np.random.shuffle(idcs)
                triplets = triplets[idcs]
                weights = weights[idcs]
            if max_points is not None:
                if max_points=='auto':
                    max_points = self.data.shape[0]*3
                if triplets.shape[0]>max_points:
                    np.random.seed(seed)
                    idcs = np.random.choice(np.arange(triplets.shape[0]), size=max_points, replace=False)
                    np.random.shuffle(idcs)
                    triplets = triplets[idcs]
                    weights = weights[idcs]
            idcs_anch = triplets[:, 0]
            idcs_pos  = triplets[:, 1]
            idcs_neg  = triplets[: ,2]
        
        ## Get point coordinate matrices

        x_anch = self.data[idcs_anch].astype(np.float64)
        x_pos  = self.data[idcs_pos].astype(np.float64)
        x_neg  = self.data[idcs_neg].astype(np.float64)

        return [x_anch, x_pos, x_neg, weights, idcs_anch, idcs_pos, idcs_neg]

    def loss_reconstruction(self, y_true: tf.Tensor, y_pred: tf.Tensor, bce: bool = False) -> tf.Tensor:
        """
        Compute reconstruction loss as mean squared error

        - y_true: batch of high-dimensional points (tf.Tensor)
        - y_pred: reconstruction of `y_true` generated by the decoder (tf.Tensor)
        - bce:    use binary cross-entropy (of scaled data) instead of mean squared error? (bool)
        """

        if bce:
            res = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        else:
            res = tf.reduce_mean((y_true - y_pred) ** 2)
        return res

    def loss_kldiv(self, z_mu: tf.Tensor, z_sigma: tf.Tensor) -> tf.Tensor:
        """
        Compute KL-divergence from latent prior distribution

        - z_mu:    encoded mu (mean) (tf.Tensor)
        - z_sigma: encoded sizma (mean) (tf.Tensor)
        """
        res = -0.5 * tf.reduce_mean(z_sigma + tf.math.log(eps_sq) - tf.square(z_mu) - eps_sq * tf.exp(z_sigma))
        res /= tf.shape(z_sigma)[0]
        return res

    def loss_tsne(
        self,
        x:          tf.Tensor,
        z:          tf.Tensor,
        perplexity: float = 30.,
        tolerance:  float = 1e-4,
        max_iter:   int = 50
    ) -> tf.Tensor:
        """
        Compute t-SNE dissimilarity loss

        Computes batch-wise dissimilarity in local point neighbourhoods between high-dimensional data and lower-dimensional
        projection, the way t-SNE does.

        - x:          row-wise batch of high-dimensional point coordinates (tf.Tensor)
        - z:          row-wise batch of low-dimensional point coordinates (tf.Tensor)
        - perplexity: t-SNE perplexity parameter, governing neighbourhood size (float)
        - tolerance:  tolerance parameter for numerical solution for conditional probabilities  (float)
        - max_iter:   maximum iteration count for numerical solution for conditional probabilities (int)

        Reference: van der Maaten, L. and Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008.
        """

        ## Get squared Euclidean distance matrix for HD
        dist_x = pairwise_distances(x)

        ## Get conditional probabilities for HD using Gaussian kernel
        p = tf.numpy_function(cond_probs, [x, dist_x, perplexity, tolerance, max_iter], tf.float64)

        ## Symmetrise, normalise and avoid zeroes
        p += tf.transpose(p)
        p /= tf.reduce_sum(p)
        p = tf.math.maximum(p, tf.keras.backend.epsilon())

        ## Get squared Euclidean distance matrix for LD
        dist_z = pairwise_distances(z)

        ## Get probabilities for LD using Student's t kernel
        q = tf.pow(tf.constant(1.0, tf.float64) + dist_z, -(tf.constant(1.0, tf.float64)))

        ## Summed attractive forces
        attract = -tf.reduce_mean(tf.math.multiply(p, tf.math.log(q + tf.keras.backend.epsilon())))
        
        ## Summed repellant forces
        repel = tf.reduce_mean(tf.math.log(tf.reduce_sum(q, 1) - 1 + tf.keras.backend.epsilon()))
        
        return tf.math.add(attract, repel)

    def loss_triplet(
        self,
        z_anch:  tf.Tensor,
        z_pos:   tf.Tensor = None,
        z_neg:   tf.Tensor = None,
        weights: Optional[tf.Tensor] = None,
        margin:  float = 1e-3
    ) -> tf.Tensor:
        """
        Compute ivis triplet loss

        Computes the triplet loss as defined in ivis. This penalises intrusions and extrusions and rewards each anchor point
        being closer to its positive reference point than to its negative reference point.

        - z_anch: row-wise batch of coordinates of anchor points in the lower-dimensional latent projection of original data (tf.Tensor)
        - z_pos: row-wise batch of coordinates of positive reference points corresponding to `z_anch` (tf.Tensor)
        - z_neg: row-wise batch of coordinates of negative reference points corresponding to `z_anch` (tf.Tensor)
        - weights: optional vector of weights per triplet (row of `z_anch`, `z_pos`, `z_neg`) (tf.Tensor)
        - margin: margin between distance to positive reference and negative reference to satisfy the triplet constraint (float)

        Reference: Szubert, B., Cole, J. E., Monaco, C. and Drozdov, I. (2019). Structure-preserving visualisation of high dimensional single-cell datasets. Sci Rep 9: 8914, 2019.
        """
        
        if z_pos is None or z_neg is None:
            return tf.constant(0., dtype=tf.float64)

        ## Compute distances for the anchor-positive-negative triplets
        d_ap = euclidean_distance(z_anch, z_pos)
        d_an = euclidean_distance(z_anch, z_neg)
        d_pn = euclidean_distance(z_pos, z_neg)

        ## Compute triplet loss term
        d_min = K.min(K.concatenate([d_an, d_pn]), keepdims=True)

        if weights is not None:
            d_min = tf.math.multiply(weights, d_min)

        return K.mean(K.maximum(d_ap - d_min + margin, 0))

    def loss_trimap(
        self,
        z_anch:  tf.Tensor,
        z_pos:   tf.Tensor,
        z_neg:   tf.Tensor,
        weights: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute TriMap loss

        Computes the loss used in TriMap, using a t-SNE-inspired similarity function to reward high similarity between anchor and
        positive reference point and low similarity between anchor and negative reference point.

        - z_anch: row-wise batch of coordinates of anchor points in the lower-dimensional latent projection of original data (tf.Tensor)
        - z_pos: row-wise batch of coordinates of positive reference points corresponding to `z_anch` (tf.Tensor)
        - z_neg: row-wise batch of coordinates of negative reference points corresponding to `z_anch` (tf.Tensor)
        - weights: optional vector of weights per triplet (row of `z_anch`, `z_pos`, `z_neg`) (tf.Tensor)

        Reference: Amid, E. and Warmuthm, M. K. (2019). TriMap: Large-scale Dimensionality Reduction Using Triplets. arXiv preprint arXiv:1910.00204.
        """
        if z_pos is None or z_neg is None:
            return tf.constant(0., dtype=tf.float64)

        s_ap = trimap_similarity(z_anch, z_pos)
        s_an = trimap_similarity(z_anch, z_neg)
        if weights is not None:
            res = tf.math.multiply(weights, tf.math.divide(s_an, tf.math.add(s_ap, s_an)))
        else:
            res = tf.math.divide(s_an, tf.math.add(s_ap, s_an))
        return tf.reduce_mean(res)

    def loss_quartet(
        self,
        x:       tf.Tensor,
        z:       tf.Tensor,
        n:       int,
        shuffle: bool = True,
        seed:    Optional[int] = 1
    ) -> tf.Tensor:
        """
        Compute quartet loss

        Computes the scale-agnostic quartet loss, dividing points within a batch into quartets ad hoc.

        - x:       row-wise batch of high-dimensional point coordinates (tf.Tensor)
        - z:       row-wise batch of low-dimensional point coordinates (tf.Tensor)
        - n:       number of rows of `x` or `z` (tf.Tensor)
        - shuffle: shuffle rows of `x` and `z` before determining quartets? Recommended (bool)
        - seed:    optional random seed for shuffling if `shuffle` is set to True (int)
        """
        n_quartet = n // 4
        coef = tf.cast(n, dtype=tf.float64)
        if shuffle:
            idcs = tf.range(start=0, limit=n, dtype=tf.int32)
            idcs_shuffled = tf.random.shuffle(idcs, seed=seed)
            x = tf.gather(x, idcs_shuffled)
            z = tf.gather(z, idcs_shuffled)

        begin_idcs = [[0,0], [n_quartet,0], [n_quartet*2,0], [n_quartet*3,0]]
        x_q = [tf.slice(x, begin=idcs, size=[n_quartet,self.full_dim]) for idcs in begin_idcs]
        z_q = [tf.slice(z, begin=idcs, size=[n_quartet,self.latent_dim]) for idcs in begin_idcs]

        res = tf.math.multiply(quartet_cost(x_q[0], x_q[1], x_q[2], x_q[3], z_q[0], z_q[1], z_q[2], z_q[3]), coef)
        res = tf.math.divide(res, n_quartet)
        return res

    def loss_quintet(
        self,
        x:       tf.Tensor,
        z:       tf.Tensor,
        n:       int,
        shuffle: bool = True,
        seed:    Optional[int] = 1
    ) -> tf.Tensor:
        """
        Compute quintet loss

        Computes the scale-agnostic quintet loss, dividing points within a batch into quintets ad hoc.
        
        - x:       row-wise batch of high-dimensional point coordinates (tf.Tensor)
        - z:       row-wise batch of low-dimensional point coordinates (tf.Tensor)
        - n:       number of rows of `x` or `z` (tf.Tensor)
        - shuffle: shuffle rows of `x` and `z` before determining quartets? Recommended (bool)
        - seed:    optional random seed for shuffling if `shuffle` is set to True (int)
        """
        n_quintet = n // 5
        coef = tf.cast(n, dtype=tf.float64)
        if shuffle:
            idcs = tf.range(start=0, limit=n, dtype=tf.int32)
            idcs_shuffled = tf.random.shuffle(idcs, seed=seed)
            x = tf.gather(x, idcs_shuffled)
            z = tf.gather(z, idcs_shuffled)

        begin_idcs = [[0,0], [n_quintet,0], [n_quintet*2,0], [n_quintet*3,0], [n_quintet*4,0]]
        x_q = [tf.slice(x, begin=idcs, size=[n_quintet,self.full_dim]) for idcs in begin_idcs]
        z_q = [tf.slice(z, begin=idcs, size=[n_quintet,self.latent_dim]) for idcs in begin_idcs]

        res = tf.math.multiply(quintet_cost(x_q[0], x_q[1], x_q[2], x_q[3], x_q[4], z_q[0], z_q[1], z_q[2], z_q[3], z_q[4]), coef)
        res = tf.math.divide(res, n_quintet)
        return res
    
    def loss_sextet(
        self,
        x:       tf.Tensor,
        z:       tf.Tensor,
        n:       int,
        shuffle: bool = True,
        seed:    Optional[int] = 1
    ):
        """
        Compute sextet loss

        Computes the scale-agnostic sextet loss, dividing points within a batch into sextets ad hoc.

        - x:       row-wise batch of high-dimensional point coordinates (tf.Tensor)
        - z:       row-wise batch of low-dimensional point coordinates (tf.Tensor)
        - n:       number of rows of `x` or `z` (tf.Tensor)
        - shuffle: shuffle rows of `x` and `z` before determining quartets? Recommended (bool)
        - seed:    optional random seed for shuffling if `shuffle` is set to True (int)
        """
        n_sextet = n // 6
        coef = tf.cast(n, dtype=tf.float64)
        if shuffle:
            idcs = tf.range(start=0, limit=n, dtype=tf.int32)
            idcs_shuffled = tf.random.shuffle(idcs, seed=seed)
            x = tf.gather(x, idcs_shuffled)
            z = tf.gather(z, idcs_shuffled)

        begin_idcs = [[0,0], [n_sextet,0], [n_sextet*2,0], [n_sextet*3,0], [n_sextet*4,0], [n_sextet*5,0]]
        x_q = [tf.slice(x, begin=idcs, size=[n_sextet,self.full_dim]) for idcs in begin_idcs]
        z_q = [tf.slice(z, begin=idcs, size=[n_sextet,self.latent_dim]) for idcs in begin_idcs]

        res = tf.math.multiply(sextet_cost(x_q[0], x_q[1], x_q[2], x_q[3], x_q[4], x_q[5], z_q[0], z_q[1], z_q[2], z_q[3], z_q[4], z_q[5]), coef)
        res = tf.math.divide(res, n_sextet)
        return res
    
    def call(self, x):
        """
        Run forward pass through cyen model

        - x: list of
            - row-wise coordinate matrix of high-dimensional points (anchor points if Siamese network-training on triplets) (np.ndarray/tf.Tensor)
            - optional row-wise coordinate matrix of positive reference points to `x_anch` for Siamese network-training on triplets (or `np.array([0.])`) (np.ndarray/tf.Tensor)
            - optional row-wise coordinate matrix of negative reference points to `x_anch` for Siamese network-training on triplets (or `np.array([0.]`) (np.ndarray/tf.Tensor)
            - optional vector of weights per triplet for Siamese network-training (or `np.array([0.]`) (np.array/tf.Tensor)
        """

        x_anch = x[0]
        x_pos = x[1]
        x_neg = x[2]
        weights = x[3]
        
        ## Set up Siamese network-training
        if self.w['triplet'] > 0. or self.w['trimap'] > 0.:
            ## Already encode pos & neg refs and sample their latent projections
            z_pos_mu, z_pos_sigma = self.encoder(x_pos) 
            z_neg_mu, z_neg_sigma = self.encoder(x_neg)
            z_pos = self.sampler(z_pos_mu, z_pos_sigma)
            z_neg = self.sampler(z_neg_mu, z_neg_sigma)

        ## Encode HD data
        z_mu, z_sigma = self.encoder(x_anch)

        ## Sample latent projection
        z_anch = self.sampler(z_mu, z_sigma)

        ## Initialise loss function (can be made up of multiple terms)
        l = []

        ## Compute reconstruction loss
        if self.w['reconstruction'] > 0.:
            recon = self.decoder(z_anch)
            val   = self.loss_reconstruction(x_anch, recon)
            self.add_metric(val, aggregation='mean', name='reconstruction')
            l.append(self.w['reconstruction'] * val)
        else:
            self.add_metric(tf.constant(0., dtype=tf.float64), aggregation='mean', name='reconstruction')

        ## Compute KL divergence from prior
        if self.w['kldiv'] > 0.:
            val = self.loss_kldiv(z_mu, z_sigma)
            self.add_metric(val, aggregation='mean', name='kldiv')
            l.append(self.w['kldiv'] * val)
        else:
            self.add_metric(tf.constant(0., dtype=tf.float64), aggregation='mean', name='kldiv')

        ## Compute t-SNE dissimilarity
        if self.w['tsne'] > 0.:
            val = self.loss_tsne(x_anch, z_anch)
            self.add_metric(val, aggregation='mean', name='tsne')
            l.append(self.w['tsne'] * val)
        else:
            self.add_metric(tf.constant(0., dtype=tf.float64), aggregation='mean', name='tsne')

        ## Compute triplet loss
        if self.w['triplet'] > 0.:
            val = self.loss_triplet(z_anch, z_pos, z_neg)
            self.add_metric(val, aggregation='mean', name='triplet')
            l.append(self.w['triplet'] * val)
        else:
            self.add_metric(tf.constant(0., dtype=tf.float64), aggregation='mean', name='triplet')

        ## Compute TriMap loss
        if self.w['trimap'] > 0.:
            val = self.loss_trimap(z_anch, z_pos, z_neg, weights)
            self.add_metric(val, aggregation='mean', name='trimap')
            l.append(self.w['trimap'] * val)
        else:
            self.add_metric(tf.constant(0., dtype=tf.float64), aggregation='mean', name='trimap')

        ## Compute quartet loss
        if self.w['quartet'] > 0.:
            n = tf.shape(x_anch)[0]
            n_samples = 5
            vals = [self.loss_quartet(x_anch, z_anch, n, shuffle=True, seed=x) for x in range(n_samples)]
            val = tf.math.divide(tf.math.add_n(vals), tf.constant(n_samples, dtype=tf.float64))
            self.add_metric(val, aggregation='mean', name='quartet')
            l.append(self.w['quartet'] * val)
        else:
            self.add_metric(tf.constant(0., dtype=tf.float64), aggregation='mean', name='quartet')
        
        ## Compute quintet loss
        if self.w['quintet'] > 0.:
            n = tf.shape(x_anch)[0]
            n_samples = 5
            vals = [self.loss_quintet(x_anch, z_anch, n, shuffle=True, seed=x) for x in range(n_samples)]
            val = tf.math.divide(tf.math.add_n(vals), tf.constant(n_samples, dtype=tf.float64))
            self.add_metric(val, aggregation='mean', name='quintet')
            l.append(self.w['quintet'] * val)
        else:
            self.add_metric(tf.constant(0., dtype=tf.float64), aggregation='mean', name='quintet')
        
        ## Compute sextet loss
        if self.w['sextet'] > 0.:
            n = tf.shape(x_anch)[0]
            n_samples = 5
            vals = [self.loss_sextet(x_anch, z_anch, n, shuffle=True, seed=x) for x in range(n_samples)]
            val = tf.math.divide(tf.math.add_n(vals), tf.constant(n_samples, dtype=tf.float64))
            self.add_metric(val, aggregation='mean', name='sextet')
            l.append(self.w['sextet'] * val)
        else:
            self.add_metric(tf.constant(0., dtype=tf.float64), aggregation='mean', name='sextet')
        
        self.add_loss(l)

        return l

def plot(
    proj:       np.ndarray,
    annot:      np.array,
    unassigned: Optional[Union[str,list]] = None,
    fname:      Optional[str] = None,
    dpi:        int = 120,
    title:      str = '',
    palette:    Optional[list] = None,
    show:       bool = True,
    figsize:    tuple = (16,12),
    point_size: Optional[int] = None,
    no_legend:  bool = False
):
    """
    Plot 2-dimensional projection of high-dimensional data

    - proj:       2-dimensional coordinate matrix (numpy.ndarray)
    - annot:      labels per data point (numpy.array)
    - unassigned: optional names of labels in `annot` that correspond to unlabelled points (str/list)
    - fname:      if specified, name of PNG, PDF or SVG file to save plot (str)
    - dpi:        if `fname` specified, pixel density per inch for the saved figure (int)
    - title:      plot title (str)
    - palette:    optional non-default palette of hex codes for colours per each labelled population (list)
    - show:       whether to display the generated figure (Boolean)
    - figsize:    tuple specifying width and height of plot in inches (tuple)
    - point_size: optional point size parameter (if not given, determined automatically based on the number of points) (int)
    - no_legend:  whether to exclude the legend with labels from `annot` (bool)
    """
    
    if palette is None:
        palette = PALETTE
    fig, ax = plt.subplots(figsize=figsize)
    if point_size is not None:
        s = point_size
    else:
        s = 0.15
        if proj.shape[0] < 10000:
            s = 0.5
        if proj.shape[0] < 7000:
            s = 0.8
        if proj.shape[0] < 5000:
            s = 1.0

    if unassigned is not None:
        multiple = isinstance(unassigned, list) and len(unassigned) > 1
        idcs = np.argwhere(np.isin(annot, np.array(unassigned))).ravel()
        ax.scatter(proj[idcs,0], proj[idcs,1], label = ', '.join(unassigned) if multiple else [unassigned][0], s=s, c='#c9c9c9')
        ann = np.delete(annot, idcs)
        p = np.delete(proj, idcs, axis=0)
    else:
        ann = annot
        p = proj

    idx_pop = 0
    for pop in np.unique(ann):
        idcs = np.where(ann == pop)
        ax.scatter(p[idcs,0], p[idcs,1], label = pop, s=s, c=palette[idx_pop])
        idx_pop += 1
    if not no_legend:
        l = plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
        for handle in l.legendHandles:
            handle.set_sizes([50.0])
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    fig.suptitle(title)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

class ViVAE:
    """
    ViVAE model
    """
    def __init__(
        self,
        full_dim:     int = int,
        enc_shape:    list = [32,64,128,32],
        dec_shape:    list = [32,128,64,32],
        latent_dim:   int = 2,
        dropout_rate: float = 0.,
        activation:   str = 'selu',
        gm_prior:     bool = True
    ):
        """
        Create a ViVAE dimension-reduction model

        - full_dim:     original input data dimensionality (int)
        - enc_shape:    list of consecutive node counts defining the size of each layer of the encoder (list of ints)
        - dec_shape:    list of consecutive node counts defining the size of each layer of the decoder (list of ints)
        - latent_dim:   target dimensionality of the output data projection (int)
        - dropout_rate: rate of dropout for regularisation (float)
        - activation:   activation function in each node of the encoder and decoder networks: eg. 'selu', 'relu', 'sigmoid' (str)
        - gm_prior:     use Gaussian mixture instead of isotropic Gaussian? (bool)
        """
        self.model = ViVAE_network(
            full_dim=    full_dim,
            enc_shape=   enc_shape,
            dec_shape=   dec_shape,
            latent_dim=  latent_dim,
            dropout_rate=dropout_rate,
            activation=  activation,
            gm_prior=    gm_prior,
            verbose=     False
        )
        self.model.fitted = False

    def __repr__(self):
        return f'cyen DR model (fitted={self.model.fitted}, full_dim={self.model.full_dim}, latent_dim={self.model.latent_dim})'

    def reset(self):
        """
        Reset model weights

        Resets the parameters learned in fitting a model, so that it can be re-trained independently again.
        """

        full_dim     = self.model.full_dim
        enc_shape    = self.model.enc_shape
        dec_shape    = self.model.dec_shape
        latent_dim   = self.model.latent_dim
        dropout_rate = self.model.dropout_rate
        activation   = self.model.activation
        gm_prior     = self.model.gm_prior

        self.model = None
        self.model = ViVAE_network(
            full_dim=full_dim,
            enc_shape=enc_shape,
            dec_shape=dec_shape,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            gm_prior=gm_prior,
            verbose=False
        )
        self.model.fitted = False

    def fit(
        self,
        X:                 np.ndarray,
        knn:               Optional[list] = None,
        k:                 int = 100,
        triplet_sampling:  str = 'trimap',
        k_triplet:         int = 50,
        l_triplet:         Optional[int] = None,
        trimap_maxpoints:  Union[str,int] = 'auto',
        trimap_n_inliers:  int = 3,
        trimap_n_outliers: int = 3,
        trimap_n_random:   int = 1,
        loss:              Union[dict,str] = 'VAE-quartet',
        batch_size:        int = 1024,
        n_epochs:          int = 100,
        early_stopping:    bool = True,
        no_reset:          bool = False,
        monitor_quantity:  str = 'loss',
        min_delta:         float = 0.,
        patience:          int = 5,
        learning_rate:     float = 0.003,
        callback_csv:      bool = False,
        callback_tb:       bool = False,
        fpath_storage:     Optional[str] = None,
        seed:              Optional[int] = 1,
        verbose:           bool = True
    ):
        """
        Fit a ViVAE model

        Trains ViVAE model on input high-dimensional data.
        
        An important argument is `loss`: enter either a `dict` with weights or a string containing names of loss terms to include
        in the loss function. If a string is given (for example '`trimap-quartet`'), indicated loss terms will be added. Use the
        '`VAE'` keyword to use the `reconstruction` and `kldiv` loss terms (eg. '`VAE-quartet`'). All of the loss term names (keys
        of the `dict` if used) are: '`reconstruction`', '`kldiv`', '`triplet`', '`trimap`', '`tsne`', '`quartet`', '`quintet`'
        and '`sextet`'. Beware, computing the '`tsne`' loss uses CPU and slows down training.

        To train the model in multiple phases with different loss functions, call `fit` multiple times with the argument `update`
        set to True.

        Use arguments `callback_csv` and `callback_tb` to set up monitoring of loss term values throughout training.

        Use the `transform` method to produce a lower-dimensional embedding using the trained model.

        - X:                 row-wise coordinate matrix of high-dimensional input data (np.ndarray)
        - knn:               optional k-nearest-neighbour graph as generates with `cyen.make_knn` (list)
        - k:                 maximum nearest neighbour count to use, also if new k-nearest-neighbour graph needs to be constructed (int)
        - triplet_sampling:  for (optional) Siamese network training, use ivis-inspired '`simple`' or TriMap-inspired '`trimap`' sampling for generating and weighting triplets? (str)
        - k_triplet:         positive reference neighbour rank for 'simple' or 'trimap' triplet sampling (int)
        - l_triplet:         optional negative reference neighbour rank for 'simple' triplet sampling (int)
        - trimap_maxpoints:  maximum number of triplets generated with 'trimap' sampling. Can be 'auto' for dataset size times 3 (int/str)
        - trimap_n_inliers:  for 'trimap' sampling; number of sampled nearest neighbours per each anchor point to take as positive references
        - trimap_n_outliers: for 'trimap' sampling: number of sampled negative positive references for each positive reference in neighbourhood of anchor
        - trimap_n_random:   for 'trimap' sampling: number of randomly sampled triplets per anchor
        - loss:              name of model to use or dictionary of weights for each loss function term ('reconstruction', 'kldiv', 'triplet', 'trimap', 'quartet', 'quintet', 'sextet', 'tsne') (str/dict)
        - batch_size:        size of each mini-batch for training (int)
        - n_epochs:          number of training epochs (or maximum number if early stopping is enabled) (int)
        - early_stopping:    enable early stopping if value of evaluation metric (`monitor_quantity`) does not improve over some training epochs? (bool)
        - no_reset:          (experimental) if model has been trained already, continue training instead of refitting? (bool)
        - monitor_quantity:  quantity (evaluation metric) to monitor for early stopping ('`loss`' or any loss term name) (str)
        - min_delta:         minimal change in monitored quantity (float)
        - patience:          number of epochs without improvement which triggers early stopping (int)
        - learning_rate:     Adam optimiser learning rate parameter for training (float)
        - callback_csv:      write values of all monitored loss terms per training epoch to a CSV file? (bool)
        - callback_tb:       enable model visualisations through TensorBoard during training? (bool)
        - fpath_storage:     name of directory to store callback results or model weights when needed (str)
        - seed:              optional random seed for NumPy and TensorFlow for reproducibility (int),
        - verbose:           print progress messages? (bool)
        """
        
        self.model.verbose = verbose
        use_storage = False

        if self.model.fitted and not no_reset:
            self.reset()

        ## Set up loss function and loss term monitoring
        if isinstance(loss, str):
            l = dict()
            loss = loss.lower()
            for key in LOSS_TERMS:
                if loss.find(key) != -1:
                    if key in ['quartet', 'quintet', 'sextet']:
                        l[key] = 10.
                    elif key in ['trimap', 'triplet']:
                        l[key] = 10.
                    else:
                        l[key] = 1.
                else:
                    l[key] = 0.
            if loss.find('vae') != -1:
                l['reconstruction'] = 1.
                l['kldiv'] = 1.
            else:
                l['reconstruction'] = 0.
                l['kldiv'] = 0.
            loss = l
        elif isinstance(loss, dict):
            keys = list(loss.keys())
            for key in LOSS_TERMS:
                if key not in keys:
                    loss[key] = 0.
            for key in keys:
                if key not in LOSS_TERMS:
                    raise ValueError(f'Loss term `{key}` in `loss` not recognised')
        self.model.w = loss

        triplet_samp = self.model.w['triplet'] > 0. or self.model.w['trimap'] > 0.
        self.model.knn_required = triplet_samp

        if self.model.w['reconstruction'] > 0.:
            self.model.decoder = Decoder(full_dim=self.model.full_dim, shape=self.model.dec_shape, dropout=self.model.dropout_rate, activation=self.model.activation)
        
        self.model.batch_size = batch_size

        ## Resolve k-NNG
        if self.model.knn_required:
            self.model.attach_knn(knn=knn, x=X, k=k)

        ## Resolve callbacks
        callbacks = []
        if callback_tb:
            fpath_logs = fpath_storage+'/logs/scalars/'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=fpath_logs))
            use_storage = True
        if callback_csv:
            fname_csv = fpath_storage+'/losses.csv'
            if not os.path.exists(fpath_storage):
                os.makedirs(fpath_storage)
            if os.path.exists(fname_csv):
                f = open(fname_csv, 'a')
                f.close()
            else:
                colnames = LOSS_TERMS
                colnames.append('loss')
                colnames = sorted(colnames)
                colnames.insert(0, 'epoch')
                df = pd.DataFrame(columns=colnames)
                df.to_csv(fname_csv, index=False)
            callbacks.append(tf.keras.callbacks.CSVLogger(filename=fname_csv, separator=',', append=True))
            use_storage = True
        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor_quantity, min_delta=min_delta, patience=patience, verbose=1))
        
        ## Resolve storage directory
        if use_storage:
            if fpath_storage is None:
                raise ValueError('`fpath_storage` must be specified for TensorBoard callback, CSV logging or saving model weights')
            if not os.path.exists(fpath_storage):
                os.makedirs(fpath_storage)

        ## Compile model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        ## Attach data
        self.model.attach_data(X)

        ## Generate triplets if needed
        if triplet_samp:
            if k_triplet>self.model.knn_idcs.shape[1]-1:
                k_triplet = self.model.knn_idcs.shape[1]-1
            if k_triplet is None:
                k_triplet = self.model.knn_idcs.shape[1]-1
                l_triplet = None
            self.model.triplet_sampling = triplet_sampling
            if triplet_sampling == 'simple':
                if verbose:
                    print('Triplet sampling using "simple" method enabled')
                input = self.model.get_triplet_data(method='simple', k=k_triplet, l=l_triplet, n_inliers=trimap_n_inliers, n_outliers=trimap_n_outliers, n_random=trimap_n_random, seed=seed)
            elif triplet_sampling == 'trimap':
                if verbose:
                    print('Triplet sampling using "trimap" method enabled')
                input = self.model.get_triplet_data(method='trimap', max_points=trimap_maxpoints, n_inliers=trimap_n_inliers, n_outliers=trimap_n_outliers, n_random=trimap_n_random, seed=seed)
        else:
            if verbose:
                print('Triplet sampling disabled')
            input = [X, np.repeat(0., repeats=X.shape[0]), np.repeat(0., repeats=X.shape[0]), np.repeat(1., repeats=X.shape[0])]

        ## Train model on input data
        if seed is not None:
            tf.random.set_seed(seed)
        fit_args = {
            'x':                input,
            'y':                None,
            'batch_size':       batch_size,
            'epochs':           n_epochs,
            'shuffle':          True,
            'callbacks':        callbacks,
            'validation_split': .0,
            'verbose':          1 if self.model.verbose else 0
        }
        self.model.fit(**fit_args)
        self.model.fitted = True

        pass

    def transform(self, X: np.ndarray):
        """
        Transform data using cyen model

        Using a trained cyen model, generate a lower-dimensional embedding of a high-dimensional dataset.

        - X: high-dimensional data coordinate matrix (np.ndarray)
        """

        if not self.model.fitted:
            raise AttributeError('Model not trained. Call `fit` before transforming data')

        inputs = layers.Input(shape=X.shape[1], )
        outputs = self.model.encoder(inputs)[0]
        enc = tf.keras.models.Model(inputs, outputs)
        proj = enc.predict(X, batch_size=self.model.batch_size)

        return proj

    def fit_transform(
        self,
        X:                 np.ndarray,
        knn:               Optional[list] = None,
        k:                 int = 100,
        triplet_sampling:  str = 'trimap',
        k_triplet:         int = 5,
        l_triplet:         int = 10,
        trimap_maxpoints:  Union[str,int] = 'auto',
        trimap_n_inliers:  int = 3,
        trimap_n_outliers: int = 3,
        trimap_n_random:   int = 1,
        loss:              Union[dict,str] = 'VAE-quartet',
        batch_size:        int = 1024,
        n_epochs:          int = 100,
        early_stopping:    bool = True,
        monitor_quantity:  str = 'loss',
        min_delta:         float = 0.,
        patience:          int = 5,
        learning_rate:     float = 0.001,
        callback_csv:      bool = False,
        callback_tb:       bool = False,
        fpath_storage:     Optional[str] = None,
        seed:              Optional[int] = 1,
        verbose:           bool = True
    ):
        """
        Fit a ViVAE model and transform data

        Trains ViVAE model on input high-dimensional data and generate a lower-dimensional embedding using the model.
        
        An important argument is `loss`: enter either a `dict` with weights or a string containing names of loss terms to include
        in the loss function. If a string is given (for example '`trimap-quartet`'), indicated loss terms will be added. Use the
        '`VAE'` keyword to use the `reconstruction` and `kldiv` loss terms (eg. '`VAE-quartet`'). All of the loss term names (keys
        of the `dict` if used) are: '`reconstruction`', '`kldiv`', '`triplet`', '`trimap`', '`tsne`', '`quartet`', '`quintet`'
        and '`sextet`'. Beware, computing the '`tsne`' loss uses CPU and slows down training.

        Use arguments `callback_csv` and `callback_tb` to set up monitoring of loss term values throughout training.

        - X:                row-wise coordinate matrix of high-dimensional input data (np.ndarray)
        - knn:              optional k-nearest-neighbour graph as generates with `cyen.make_knn` (list)
        - k:                maximum nearest neighbour count to use, also if new k-nearest-neighbour graph needs to be constructed (int)
        - triplet_sampling: for (optional) Siamese network training, use ivis-inspired '`simple`' or TriMap-inspired '`trimap`' sampling for generating and weighting triplets? (str)
        - k_triplet:        positive reference neighbour rank for 'simple' triplet sampling (int)
        - l_triplet:        optional negative reference neighbour rank for 'simple' triplet sampling (int)
        - trimap_maxpoints: maximum number of triplets generated with 'trimap' sampling. Can be 'auto' for dataset size times 3 (int/str)
        - trimap_n_inliers:  for 'trimap' sampling; number of sampled nearest neighbours per each anchor point to take as positive references
        - trimap_n_outliers: for 'trimap' sampling: number of sampled negative positive references for each positive reference in neighbourhood of anchor
        - trimap_n_random:   for 'trimap' sampling: number of randomly sampled triplets per anchor
        - loss:             name of model to use or dictionary of weights for each loss function term ('reconstruction', 'kldiv', 'triplet', 'trimap', 'quartet', 'quintet', 'sextet', 'tsne') (str/dict)
        - batch_size:       size of each mini-batch for training (int)
        - n_epochs:         number of training epochs (or maximum number if early stopping is enabled) (int)
        - early_stopping:   enable early stopping if value of evaluation metric (`monitor_quantity`) does not improve over some training epochs? (bool)
        - monitor_quantity: quantity (evaluation metric) to monitor for early stopping ('`loss`' or any loss term name) (str)
        - min_delta:        minimal change in monitored quantity (float)
        - patience:         number of epochs without improvement which triggers early stopping (int)
        - learning_rate:    Adam optimiser learning rate parameter for training (float)
        - callback_csv:     write values of all monitored loss terms per training epoch to a CSV file? (bool)
        - callback_tb:      enable model visualisations through TensorBoard during training? (bool)
        - fpath_storage:    name of directory to store callback results or model weights when needed (str)
        - seed:             optional random seed for NumPy and TensorFlow for reproducibility (int),
        - verbose:          print progress messages? (bool)
        """
        self.fit(
            X=X, knn=knn, k=k, triplet_sampling=triplet_sampling, k_triplet=k_triplet, l_triplet=l_triplet,
            trimap_maxpoints=trimap_maxpoints, trimap_n_inliers=trimap_n_inliers, trimap_n_outliers=trimap_n_outliers,
            trimap_n_random=trimap_n_random, loss=loss, batch_size=batch_size, n_epochs=n_epochs,
            early_stopping=early_stopping, monitor_quantity=monitor_quantity, min_delta=min_delta,
            patience=patience, learning_rate=learning_rate, callback_csv=callback_csv, callback_tb=callback_tb,
            fpath_storage=fpath_storage, seed=seed, verbose=verbose
        )
        return self.transform(X=X)      

    def get_knn(self):
        """
        Extract k-nearest-neighbour graph if it was constructed

        If the k-NNG is attached to the cyen model fitted to a dataset, this function returns it. If the k-NNG has not been
        constructed, use the function `cyen.make_knn` instead to construct it.
        """
        
        if self.model.knn_idcs is None or self.model.knn_dist is None:
            raise AttributeError('k-nearest neighbour graph not available, use `cyen.make_knn` instead')
        
        return [self.model.knn_idcs, self.model.knn_dist]
