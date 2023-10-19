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

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
from tensorflow_probability import distributions

eps_std = tf.constant(1e-2, dtype=tf.float64)
eps_sq = eps_std ** 2
K.set_floatx('float64')

class Encoder(layers.Layer):
  """
  Encoder class

  Encodes high-dimensional data into a latent distribution from which a lower-dimensional representation
  can be generated.
  """

  def __init__(
    self,
    dropout:    float = 0.05,
    shape:      list = [32,64,128,32],
    latent_dim: int = 2,
    activation: str = 'selu',
    name:       str = 'Encoder',
    **kwargs
  ):
    """
    Instantiate encoder

    - dropout:    rate of dropout for regularisation (float)
    - shape:      list of consecutive node counts defining the size of each layer of the encoder (list of ints)
    - latent_dim: dimensionality of the latent representation (int)
    - activation: activation function in each node of the network: eg. 'selu', 'relu', 'sigmoid' (str)
    - name:       name of the encoder network (str)
    """
    super(Encoder, self).__init__(name=name, **kwargs)
    self.shape = shape
    self.drop = layers.Dropout(rate=dropout)
    self.layers = [None] * len(self.shape)
    for idx, n in enumerate(self.shape):
      self.layers[idx] = layers.Dense(n, activation=activation)
    self.mu = layers.Dense(latent_dim)
    self.sigma = layers.Dense(latent_dim)
    
  def call(self, x, training=None):
    for l in self.layers:
      x = self.drop(l(x))
    return self.mu(x), self.sigma(x)

class Sampler(layers.Layer):
    """
    Sampler class

    Samples a compressed latent representation from a lower-dimensional distribution learned by the encoder.
    KL-divergence from a latent prior is minimised in training. This latent prior can be an isotropic Gaussian
    or a Gaussian mixture.
    """

    def __init__(
      self,
      gm_prior: bool = True,
      name:     str = 'Sampler',
      **kwargs
    ):
      """
      Instantiate sampler

      - gm_prior: use Gaussian mixture instead of isotropic Gaussian? (bool)
      - name:     name of sampler (str)
      """

      super(Sampler, self).__init__(name=name, **kwargs)
      self.gm_prior = gm_prior

    def call(self, z_mu, z_sigma, training=None):
      if self.gm_prior:
        return distributions.MultivariateNormalDiag(loc=z_mu, scale_diag=z_sigma).sample()
      else:
        eps = tf.keras.backend.random_normal(shape=tf.shape(z_mu))
        return z_mu + eps_std * tf.exp(0.5*z_sigma) * eps
        

class Decoder(layers.Layer):
    """
    Decoder class

    Decodes a compressed representation of data from a lower-dimensional distribution learned by an encoder
    to reconstruct the high-dimensional data.
    """

    def __init__(
      self,
      full_dim:   int,
      dropout:    float = 0.05,
      shape:      list = [32,128,64,32],
      activation: str = 'selu',
      name:       str = 'Decoder',
      **kwargs
    ):
        """
        Instantiate decoder

        - full_dim:   dimensionality of the original high-dimensional data (int)
        - dropout:    rate of dropout for regularisation (float)
        - shape:      list of consecutive node counts defining the size of each layer of the decoder (list of ints)
        - latent_dim: dimensionality of the latent representation (int)
        - activation: non-linear activation function in each node of the network: eg. 'selu', 'relu', 'sigmoid' (str)
        - name:       name of the encoder network (str)
        """
        super(Decoder, self).__init__(name=name, **kwargs)
        self.shape = shape
        self.drop = layers.Dropout(rate=dropout)
        self.layers = [None] * len(self.shape)
        for idx, n in enumerate(self.shape):
            self.layers[idx] = layers.Dense(n, activation=activation)
        self.recon = layers.Dense(full_dim)
        
    def call(self, x, training=None):
        for l in self.layers:
            x = self.drop(l(x))
        return self.recon(x)
