# ViVAE

ViVAE is a toolkit for single-cell data denoising and dimensionality reduction.

**It is published together with [ViScore](https://github.com/saeyslab/ViScore), a collection of tools for evaluation of dimensionality reduction.**

## Installation

ViVAE is a Python package built on top of TensorFlow.
We recommend creating a new Anaconda environment for ViVAE.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

```
conda create --name ViVAE python=3.9 \
    numpy numba pandas matplotlib scipy pynndescent scikit-learn scanpy
```

Next, activate the new environment and install `tensorflow` and `tensorflow_probability`.
TensorFlow installation is platform-specific.
GPU acceleration, when available, is highly recommended.

### macOS (Metal)

```
conda activate ViVAE
pip install tensorflow==2.13.0
pip install tensorflow-macos
pip install tensorflow-metal
pip install tensorflow_probability
```

Consult [this tutorial](https://developer.apple.com/metal/tensorflow-plugin/) in case of problems.

### Windows (CUDA)

```
conda install conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"
pip install tensorflow_probability
```

Consult [this tutorial](https://www.tensorflow.org/install/pip#windows-native) in case of problems.

### Linux (CUDA)

```
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install tensorflow_probability

```

Consult [this tutorial](https://www.tensorflow.org/install/pip#linux) in case of problems.

### CPU

```
pip install tensorflow
pip install tensorflow_probability
```

Consult [this tutorial](https://www.tensorflow.org/install/pip#cpu) in case of problems.

### Verification

To verify correct installation of TensorFlow, activate the environment and run the following line:

```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

This should return a non-empty list.

## Usage

Recommended workflow uses pre-smoothing (denoising) of input data and training a SQuadVAE (Stochastic Quartet loss-regularised Variational AutoEncoder) model.

```
# `X` is np.ndarray of high-dimensional data
# `annot` is np.ndarray of labels per row of `X`

import ViVAE
knn = ViVAE.make_knn(X, fname='knn.npy'
                               # build k-NNG if not available already (first run may take longer with pynndescent)
Xs = ViVAE.smooth(X, knn=knn, coef=1.0, n_iter = 1)
                               # denoise input to VAE
proj = ViVAE.ViVAE(full_dim=Xs.shape[1]).fit_transform(X=Xs)
                               # train model and create embedding
ViVAE.plot(proj=proj, annot=annot) # plot embedding
```
