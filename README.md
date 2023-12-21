# ViVAE

ViVAE is a toolkit for single-cell data denoising and dimensionality reduction.

**It is published together with [ViScore](https://github.com/saeyslab/ViScore), a collection of tools for evaluation of dimensionality reduction.**

## Installation

ViVAE is a Python package built on top of TensorFlow.
We recommend creating a new Anaconda environment for ViVAE.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

*(A test install run on 2020 MacBook Air runs below 4 minutes.)*

```
conda create --name ViVAE python=3.9 \
    numpy==1.22.4 numba==0.58.1 pandas==2.1.4 matplotlib==3.8.2 scipy==1.11.4 pynndescent==0.5.11 scikit-learn==1.3.2 scanpy==1.9.6
```

Next, activate the new environment and install `tensorflow` and `tensorflow_probability`.
TensorFlow installation is platform-specific.
GPU acceleration, when available, is highly recommended.

### macOS (AMD or Apple Silicon GPUs)

```
conda activate ViVAE
pip install tensorflow-macos==2.9.0
pip install tensorflow-metal==0.5.0
conda install -c apple tensorflow-deps==2.9.0
pip install tensorflow_probability==0.17.0
pip install --upgrade git+https://github.com/saeyslab/ViVAE.git
```

(The macOS install now uses an older version of TensorFlow due to compatibility issues otherwise.)

Consult [this tutorial](https://developer.apple.com/metal/tensorflow-plugin/) in case of TensorFlow installation problems.

### Windows (CUDA)

```
conda activate ViVAE
conda install conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow==2.15.0"
pip install "tensorflow_probability==0.23.0"
pip install --upgrade git+https://github.com/saeyslab/ViVAE.git
```

Consult [this tutorial](https://www.tensorflow.org/install/pip#windows-native) in case of TensorFlow installation problems.

### Linux (CUDA)

```
conda activate ViVAE
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.15.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install tensorflow_probability==0.23.0
pip install --upgrade git+https://github.com/saeyslab/ViVAE.git
```

Consult [this tutorial](https://www.tensorflow.org/install/pip#linux) in case of TensorFlow installation problems.

### CPU

```
conda activate ViVAE
pip install tensorflow==2.15.0
pip install tensorflow_probability==0.23.0
pip install --upgrade git+https://github.com/saeyslab/ViVAE.git
```

Consult [this tutorial](https://www.tensorflow.org/install/pip#cpu) in case of TensorFlow installation problems.

### GPU verification

To verify whether TensorFlow can use GPU acceleraction, activate the environment and run the following line:

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
knn = ViVAE.make_knn(X, fname='knn.npy')
                               # build k-NNG if not available already (first run may take longer with pynndescent)
Xs = ViVAE.smooth(X, knn=knn, coef=1.0, n_iter=1)
                               # denoise input to VAE
proj = ViVAE.ViVAE(full_dim=Xs.shape[1]).fit_transform(X=Xs)
                               # train model and create embedding
ViVAE.plot(proj=proj, annot=annot) # plot embedding
```

## Example with data

The `example/example.ipynb` Jupyer notebook shows usage of ViVAE and ViScore on a provided dataset, including the scoring methodology used in our [paper](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v2).

Use [Git LFS](https://git-lfs.com) to download the attached datasets.

## Pre-print

The pre-print of our publication is available [here](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v2) on bioRxiv.

It describes underlying methodology of ViVAE and ViScore, reviews past work in dimensionality reduction and evaluation of it and links to publicly available datasets on which performance of ViVAE was evaluated.
