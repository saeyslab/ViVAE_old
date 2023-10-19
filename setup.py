
from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Single-cell data denoising & dimension-reduction toolbox'
LONG_DESCRIPTION = 'Autoencoder-based models for dimension reduction of high-dimensional cytometry and single-cell RNA-seq data based on geometry and nearest-neighbour relations'

setup(
        name='ViVAE', 
        version=VERSION,
        author="David Novak",
        author_email="<davidnovakcz@hotmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'scipy', 'numba', 'tensorflow'],
        keywords=['python'],
        classifiers= [
            "Development Status :: 4 - Beta",
            "Intended Audience :: Bioinformatics",
            "Programming Language :: Python :: 3"
        ]
)
