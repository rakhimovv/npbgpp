#!/usr/bin/env python

from setuptools import setup, find_packages

__version__ = '0.0.1'
install_requires = [
    'hydra-core',
    'pytorch-lightning',
    'matplotlib',
    'opencv-python',
    'torch',
    'torchvision',
    'test_tube',
    'tensorboard',
    'scipy',
    'tabulate',
    'trimesh',
    'torch_scatter',
    'pytorch3d',
    'lpips',
    'timm',
    'plotly'
]

setup(name='npbgplusplus',
      version=__version__,
      description='npbgplusplus',
      author='npbgpp',
      author_email='',
      install_requires=install_requires,
      packages=find_packages()
      )
