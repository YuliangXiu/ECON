from setuptools import setup
from Cython.Build import cythonize

setup(name='libvoxelize', ext_modules=cythonize("*.pyx"))
