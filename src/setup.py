import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension("c_optimality_intervals", sources=["optimality_intervals.pyx"])
]
setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
