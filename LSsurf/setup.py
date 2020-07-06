from setuptools import setup
from Cython.Build import cythonize
import numpy
setup(
    name="spsolve_tr_upper",
    ext_modules=cythonize("spsolve_tr_upper.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
