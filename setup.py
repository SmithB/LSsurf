from setuptools import setup
from Cython.Build import cythonize
import numpy

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='LSsurf', 
    version='1.0.0.0',
    description='Utilities for fitting smooth surfaces to point data', 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/SmithB/LSsurf',
    author='Ben Smith',
    author_email='besmith@uw.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),

    ext_modules=cythonize("LSsurf/*.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
