from setuptools import setup, Extension, find_packages
import subprocess
import logging
import sys
import os

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger()

# check for numpy installation
try:
    import numpy
except ImportError:
    raise ImportError('NumPy is required to install LSsurf')

# get long_description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

# get install requirements
with open('requirements.txt') as fh:
    install_requires = fh.read().splitlines()

# run cmd from the command line
def check_output(cmd):
    return subprocess.check_output(cmd).decode('utf')

# check if GDAL is installed
gdal_output = [None] * 4
try:
    for i, flag in enumerate(("--cflags", "--libs", "--datadir", "--version")):
        gdal_output[i] = check_output(['gdal-config', flag]).strip()
except:
    log.warning('Failed to get options via gdal-config')
else:
    log.info("GDAL version from via gdal-config: {0}".format(gdal_output[3]))
# if setting GDAL version from via gdal-config
if gdal_output[3]:
    # add version information to gdal in install_requires
    gdal_index = install_requires.index('gdal')
    install_requires[gdal_index] = 'gdal=={0}'.format(gdal_output[3])

# append conda include path
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    include_dirs=[numpy.get_include(), os.path.join(conda_prefix, 'include')]
else:
    include_dirs=[numpy.get_include()]

# Setuptools 18.0 properly handles Cython extensions.
setup_requires=[
    'setuptools>=18.0',
    'cython',
]
# cythonize extensions
ext_modules=[
    Extension('LSsurf.inv_tr_upper', sources=['LSsurf/inv_tr_upper.pyx']),
    Extension('LSsurf.propagate_qz_errors', sources=['LSsurf/propagate_qz_errors.pyx']),
    Extension('LSsurf.spsolve_tr_upper', sources=['LSsurf/spsolve_tr_upper.pyx'])
]

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
    include_dirs=include_dirs,
    setup_requires=setup_requires,
    ext_modules=ext_modules,
    install_requires=install_requires,
    zip_safe=False
)
