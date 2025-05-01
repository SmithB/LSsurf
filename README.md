# LSsurf
Utilities for performing smooth least-squares fits in Python

These scripts include utilities for fitting smooth surfaces to data in a least-squares sense.  They operate on the principle of building a design matrix while keeping track of the relationship between degrees of freedom and columns and between data, constraint parameters, and rows.  The functionality is built around my pointCollection library (https://www.github.com/smithb/pointCollection.git) which provides data structures for manipulating point data.

## Contents

The _LSsurf_ module provides two main classes:

* _fd_grid_ :
  A class that defines a finite-difference grid, up to an arbitrary number of dimensions.  Includes relationships between node indices and a global index that spans multiple grids for a multi-resolution least-squares problem
* _lin\_op_ :
  A class that defines operators that multiply the values in a _fd__grid_.  Examples include linear interpolations into the grid (2-D and 3-D), averages of grid points, and finite-difference derivatives of grid fields.

  ** _fd\_op : A subclass of _lin\_op_ that constructs _lin\_op_ objects that perform template operations on a neighborhood of points.  Pre-defined subclasses perform derivatives of grid fields.

The main functions of the repository are used in LSsurf/smooth_fit.py, which fits smooth gridded surfaces to pointwise input data.  See the smooth_fit_demo.ipynb notebook in the notebooks directory for a demonstration.
  
# Setup

To use the functions in this repo, or to run the demo notebook, you'll need (do these in order):

The suitesparse library:

From conda:

> conda install suitesparse

On ubuntu/similar linux

> apt-get install suitesparse

@Yig's PySPQR repository (This is where you need suitesparse)

https://github.com/yig/PySPQR.git

My pointCollection repository:

https://www.github.com/smithb/pointCollection.git

For each repository, you'll need to clone the repo (git clone [url to .git file]), then cd to the 
directory that git makes, and type:

> python3 setup.py install --user 

Good luck!
