# LSsurf
Utilities for performing smooth least-squares fits in Python

These scripts include utilities for fitting smooth surfaces to data in a least-squares sense.  They operate on the principle of building a design matrix while keeping track of the relationship between degrees of freedom and columns and between data, constraint parameters, and rows.

The main functions of the repository are used in LSsurf/smooth_fit.py, which fits smooth gridded surfaces to pointwise input data.  See the xytb_fit_demo.ipynb notebook in the notebooks directory for a demonstration.  The functionality is built around my pointCollection library (https://www.github.com/smithb/pointCollection.git) which provides data structures for manipulating point data.


# Setup

To run the demo notebook, you'll need (do these in order):

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
