from __future__ import division
cimport cython
import numpy as np
cimport numpy as np
ITYPE=np.int32
ctypedef np.int32_t ITYPE_t
FTYPE=np.float
ctypedef np.float_t FTYPE_t
cdef extern from "math.h":
    double sqrt(double x)

@cython.boundscheck(False)

#def propagate_qz_errors(np.ndarray[ITYPE_t, ndim=1] indptr, np.ndarray[ITYPE_t, ndim=1] indices, np.ndarray[FTYPE_t, ndim=1] data, np.ndarray[FTYPE_t, ndim=1] E):
def propagate_qz_errors(R):
    """
    Solves the equation R Rinv = I for Rinv, calculates the row-wise RSS of Rinv
    ----------
    R : (M, M) sparse matrix
        A sparse square upper triangular matrix. Should be in CSR format.

    Returns
    -------
    E : (M,) 
        The RSS of Rinv, equal to the per-element error of R.

    .. Derived from the spsolve_triangular function in scipy version:: 0.19.0
    """
    cdef Py_ssize_t i, indptr_start, indptr_stop, col, row
    cdef Py_ssize_t  A_off_diagonal_index_row_i, A_column_index_in_row_i
    cdef FTYPE_t this_x
    # pull apart R and explicitly type the pieces
    cdef np.ndarray[ITYPE_t, ndim=1] indptr=R.indptr    
    cdef np.ndarray[ITYPE_t, ndim=1] indices=R.indices
    cdef np.ndarray[FTYPE_t, ndim=1] data=R.data
    # Result matrix
    cdef np.ndarray[FTYPE_t, ndim=1] E=np.zeros(R.shape[0])
    # work matrix
    cdef np.ndarray[FTYPE_t, ndim=1] x=np.zeros(R.shape[0]) 
    # loop over the columns of I 
    for col in range(len(E)-1, -1, -1):
        # make the current column of I, which will be turned into a column of Rinv
        for row in range(len(E)):
            x[row]=0
        x[col]=1
        # Fill x iteratively.  All elements of Rinv[:,col] are zero up to 'col'
        for i in range(col, -1, -1):
            # Get indices for i-th row.
            indptr_start = indptr[i]
            indptr_stop = indptr[i+1]
 
            this_x=x[i]
            # Incorporate off-diagonal entries.
            for A_off_diagonal_index_row_i in range(indptr_start+1, indptr_stop):
                A_column_index_in_row_i = indices[A_off_diagonal_index_row_i]
                if A_column_index_in_row_i > col:
                    # we know that the entries of x for j > col are all zero,
                    # so once we find one index that is > 'col', we quit 
                    break
                this_x -= data[A_off_diagonal_index_row_i]*x[A_column_index_in_row_i]
                # Apply the diagonal entry
            this_x /= data[indptr_start]
            x[i]=this_x
            # add the square of the current value of x to E
            E[i] += this_x*this_x
    # not sure why np.sqrt[E] doesn't work here, but this sure dones
    for i in range(len(E)):
        E[i]=sqrt(E[i])
    return E