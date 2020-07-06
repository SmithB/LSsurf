from __future__ import division
import numpy as np
cimport numpy as np
ITYPE=np.int32
ctypedef np.int32_t ITYPE_t
FTYPE=np.float
ctypedef np.float_t FTYPE_t
cimport cython
@cython.boundscheck(False)

def spsolve_tr_upper(A, np.ndarray[FTYPE_t, ndim=1] b):
    """
    Solve the equation `A x = b` for `x`, assuming A is a triangular matrix.
    Parameters
    ----------
    A : (M, M) sparse matrix
        A sparse square upper triangular matrix. Should be in CSR format.
    b : (M,) 
        Right-hand side matrix in `A x = b`

    Returns
    -------
    x : (M,) 
        Solution to the system `A x = b`.  Shape of return matches shape of `b`.

    .. copied from scipy version:: 0.19.0
    """
    # pull apart R and explicitly type the pieces
    cdef np.ndarray[ITYPE_t, ndim=1] indptr=A.indptr    
    cdef np.ndarray[ITYPE_t, ndim=1] indices=A.indices
    cdef np.ndarray[FTYPE_t, ndim=1] data=A.data
    # Result matrix
    cdef np.ndarray[FTYPE_t, ndim=1] x=b.copy()
    # temporary matrices and indexes
    cdef Py_ssize_t i, indptr_start, indptr_stop  
    cdef Py_ssize_t  A_off_diagonal_index_row_i, A_column_index_in_row_i
    cdef FTYPE_t this_x
 
    # Fill x iteratively.
    for i in range(len(b)-1, -1, -1):
        # Get indices for i-th row.
        indptr_start = indptr[i]
        indptr_stop = indptr[i+1]
 
        A_diagonal_index_row_i = indptr_start
        this_x=x[i]
        # Incorporate off-diagonal entries.
        for A_off_diagonal_index_row_i in range(indptr_start+1, indptr_stop):
            A_column_index_in_row_i = indices[A_off_diagonal_index_row_i]
            this_x -= data[A_off_diagonal_index_row_i]*x[A_column_index_in_row_i]
        # Apply the diagonal entry
        this_x /= data[indptr_start]
        x[i]=this_x
    return x