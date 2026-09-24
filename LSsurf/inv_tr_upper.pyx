# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
cimport numpy as np
ITYPE=np.int32
ctypedef np.int32_t ITYPE_t
FTYPE=np.float64
ctypedef np.float64_t FTYPE_t
from libc.math cimport fabs

# columns per block.  Fixed (not tied to the thread count) so that the blocks,
# and therefore the output, are the same for any number of threads.
BLOCK_COLS = 256


cdef Py_ssize_t _reach_block(const ITYPE_t[:] indptr, const ITYPE_t[:] indices,
                             const FTYPE_t[:] data, Py_ssize_t col_hi, Py_ssize_t col_lo,
                             FTYPE_t[:] x, np.int8_t[:] mark,
                             ITYPE_t[:] out_rows, ITYPE_t[:] out_cols, FTYPE_t[:] out_vals,
                             double tol) noexcept nogil:
    """
    Solve R x = e_col for columns col_hi-1 down to col_lo, writing the entries of
    each x that are on the diagonal or larger than tol in magnitude.

    R is upper triangular in CSC form with sorted indices, so the diagonal is
    the last entry of each column.  Once x[j] is final (every column > j has
    been applied), it is scattered up column j of R.  Only rows that have
    received a contribution (marked) are visited; the rest of the column is
    zero, so it costs one byte test per row.

    x and mark must be all zero on entry, and are left all zero.  tol is a
    double here because converting a float inside the inner loop costs ~14%;
    callers pass the float32-rounded value, so the threshold is unchanged.
    Returns the number of entries written, or -1 if out_rows filled up.
    """
    cdef Py_ssize_t col, j, k, i, lo, out_ind = -1, max_ind = out_rows.shape[0] - 1
    cdef FTYPE_t xj
    for col in range(col_hi - 1, col_lo - 1, -1):
        x[col] = 1.0
        mark[col] = 1
        lo = col
        j = col
        while j >= lo:
            if mark[j]:
                mark[j] = 0
                xj = x[j] / data[indptr[j + 1] - 1]
                x[j] = 0.0
                for k in range(indptr[j], indptr[j + 1] - 1):
                    i = indices[k]
                    x[i] -= data[k] * xj
                    if not mark[i]:
                        mark[i] = 1
                        if i < lo:
                            lo = i
                if j == col or fabs(xj) > tol:
                    out_ind += 1
                    if out_ind > max_ind:
                        return -1
                    out_rows[out_ind] = j
                    out_cols[out_ind] = col
                    out_vals[out_ind] = xj
            j -= 1
    return out_ind + 1


def _solve_block(Rc, Py_ssize_t col_hi, Py_ssize_t col_lo, Py_ssize_t nmax, float tol, work):
    cdef const ITYPE_t[:] indptr = Rc.indptr
    cdef const ITYPE_t[:] indices = Rc.indices
    cdef const FTYPE_t[:] data = Rc.data
    cdef FTYPE_t[:] x = work[0]
    cdef np.int8_t[:] mark = work[1]
    rr = np.empty(nmax, dtype=ITYPE)
    cc = np.empty(nmax, dtype=ITYPE)
    vv = np.empty(nmax, dtype=FTYPE)
    cdef ITYPE_t[:] rrv = rr
    cdef ITYPE_t[:] ccv = cc
    cdef FTYPE_t[:] vvv = vv
    cdef Py_ssize_t n
    with nogil:
        n = _reach_block(indptr, indices, data, col_hi, col_lo, x, mark, rrv, ccv, vvv, tol)
    if n < 0:
        # the block alone holds more than nmax entries; the caller reports status=1
        return None
    return rr[:n], cc[:n], vv[:n]


def inv_tr_upper(R, Py_ssize_t nnz, float tol, int threads=1):
    """
    Solves the equation R Rinv = I for Rinv, keeping the entries of Rinv that
    are on the diagonal or larger than tol in magnitude.
    ----------
    R : (M, M) sparse matrix
        A sparse square upper triangular matrix with a nonzero diagonal.
    nnz : int
        The largest number of entries to return.
    tol : float
        Off-diagonal entries of Rinv with |value| <= tol are not returned.
        They still take part in the solve, so the returned entries are exact.
    threads : int
        Number of threads to use.  The output does not depend on it.

    Returns
    -------
    rows, cols, vals : the entries of Rinv, by column from the last to the
        first, and by row from the diagonal up within each column.
    status : 0 on success, 1 if Rinv has more than nnz entries to return (the
        other outputs are then incomplete, and the caller should retry with a
        larger nnz).
    """
    cdef Py_ssize_t N = R.shape[0]
    Rc = R.tocsc(copy=True)
    Rc.sort_indices()
    if Rc.indices.dtype != ITYPE or Rc.indptr.dtype != ITYPE:
        Rc.indices = Rc.indices.astype(ITYPE)
        Rc.indptr = Rc.indptr.astype(ITYPE)
    if Rc.data.dtype != FTYPE:
        Rc.data = Rc.data.astype(FTYPE)

    out_rows = np.empty(nnz, dtype=ITYPE)
    out_cols = np.empty(nnz, dtype=ITYPE)
    out_vals = np.empty(nnz, dtype=FTYPE)

    # blocks of columns, highest columns first, matching the output order
    blocks = [(hi, max(hi - BLOCK_COLS, 0)) for hi in range(N, 0, -BLOCK_COLS)]
    local = threading.local()

    def run(block):
        col_hi, col_lo = block
        if not hasattr(local, 'work'):
            local.work = (np.zeros(N, dtype=FTYPE), np.zeros(N, dtype=np.int8))
        # a block can hold at most (col_hi - col_lo) * col_hi entries
        nmax = min((col_hi - col_lo) * col_hi, nnz) + 1
        return _solve_block(Rc, col_hi, col_lo, nmax, tol, local.work)

    offset = 0
    status = 0

    def take(result):
        # copy one block's entries into the output; False on overflow
        nonlocal offset
        if result is None or offset + result[0].size > nnz:
            return False
        n = result[0].size
        out_rows[offset:offset + n] = result[0]
        out_cols[offset:offset + n] = result[1]
        out_vals[offset:offset + n] = result[2]
        offset += n
        return True

    if threads <= 1:
        for block in blocks:
            if not take(run(block)):
                status = 1
                break
    else:
        # keep at most 2*threads blocks in flight, and consume them in order,
        # so that finished blocks do not pile up in memory
        with ThreadPoolExecutor(threads) as ex:
            pending = deque()
            todo = iter(blocks)
            for block in todo:
                pending.append(ex.submit(run, block))
                if len(pending) >= 2 * threads:
                    break
            while pending:
                if not take(pending.popleft().result()):
                    status = 1
                    for f in pending:
                        f.cancel()
                    break
                for block in todo:
                    pending.append(ex.submit(run, block))
                    break

    return out_rows[0:offset], out_cols[0:offset], out_vals[0:offset], status
