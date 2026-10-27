"""
Row sums of squares of Rinv = R^-1, and of op Rinv for a set of operators,
without storing Rinv.

smooth_fit's error propagation needs only these sums: the error of unknown i
is sqrt(sum_j Rinv[i, j]**2), and the error of row k of an operator is
sqrt(sum_j (op Rinv)[k, j]**2).  inv_tr_upper's kernel computes Rinv one block
of columns at a time, so each block's contribution is added to the sums and
the block is dropped.  Storing Rinv instead took ~16 bytes per entry for the
triplets plus ~12 for each sparse copy: on the AA tile E-560_N-1240 (a
126,600-column R, 558.7M entries in Rinv) the error step peaked at 23 GB
resident and 75 GB virtual (60 GB of it inv_tr_upper's preallocated output).
Streamed, it peaks at 10.9 GB resident and 11.9 GB virtual, set by the 10.4 GB
of the QR factorization (sparseqr.rz) that comes before it, and runs in 314 s
instead of 373 s.

The sums use the entries inv_tr_upper keeps (the diagonal, and off-diagonal
entries larger than tol in magnitude), and the blocks are added in column
order, so the output does not depend on the number of threads.
"""
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

from LSsurf.inv_tr_upper import _solve_block, BLOCK_COLS, ITYPE, FTYPE


def inv_tr_upper_sumsq(R, tol, perm=None, ops=None, threads=1):
    """
    Row sums of squares of Rinv and of each op times Rinv.

    Parameters
    ----------
    R : (N, N) sparse matrix
        A sparse square upper triangular matrix with a nonzero diagonal.  A
        CSC matrix with sorted int32 indices and float64 values is used
        without a copy.
    tol : float
        Off-diagonal entries of R^-1 with |value| <= tol are left out of the
        sums (as inv_tr_upper leaves them out of its output).
    perm : (N,) int array, optional
        Row permutation of R^-1: row r of R^-1 is row perm[r] of Rinv
        (sparseqr.rz's column permutation).  The default is no permutation.
    ops : dict of (K, N) sparse matrices, optional
        Operators whose products with Rinv are summed.
    threads : int
        Number of threads to use.  The output does not depend on it.

    Returns
    -------
    E0_sq : (N,) array
        sum_j Rinv[i, j]**2 for each row i.
    op_sq : dict of (K,) arrays
        sum_j (op Rinv)[k, j]**2 for each row k of each op in ops.
    """
    if ops is None:
        ops = {}
    N = R.shape[0]
    Rc = R.tocsc()
    if not Rc.has_sorted_indices:
        Rc = Rc.sorted_indices()
    if Rc.indices.dtype != ITYPE or Rc.indptr.dtype != ITYPE or Rc.data.dtype != FTYPE:
        Rc = sp.csc_matrix((Rc.data.astype(FTYPE), Rc.indices.astype(ITYPE),
                            Rc.indptr.astype(ITYPE)), shape=Rc.shape)
    if perm is None:
        perm = np.arange(N)
    ops = {key: sp.csr_matrix(op) for key, op in ops.items()}

    # blocks of columns, highest columns first, as in inv_tr_upper
    blocks = [(hi, max(hi - BLOCK_COLS, 0)) for hi in range(N, 0, -BLOCK_COLS)]
    local = threading.local()

    def run(block):
        col_hi, col_lo = block
        if not hasattr(local, 'work'):
            local.work = (np.zeros(N, dtype=FTYPE), np.zeros(N, dtype=np.int8))
        # column c of R^-1 has at most c+1 entries, so the block cannot
        # overflow.  The buffers are only touched as they fill; copying the
        # entries out frees them.
        nmax = (col_hi - col_lo) * (col_hi + col_lo + 1) // 2 + 1
        rr, cc, vv = _solve_block(Rc, col_hi, col_lo, nmax, tol, local.work)
        return col_hi, col_lo, rr.copy(), cc.copy(), vv.copy()

    E0_sq = np.zeros(N)
    op_sq = {key: np.zeros(op.shape[0]) for key, op in ops.items()}

    def take(result):
        # add one block's entries to the sums
        col_hi, col_lo, rr, cc, vv = result
        rows = perm[rr]
        E0_sq[:] += np.bincount(rows, weights=vv * vv, minlength=N)
        if len(ops) == 0:
            return
        B = sp.csc_matrix((vv, (rows, cc - col_lo)), shape=(N, col_hi - col_lo))
        for key, op in ops.items():
            P = (op @ B).tocsr()
            P.data **= 2
            op_sq[key] += np.asarray(P.sum(axis=1)).ravel()

    threads = max(threads, 1)
    with ThreadPoolExecutor(threads) as ex:
        # keep at most 2*threads blocks in flight, and add them in order
        pending = deque()
        todo = iter(blocks)
        for block in todo:
            pending.append(ex.submit(run, block))
            if len(pending) >= 2 * threads:
                break
        while pending:
            take(pending.popleft().result())
            for block in todo:
                pending.append(ex.submit(run, block))
                break
    return E0_sq, op_sq
