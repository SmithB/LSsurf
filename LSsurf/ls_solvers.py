"""
Sparse least-squares solvers for smooth_fit.iterate_fit:  min ||A x - b||.

solver='spqr' (the default) is the QR solve smooth_fit has always used.

solver='cholmod' is opt-in: a Cholesky factorization of the normal equations
A'A x = A'b (scikit-sparse >= 0.5, conda-forge scikit-sparse), followed by
iterative refinement against the original problem,
    x += (A'A)^-1 A'(b - A x),
which restores the digits the normal equations lose: on the ATL1415 fits
cond(A'A) is ~1e12, and one step takes the error from ~3e-5 m to ~1e-9 m.
On the saved E1340_N-2420 system it is 18 s against 84 s for SPQR at 4
threads, and 30 s against 167 s at 1 (ATL1415 docs/plan_cholmod_fit.sh).

The cholmod solve falls back to SPQR, and says why, when CHOLMOD raises (the
normal matrix is not numerically positive definite), when its condition
estimate says A'A is numerically singular (rcond < RCOND_MIN), or when
refinement does not converge.  A missing scikit-sparse is an error, not a fallback: whoever
asked for cholmod should find out it is not installed.
"""
import warnings
from time import time

import numpy as np
import sparseqr
from threadpoolctl import threadpool_limits

# refinement stops when the update is this small relative to x ...
REFINE_TOL = 1.e-10
# ... after at most this many steps; a last update above FAIL_TOL means the
# normal equations are too ill conditioned to trust: fall back to SPQR.
MAX_REFINE = 3
FAIL_TOL = 1.e-6
# CHOLMOD's reciprocal condition estimate of A'A below this: numerically
# singular, and refinement can converge to A solution that is not SPQR's
# (a repeated column gave rcond 2.9e-16 and no error).  The ATL1415
# E1340_N-2420 system has rcond 1.1e-12.
RCOND_MIN = 1.e-14


def solve_spqr(A, b, threads=1):
    """SPQR's QR solve, trying METIS (6) then AMD (5) orderings."""
    for ordering in [6, 5]:
        try:
            with threadpool_limits(limits={'openmp': threads, 'blas': threads}):
                return np.asarray(sparseqr.solve(A.tocoo(), b, ordering=ordering)).ravel()
        except Exception as e:
            print(f"for ordering {ordering}, encountered exception {e}")
    raise AssertionError("LSsurf.smooth_fit: did not find an ordering that could solve the LS equations")


def solve_cholmod(A, b, threads=1, verbose=True):
    """
    Normal-equation solve with refinement.  Returns (x, info); x is None if
    the solve should fall back to SPQR, and info['reason'] says why.
    """
    try:
        from sksparse.cholmod import CholeskyFactor, CholmodError
    except ImportError as e:
        raise ImportError("solver='cholmod' needs scikit-sparse >= 0.5 "
                          "(conda install -c conda-forge scikit-sparse)") from e
    info = {}
    with threadpool_limits(limits={'openmp': threads, 'blas': threads}):
        tic = time()
        A = A.tocsc()
        AT = A.T.tocsc()
        N = (AT @ A).tocsc()
        info['AtA_s'] = time() - tic
        tic = time()
        try:
            # a fresh factor every call: one reused factor was once 40x slower
            with warnings.catch_warnings():
                # "nearly singular" is expected (cond ~1e12); rcond is reported
                warnings.simplefilter('ignore')
                F = CholeskyFactor(N, order='metis')
                F.factorize(N)
                info['rcond'] = F.rcond
                if not info['rcond'] >= RCOND_MIN:
                    info['reason'] = f"A'A numerically singular (rcond {info['rcond']:.1e} < {RCOND_MIN:.0e})"
                    return None, info
                x = F.solve(AT @ b)
        except CholmodError as e:
            info['reason'] = f'{type(e).__name__}: {e}'
            return None, info
        info['factor_s'] = time() - tic
        tic = time()
        x_norm = np.linalg.norm(x)
        info['refine'] = []
        for _ in range(MAX_REFINE):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dx = F.solve(AT @ (b - A @ x))
            x = x + dx
            rel = np.linalg.norm(dx) / x_norm if x_norm > 0 else np.linalg.norm(dx)
            info['refine'].append(rel)
            if not np.isfinite(rel) or rel < REFINE_TOL:
                break
        info['refine_s'] = time() - tic
    if not np.all(np.isfinite(x)) or not info['refine'][-1] < FAIL_TOL:
        info['reason'] = f"refinement did not converge (last |dx|/|x| = {info['refine'][-1]:.1e})"
        return None, info
    return x, info


def solve_ls(A, b, threads=1, solver='spqr', verbose=True):
    """min ||A x - b|| with the named solver; see the module docstring."""
    if solver == 'spqr':
        return solve_spqr(A, b, threads)
    if solver != 'cholmod':
        raise ValueError(f"solver must be 'spqr' or 'cholmod', got {solver!r}")
    x, info = solve_cholmod(A, b, threads, verbose=verbose)
    if verbose:
        steps = ', '.join(f'{r:.1e}' for r in info.get('refine', []))
        print(f"\tcholmod: A'A {info.get('AtA_s', 0):.1f} s, factor+solve {info.get('factor_s', 0):.1f} s, "
              f"rcond {info.get('rcond', float('nan')):.1e}, refinement |dx|/|x|: [{steps}] "
              f"in {info.get('refine_s', 0):.1f} s", flush=True)
    if x is None:
        print(f"\tcholmod: FALLING BACK TO SPQR -- {info['reason']}", flush=True)
        return solve_spqr(A, b, threads)
    return x
