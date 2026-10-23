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

scale_columns=True (opt-in, cholmod only) factors D A'A D instead of A'A, with
D = diag(1/||a_j||), so the normal matrix has a unit diagonal, then maps the
solution back (x = D y).  The solution is the same, and the factorization has
the same sparsity and cost, but CHOLMOD's rcond, (min diag(L)/max diag(L))**2,
stops measuring the spread of column norms (units and weights) and measures
near-dependence between columns: on the north-Greenland E210_N-680 fits the
unscaled rcond is 3e-16 to 1.4e-14, mostly below RCOND_MIN, though the solution
matches SPQR to 1e-8 m; scaled, it is 1e-7 to 2e-5.  Scaling adds ~1% to the
solve.  A repeated column still gives a singular scaled matrix.

ordering_cache (opt-in, cholmod only): a dict that keeps the METIS ordering
between calls.  The first call factors as usual and stores the factor's
permutation there, at no extra cost.  Later calls with the same number of
columns permute A'A by it and give CHOLMOD no ordering of its own (just a
postorder), instead of recomputing METIS.
smooth_fit's iterations change only which rows are used and how they are
weighted.  On E210_N-680, METIS is 4.3 s of the 7.7 s analysis and
factorization, and the later iterations' A'A has no entries the first
iteration's did not have.  With the cached ordering, their analysis and
factorization take 4.3 s, with nnz(L) within 1.2% and solutions within
2e-11 m of a fresh ordering.  If a factor ends up with more than REORDER_FILL_RATIO times the
nnz(L) the ordering had when it was computed, the cache is emptied and the
next call reorders.  (scikit-sparse 0.5 cannot take a given ordering:
order='given' silently uses AMD.  Refactorizing an existing factor with a new
matrix is ~35x slower, so each call builds a new factor.)
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
# with an ordering cache: a factor with more than this times the nnz(L) the
# cached ordering gave when it was computed empties the cache, so the next
# call reorders (the pattern of A'A has grown since)
REORDER_FILL_RATIO = 1.1


def factor_nnz(N):
    """
    nnz of the Cholesky factor of N in its given order, from CHOLMOD's
    symbolic factorization.

    WORKAROUND for scikit-sparse 0.5.0: reading CholeskyFactor.nnz (or
    .colcount) leaks a reference to the factor, so it is never freed.  With
    ordering_cache on, solve_cholmod read F.nnz on every call, and each call
    leaked its whole factor (8.7 GB per iteration on the AA tile E-560_N-1240,
    37 GB after 4 iterations).  symbfact gives the same count, without a
    factor.
    """
    from sksparse.cholmod import symbfact
    return int(symbfact(N)[0].sum())


def solve_spqr(A, b, threads=1):
    """SPQR's QR solve, trying METIS (6) then AMD (5) orderings."""
    for ordering in [6, 5]:
        try:
            with threadpool_limits(limits={'openmp': threads, 'blas': threads}):
                return np.asarray(sparseqr.solve(A.tocoo(), b, ordering=ordering)).ravel()
        except Exception as e:
            print(f"for ordering {ordering}, encountered exception {e}")
    raise AssertionError("LSsurf.smooth_fit: did not find an ordering that could solve the LS equations")


def solve_cholmod(A, b, threads=1, verbose=True, scale_columns=False, ordering_cache=None):
    """
    Normal-equation solve with refinement.  Returns (x, info); x is None if
    the solve should fall back to SPQR, and info['reason'] says why.
    scale_columns=True factors the unit-diagonal D A'A D; ordering_cache (a
    dict) keeps the METIS ordering between calls; see the module docstring.
    """
    try:
        from sksparse.cholmod import CholeskyFactor, CholmodError
    except ImportError as e:
        raise ImportError("solver='cholmod' needs scikit-sparse >= 0.5 "
                          "(conda install -c conda-forge scikit-sparse)") from e
    info = {'scaled': scale_columns}
    with threadpool_limits(limits={'openmp': threads, 'blas': threads}):
        tic = time()
        A = A.tocsc()
        AT = A.T.tocsc()
        N = (AT @ A).tocsc()
        info['AtA_s'] = time() - tic
        d = None
        if scale_columns:
            tic = time()
            # d_j = 1/||a_j||.  An all-zero column keeps d_j = 1: its zero
            # diagonal still makes the factorization fail, as it does unscaled
            N_jj = N.diagonal()
            d = 1. / np.sqrt(np.where(N_jj > 0, N_jj, 1.))
            # N_ij * d_i * d_j on the stored values (N is CSC: indices are
            # rows): 0.1 s on E210_N-680, against 0.6 s for D @ N @ D (most of
            # that sorting the indices, which CHOLMOD otherwise does itself)
            N.data *= d[N.indices]
            N.data *= np.repeat(d, np.diff(N.indptr))
            info['scale_s'] = time() - tic

        # the cached ordering, if there is one for this many columns
        p = None
        if ordering_cache is not None and len(ordering_cache.get('perm', ())) == N.shape[0]:
            p = ordering_cache['perm']
            info['ordering'] = 'cached'

        def solve_N(r):
            # (A'A)^-1 r, through the factor of D A'A D when scaling, and of
            # its permutation by p (row and column i are N's p[i]) when cached
            if d is not None:
                r = d * r
            if p is None:
                y = F.solve(r)
            else:
                y = np.empty_like(r)
                y[p] = F.solve(r[p])
            return y if d is None else d * y

        tic = time()
        try:
            # a fresh factor every call: refactorizing one is ~35x slower
            with warnings.catch_warnings():
                # "nearly singular" is expected (cond ~1e12); rcond is reported
                warnings.simplefilter('ignore')
                if p is None:
                    F = CholeskyFactor(N, order='metis')
                    if ordering_cache is not None:
                        # F.perm (METIS, postordered) is a view into F: keep a copy
                        ordering_cache.clear()
                        # WORKAROUND (see factor_nnz): count nnz(L) without F.nnz,
                        # which leaks the factor.  Original:
                        # ordering_cache.update(perm=np.array(F.perm), nnz_L=F.nnz)
                        perm = np.array(F.perm)
                        ordering_cache.update(perm=perm, nnz_L=factor_nnz(N[perm][:, perm]))
                        info['ordering'] = 'new'
                else:
                    N = N[p][:, p]
                    # already ordered and postordered: CHOLMOD's postorder is
                    # the identity, and the factor is the one F.perm gave
                    F = CholeskyFactor(N, order='postordered')
                    # WORKAROUND (see factor_nnz): count nnz(L) without F.nnz,
                    # which leaks the factor.  Original:
                    # fill = F.nnz / ordering_cache['nnz_L']
                    fill = factor_nnz(N) / ordering_cache['nnz_L']
                    if fill > REORDER_FILL_RATIO:
                        info['fill_ratio'] = fill
                        ordering_cache.clear()
                F.factorize(N)
                info['rcond'] = F.rcond
                if not info['rcond'] >= RCOND_MIN:
                    which = "scaled A'A" if scale_columns else "A'A"
                    info['reason'] = f"{which} numerically singular (rcond {info['rcond']:.1e} < {RCOND_MIN:.0e})"
                    return None, info
                x = solve_N(AT @ b)
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
                dx = solve_N(AT @ (b - A @ x))
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


def solve_ls(A, b, threads=1, solver='spqr', verbose=True, scale_columns=False,
             ordering_cache=None):
    """
    min ||A x - b|| with the named solver; see the module docstring.
    scale_columns and ordering_cache apply to the cholmod solve only (not to
    spqr or its fallback).
    """
    if solver == 'spqr':
        return solve_spqr(A, b, threads)
    if solver != 'cholmod':
        raise ValueError(f"solver must be 'spqr' or 'cholmod', got {solver!r}")
    x, info = solve_cholmod(A, b, threads, verbose=verbose, scale_columns=scale_columns,
                            ordering_cache=ordering_cache)
    if verbose:
        steps = ', '.join(f'{r:.1e}' for r in info.get('refine', []))
        scaling = f"column scaling {info['scale_s']:.2f} s, " if 'scale_s' in info else ''
        if info.get('ordering') == 'new':
            ordering = 'new METIS ordering cached, '
        elif 'fill_ratio' in info:
            ordering = f"cached ordering (nnz(L) {info['fill_ratio']:.2f}x its first: reordering next call), "
        elif info.get('ordering') == 'cached':
            ordering = 'cached ordering, '
        else:
            ordering = ''
        rcond_label = 'rcond (scaled)' if info['scaled'] else 'rcond'
        print(f"\tcholmod: A'A {info.get('AtA_s', 0):.1f} s, {scaling}{ordering}"
              f"factor+solve {info.get('factor_s', 0):.1f} s, "
              f"{rcond_label} {info.get('rcond', float('nan')):.1e}, refinement |dx|/|x|: [{steps}] "
              f"in {info.get('refine_s', 0):.1f} s", flush=True)
    if x is None:
        print(f"\tcholmod: FALLING BACK TO SPQR -- {info['reason']}", flush=True)
        return solve_spqr(A, b, threads)
    return x
