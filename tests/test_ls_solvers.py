import sys
import numpy as np
import scipy.sparse as sp
import pytest
from LSsurf import ls_solvers
from LSsurf.ls_solvers import solve_ls, solve_spqr


def random_ls(m=600, n=150, scale_decades=0, seed=0):
    """Sparse full-rank A (m x n) and b; columns scaled over scale_decades
    decades, so cond(A'A) grows by ~10**(2*scale_decades)."""
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=0.03, random_state=rng, format='csc')
    A = (A + sp.eye(m, n, format='csc')).tocsc()
    A = A @ sp.diags(10.0 ** np.linspace(0, scale_decades, n))
    b = rng.standard_normal(m)
    return A.tocsc(), b


def collinear_ls(eps, m=600, n=150, k=20, seed=0):
    """Full-rank A whose last k columns are within eps of its first k:
    cond(A) ~ 1/eps.  eps=1e-5 gives rcond(A'A) ~3e-11, near the real
    E1340_N-2420 system's 1.1e-12."""
    rng = np.random.default_rng(seed)
    B = (sp.random(m, n, density=0.03, random_state=rng, format='csc')
         + sp.eye(m, n, format='csc')).tocsc()
    P = sp.random(m, k, density=0.03, random_state=rng, format='csc') + sp.eye(m, k, format='csc')
    A = sp.hstack([B, B[:, :k] + eps * P]).tocsc()
    return A, rng.standard_normal(m)


def rel_diff(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)


@pytest.mark.parametrize('scale_columns', [False, True])
@pytest.mark.parametrize('case', ['scaled', 'eps1e-4', 'eps1e-5'])
def test_cholmod_matches_spqr_without_fallback(case, scale_columns, capsys):
    # the eps cases are ill conditioned enough that the UNREFINED normal
    # equations miss 1e-9 (3.5e-8 and 9.3e-6 measured): refinement has to work
    if case == 'scaled':
        A, b = random_ls(scale_decades=5)
    else:
        A, b = collinear_ls(float(case[3:]))
    x_qr = solve_spqr(A, b)
    x_ch = solve_ls(A, b, solver='cholmod', scale_columns=scale_columns)
    assert 'FALLING BACK' not in capsys.readouterr().out
    assert rel_diff(x_ch, x_qr) < 1e-9


def test_threads_give_the_same_answer():
    A, b = random_ls(m=3000, n=800, seed=3)
    x1 = solve_ls(A, b, threads=1, solver='cholmod', verbose=False)
    x4 = solve_ls(A, b, threads=4, solver='cholmod', verbose=False)
    assert rel_diff(x4, x1) < 1e-12


def test_rank_deficient_falls_back_to_spqr(capsys):
    A, b = random_ls()
    A = sp.hstack([A, A[:, :1]]).tocsc()    # a repeated column
    x = solve_ls(A, b, solver='cholmod')
    assert 'FALLING BACK TO SPQR' in capsys.readouterr().out
    np.testing.assert_array_equal(x, solve_spqr(A, b))


def test_column_scaling_accepts_badly_scaled_columns(capsys):
    # columns over 8 decades: unscaled rcond 1.2e-16 falls back, though the
    # columns are independent; scaled rcond is 0.66 and the solve matches SPQR
    A, b = random_ls(scale_decades=8)
    solve_ls(A, b, solver='cholmod')
    assert 'FALLING BACK TO SPQR' in capsys.readouterr().out
    x_ch = solve_ls(A, b, solver='cholmod', scale_columns=True)
    assert 'FALLING BACK' not in capsys.readouterr().out
    assert rel_diff(x_ch, solve_spqr(A, b)) < 1e-9


def test_column_scaling_keeps_collinearity_in_rcond():
    # scaling removes the column-norm spread from rcond, not near-dependence:
    # collinear columns (eps=1e-6) give rcond ~3e-13 either way
    A, b = collinear_ls(1e-6)
    _, info = ls_solvers.solve_cholmod(A, b)
    _, info_s = ls_solvers.solve_cholmod(A, b, scale_columns=True)
    assert info_s['rcond'] < 1e-12
    assert 0.1 < info_s['rcond'] / info['rcond'] < 10


@pytest.mark.parametrize('case', ['repeated', 'repeated_scaled', 'zero'])
def test_column_scaling_still_falls_back_when_rank_deficient(case, capsys):
    if case == 'repeated':
        A, b = random_ls()
        A = sp.hstack([A, A[:, :1]]).tocsc()
    elif case == 'repeated_scaled':
        # repeat the largest of 5 decades of columns
        A, b = random_ls(scale_decades=5)
        A = sp.hstack([A, A[:, -1:]]).tocsc()
    else:
        A, b = random_ls()
        A = sp.hstack([A, sp.csc_matrix((A.shape[0], 1))]).tocsc()
    x = solve_ls(A, b, solver='cholmod', scale_columns=True)
    assert 'FALLING BACK TO SPQR' in capsys.readouterr().out
    np.testing.assert_array_equal(x, solve_spqr(A, b))


def edited(A, b, seed):
    """A, b with ~10% of the rows dropped and the rest reweighted, as an
    iterate_fit iteration does; the first n rows (A's identity) are kept, so
    the result stays full rank"""
    rng = np.random.default_rng(seed)
    keep = np.flatnonzero((rng.random(A.shape[0]) < 0.9) | (np.arange(A.shape[0]) < A.shape[1]))
    W = sp.diags(rng.uniform(0.5, 2, keep.size))
    return (W @ A[keep]).tocsc(), W @ b[keep]


@pytest.mark.parametrize('scale_columns', [False, True])
def test_cached_ordering_is_reused_and_matches_spqr(scale_columns):
    A, b = random_ls(m=3000, n=800, scale_decades=3, seed=7)
    cache = {}
    x, info = ls_solvers.solve_cholmod(A, b, scale_columns=scale_columns, ordering_cache=cache)
    assert info['ordering'] == 'new'
    perm = cache['perm']
    # a real reordering, so the permutation's direction matters
    assert not np.array_equal(perm, np.arange(A.shape[1]))
    assert rel_diff(x, solve_spqr(A, b)) < 1e-9
    for seed in range(3):
        A1, b1 = edited(A, b, seed)
        x1, info1 = ls_solvers.solve_cholmod(A1, b1, scale_columns=scale_columns, ordering_cache=cache)
        assert info1['ordering'] == 'cached' and cache['perm'] is perm
        assert rel_diff(x1, solve_spqr(A1, b1)) < 1e-9


def test_cached_ordering_is_recomputed_for_new_columns():
    A, b = random_ls(n=150)
    cache = {}
    ls_solvers.solve_cholmod(A, b, ordering_cache=cache)
    A2, b2 = random_ls(n=160, seed=1)
    x2, info2 = ls_solvers.solve_cholmod(A2, b2, ordering_cache=cache)
    assert info2['ordering'] == 'new' and len(cache['perm']) == 160
    assert rel_diff(x2, solve_spqr(A2, b2)) < 1e-9


def test_fill_growth_empties_the_ordering_cache(capsys):
    # an ordering computed for a diagonal A'A, then rows that couple the
    # columns: the cached ordering still solves, but the next call reorders
    n = 150
    A0, b0 = sp.eye(n, format='csc'), np.ones(n)
    A, b = random_ls(n=n)
    cache = {}
    ls_solvers.solve_cholmod(A0, b0, ordering_cache=cache)
    x, info = ls_solvers.solve_cholmod(A, b, ordering_cache=cache)
    assert info['ordering'] == 'cached' and info['fill_ratio'] > ls_solvers.REORDER_FILL_RATIO
    assert cache == {}
    assert rel_diff(x, solve_spqr(A, b)) < 1e-9
    solve_ls(A, b, solver='cholmod', ordering_cache=cache)
    assert 'new METIS ordering' in capsys.readouterr().out


def test_cached_ordering_still_falls_back_when_rank_deficient(capsys):
    # the cache built on a call that falls back must not break the next call
    A, b = random_ls()
    A = sp.hstack([A, sp.csc_matrix((A.shape[0], 1))]).tocsc()
    cache = {}
    for _ in range(2):
        x = solve_ls(A, b, solver='cholmod', ordering_cache=cache)
        assert 'FALLING BACK TO SPQR' in capsys.readouterr().out
        np.testing.assert_array_equal(x, solve_spqr(A, b))


def test_missing_scikit_sparse_is_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'sksparse.cholmod', None)
    A, b = random_ls()
    with pytest.raises(ImportError, match='scikit-sparse'):
        solve_ls(A, b, solver='cholmod')


def test_unknown_solver_is_an_error():
    A, b = random_ls()
    with pytest.raises(ValueError):
        solve_ls(A, b, solver='lsqr')


def test_spqr_is_the_default_and_unchanged():
    A, b = random_ls(seed=5)
    import sparseqr
    from threadpoolctl import threadpool_limits
    # same thread count on both sides: BLAS threads change the rounding
    with threadpool_limits(limits={'openmp': 1, 'blas': 1}):
        x_ref = np.asarray(sparseqr.solve(A.tocoo(), b, ordering=6)).ravel()
    np.testing.assert_array_equal(solve_ls(A, b), x_ref)


@pytest.mark.parametrize('cache_ordering', [False, True])
@pytest.mark.parametrize('scale_columns', [False, True])
def test_smooth_fit_cholmod_matches_spqr(scale_columns, cache_ordering, capsys):
    from test_smooth_fit import run_fit
    # three iterations, so a cached ordering gets reused
    S_qr = run_fit(solver='spqr', max_iterations=3)
    capsys.readouterr()
    S_ch = run_fit(solver='cholmod', cholmod_scale_columns=scale_columns,
                   cholmod_cache_ordering=cache_ordering, max_iterations=3)
    out = capsys.readouterr().out
    assert 'FALLING BACK' not in out
    if cache_ordering:
        assert out.count('new METIS ordering') == 1 and 'cached ordering' in out
    else:
        assert 'METIS ordering' not in out and 'cached ordering' not in out
    for group in ['z0', 'dz']:
        for field in S_qr['m'][group].fields:
            a = getattr(S_qr['m'][group], field)
            c = getattr(S_ch['m'][group], field)
            np.testing.assert_allclose(c, a, rtol=0, atol=1e-6, equal_nan=True,
                                       err_msg=f'{group}.{field}')
