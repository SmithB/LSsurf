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


@pytest.mark.parametrize('case', ['scaled', 'eps1e-4', 'eps1e-5'])
def test_cholmod_matches_spqr_without_fallback(case, capsys):
    # the eps cases are ill conditioned enough that the UNREFINED normal
    # equations miss 1e-9 (3.5e-8 and 9.3e-6 measured): refinement has to work
    if case == 'scaled':
        A, b = random_ls(scale_decades=5)
    else:
        A, b = collinear_ls(float(case[3:]))
    x_qr = solve_spqr(A, b)
    x_ch = solve_ls(A, b, solver='cholmod')
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


def test_smooth_fit_cholmod_matches_spqr():
    from test_smooth_fit import run_fit
    S_qr = run_fit(solver='spqr')
    S_ch = run_fit(solver='cholmod')
    for group in ['z0', 'dz']:
        for field in S_qr['m'][group].fields:
            a = getattr(S_qr['m'][group], field)
            c = getattr(S_ch['m'][group], field)
            np.testing.assert_allclose(c, a, rtol=0, atol=1e-6, equal_nan=True,
                                       err_msg=f'{group}.{field}')
