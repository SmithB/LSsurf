import numpy as np
import scipy.sparse as sp
import pytest
from LSsurf.inv_tr_upper import inv_tr_upper
from LSsurf.inv_tr_upper_sumsq import inv_tr_upper_sumsq
from test_inv_tr_upper import random_R, TOL


def reference(R, tol, perm, ops):
    """Row sums of squares from the stored Rinv, as smooth_fit computed them."""
    N = R.shape[0]
    rr, cc, vv, status = inv_tr_upper(R, N * N, tol)
    assert status == 0
    Rinv = sp.coo_matrix((vv, (perm[rr], cc)), shape=R.shape).tocsr()
    E0_sq = np.asarray(Rinv.power(2).sum(axis=1)).ravel()
    op_sq = {key: np.asarray((op @ Rinv).power(2).sum(axis=1)).ravel()
             for key, op in ops.items()}
    return E0_sq, op_sq


def random_ops(N, seed):
    rng = np.random.default_rng(seed)
    # a local difference operator, and a dense-ish averaging operator
    diff = sp.diags([np.ones(N - 1), -np.ones(N - 1)], [0, 1], shape=(N - 1, N))
    avg = sp.random(20, N, density=0.3, random_state=rng)
    return {'diff': diff.tocsr(), 'avg': avg.tocsr()}


@pytest.mark.parametrize('N, density, seed', [(5, 0.3, 1), (60, 0.1, 2),
                                              (300, 0.02, 3), (700, 0.01, 4)])
def test_matches_stored_Rinv(N, density, seed):
    R = random_R(N, density=density, seed=seed)
    perm = np.random.default_rng(seed).permutation(N)
    ops = random_ops(N, seed)
    E0_sq, op_sq = inv_tr_upper_sumsq(R, TOL, perm=perm, ops=ops, threads=2)
    E0_ref, op_ref = reference(R, TOL, perm, ops)
    np.testing.assert_allclose(E0_sq, E0_ref, rtol=1.e-12)
    for key in ops:
        np.testing.assert_allclose(op_sq[key], op_ref[key], rtol=1.e-12, atol=1.e-30)


def test_no_perm_no_ops():
    R = random_R(300, density=0.02, seed=5)
    E0_sq, op_sq = inv_tr_upper_sumsq(R, TOL)
    E0_ref, _ = reference(R, TOL, np.arange(300), {})
    np.testing.assert_allclose(E0_sq, E0_ref, rtol=1.e-12)
    assert op_sq == {}


@pytest.mark.parametrize('N, seed', [(300, 6), (700, 7)])
def test_threads_do_not_change_output(N, seed):
    # N > 256 spans several column blocks
    R = random_R(N, density=0.01, seed=seed)
    perm = np.random.default_rng(seed).permutation(N)
    ops = random_ops(N, seed)
    E0_1, op_1 = inv_tr_upper_sumsq(R, TOL, perm=perm, ops=ops, threads=1)
    for threads in [2, 4]:
        E0_T, op_T = inv_tr_upper_sumsq(R, TOL, perm=perm, ops=ops, threads=threads)
        np.testing.assert_array_equal(E0_1, E0_T)
        for key in ops:
            np.testing.assert_array_equal(op_1[key], op_T[key])


def test_input_format_does_not_matter():
    # CSR, CSC, and int64-indexed CSC give the same sums
    R = random_R(400, density=0.02, seed=8)
    out = [inv_tr_upper_sumsq(M, TOL)[0] for M in
           [R, R.tocsc(), sp.csc_matrix((R.tocsc().data, R.tocsc().indices.astype(np.int64),
                                         R.tocsc().indptr.astype(np.int64)), shape=R.shape)]]
    for E0_sq in out[1:]:
        np.testing.assert_array_equal(out[0], E0_sq)


def test_diagonal():
    d = np.array([1., 2., 1.e6, 8., 0.5])
    E0_sq, _ = inv_tr_upper_sumsq(sp.diags(d).tocsr(), TOL)
    np.testing.assert_array_equal(E0_sq, 1 / d**2)
