import numpy as np
import scipy.sparse as sp
import scipy.linalg
import pytest
from LSsurf.inv_tr_upper import inv_tr_upper

# the kernel compares against tol rounded to a C float
TOL = 1.e-5
TOL32 = float(np.float32(TOL))


def random_R(N, density=0.02, seed=0):
    """
    Random sparse upper-triangular matrix with a dominant diagonal, so that
    its inverse is well conditioned and has entries spanning the tolerance.
    """
    rng = np.random.default_rng(seed)
    U = sp.random(N, N, density=density, random_state=rng, format='coo')
    U = sp.triu(U, k=1)
    U.data = rng.standard_normal(U.data.size) * 0.5
    diag = sp.diags(1 + rng.random(N))
    return (U + diag).tocsr()


def reference(R, tol):
    """
    Thresholded dense inverse, in the kernel's order: columns from last to
    first, rows from the diagonal up within each column.  Also returns a
    mask of entries too close to tol to be decided at round-off.
    """
    N = R.shape[0]
    Rinv = scipy.linalg.solve_triangular(R.toarray(), np.eye(N))
    rows, cols, vals, near = [], [], [], []
    for col in range(N - 1, -1, -1):
        for row in range(col, -1, -1):
            v = Rinv[row, col]
            if row == col or abs(v) > tol:
                rows.append(row)
                cols.append(col)
                vals.append(v)
            near.append(abs(abs(v) - tol) < 1.e-9 * max(1, abs(v)))
    return np.array(rows), np.array(cols), np.array(vals), np.any(near)


def check_against_reference(R, threads):
    rr, cc, vv, status = inv_tr_upper(R, R.shape[0]**2 + 1, TOL, threads=threads)
    assert status == 0
    r0, c0, v0, near = reference(R, TOL32)
    # a random R with an entry this close to tol would make the test flaky
    assert not near
    np.testing.assert_array_equal(rr, r0)
    np.testing.assert_array_equal(cc, c0)
    np.testing.assert_allclose(vv, v0, rtol=1.e-12, atol=1.e-14)
    return rr, cc, vv


def test_new_kernel_is_installed():
    # an old build of inv_tr_upper (no threads argument) left in a checkout
    # would otherwise be tested in its place
    R = sp.identity(3, format='csr')
    inv_tr_upper(R, 10, TOL, threads=2)


@pytest.mark.parametrize('N, density, seed', [(5, 0.3, 1), (60, 0.1, 2),
                                              (300, 0.02, 3), (700, 0.01, 4)])
def test_matches_dense_inverse(N, density, seed):
    R = random_R(N, density=density, seed=seed)
    rr, cc, vv = check_against_reference(R, threads=1)
    # the test must exercise the threshold: some entries dropped, some kept
    assert rr.size < N * (N + 1) // 2
    assert np.sum(rr != cc) > 0


@pytest.mark.parametrize('N, seed', [(300, 5), (700, 6)])
def test_threads_do_not_change_output(N, seed):
    # N > 256 spans several column blocks
    R = random_R(N, density=0.01, seed=seed)
    out1 = inv_tr_upper(R, N * N, TOL, threads=1)
    for threads in [2, 4]:
        outT = inv_tr_upper(R, N * N, TOL, threads=threads)
        for a, b in zip(out1, outT):
            np.testing.assert_array_equal(a, b)
    check_against_reference(R, threads=4)


def test_one_by_one():
    R = sp.csr_matrix(np.array([[4.]]))
    rr, cc, vv, status = inv_tr_upper(R, 1, TOL)
    assert status == 0
    assert rr.tolist() == [0] and cc.tolist() == [0]
    assert vv[0] == 0.25


def test_diagonal():
    # 1/1e6 is below tol: the diagonal is kept regardless
    d = np.array([1., 2., 1.e6, 8., 0.5])
    R = sp.diags(d).tocsr()
    for threads in [1, 3]:
        rr, cc, vv, status = inv_tr_upper(R, 5, TOL, threads=threads)
        assert status == 0
        np.testing.assert_array_equal(rr, [4, 3, 2, 1, 0])
        np.testing.assert_array_equal(cc, [4, 3, 2, 1, 0])
        np.testing.assert_array_equal(vv, 1 / d[::-1])


def test_csc_and_unsorted_input():
    # the kernel copies to CSC with sorted indices; the input format must not matter
    R = random_R(80, density=0.1, seed=7)
    Rc = R.tocsc()
    Rc.indices = Rc.indices.copy()
    # reverse the row order within each column
    for j in range(Rc.shape[1]):
        s = slice(Rc.indptr[j], Rc.indptr[j + 1])
        Rc.indices[s] = Rc.indices[s][::-1]
        Rc.data[s] = Rc.data[s][::-1]
    Rc.has_sorted_indices = False
    out_r = inv_tr_upper(R, 80 * 80, TOL)
    out_c = inv_tr_upper(Rc, 80 * 80, TOL)
    for a, b in zip(out_r, out_c):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('threads', [1, 4])
def test_too_small_nnz_sets_status(threads):
    R = random_R(700, density=0.01, seed=8)
    rr, cc, vv, status = inv_tr_upper(R, 700 * 700, TOL, threads=threads)
    n_full = rr.size
    assert status == 0
    # enough for the first block but not for everything
    for nnz in [n_full - 1, n_full // 2, 10]:
        rr, cc, vv, status = inv_tr_upper(R, nnz, TOL, threads=threads)
        assert status == 1
        assert rr.size <= nnz
    rr, cc, vv, status = inv_tr_upper(R, n_full, TOL, threads=threads)
    assert status == 0 and rr.size == n_full
