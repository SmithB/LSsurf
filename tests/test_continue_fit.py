"""
Continuing a fit: smooth_fit's first_iteration, the bias edits carried by
data.bias_edited, and the sigma_extra carried by the data; and the reference
time of a prior (match_priors.prior_reference_time), which comes from the prior
itself, not from the fit it constrains.
"""
import os
import numpy as np
import pointCollection as pc
from LSsurf.smooth_fit import smooth_fit
from LSsurf.match_priors import prior_reference_time, make_prior_op, match_tile_edges

E_RMS = {'d3z_dx2dt': 0.006, 'd2z_dxdt': 0.006*2500, 'd2z0_dx2': 0.03,
         'dz0_dx': 0.03*2500, 'd2z_dt2': 6000}


def biased_data(N=3000, bad_bias=20., seed=0):
    # a smooth surface measured by 20 sensors, interleaved in space and time;
    # sensor 7 has a large bias
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4500, 4500, N)
    sensor = rng.integers(0, 20, N).astype(float)
    bias = np.where(sensor == 7, bad_bias, 0.)
    return pc.data().from_dict({'x': x, 'y': rng.uniform(-150, 150, N),
                                'time': rng.uniform(-0.9, 0.9, N),
                                'z': 10*np.sin(2*np.pi*x/5000) + bias + rng.normal(0, 0.1, N),
                                'sigma': np.full(N, 0.1), 'sigma_corr': np.ones(N),
                                'sensor': sensor})


def fit(data, **kwargs):
    args = dict(data=data, ctr={'x': 0., 'y': 0., 't': 0.}, W={'x': 1.e4, 'y': 400, 't': 2},
                spacing={'z0': 100, 'dz': 200, 'dt': 0.25}, E_RMS=E_RMS, reference_epoch=4,
                bias_params=['sensor'], bias_nsigma_edit=3, bias_nsigma_iteration=3,
                VERBOSE=False)
    return smooth_fit(**(args | kwargs))


def none_edited(S):
    D = S['data']
    return 'bias_edited' not in D.fields or not np.any(D.bias_edited > 0)


def bad_edited(S):
    D = S['data']
    return 'bias_edited' in D.fields and np.all(D.bias_edited[D.sensor == 7] > 0) \
        and not np.any(D.bias_edited[D.sensor != 7] > 0)


def test_iteration_count_continues_between_passes():
    # one fit, iterations 0-4: the bias of sensor 7 is edited after iteration 3
    S = fit(biased_data(), max_iterations=5)
    assert S['n_iterations'] == 5
    assert bad_edited(S)
    # two passes of 2 and 3 iterations, each counting from zero, never reach
    # bias_nsigma_iteration
    S1 = fit(biased_data(), max_iterations=2)
    assert S1['n_iterations'] == 2 and none_edited(S1)
    S2 = fit(S1['data'], max_iterations=3)
    assert none_edited(S2)
    # the second pass continuing the first's count edits as the single fit does
    S2 = fit(S1['data'], max_iterations=3, first_iteration=S1['n_iterations'])
    assert S2['n_iterations'] == 5
    assert bad_edited(S2)
    assert not np.any(S2['data'].three_sigma_edit[S2['data'].sensor == 7])


def test_bias_edits_and_sigma_extra_are_restored():
    S = fit(biased_data(), max_iterations=5, sigma_extra_relax=True)
    D = S['data']
    assert bad_edited(S)
    # a one-iteration continuation (e.g. a matched fit) keeps the edits, and
    # its last (only) solve is relaxed by the sigma_extra of the fit it continues
    D.three_sigma_edit[:] = True
    S1 = fit(D, max_iterations=1, first_iteration=S['n_iterations'], sigma_extra_relax=True)
    assert S1['n_iterations'] == 6
    assert bad_edited(S1)
    assert not np.any(S1['data'].three_sigma_edit[S1['data'].sensor == 7])
    assert np.allclose(S1['data'].sigma_extra, D.sigma_extra)
    # the restored edits are kept without bias_nsigma_edit, too
    S2 = fit(D, max_iterations=2, first_iteration=S['n_iterations'], bias_nsigma_edit=None)
    assert bad_edited(S2)


def dz_grid(t, t_ref, rate=0.5, x=None, y=None, sigma=0.1):
    # a dz model referenced to t_ref: dz = rate*(t - t_ref), zero error at t_ref
    if x is None:
        x = np.arange(0., 20001., 1000.)
    if y is None:
        y = np.arange(-10000., 10001., 1000.)
    dz = rate*(t[None, None, :]-t_ref)*np.ones((y.size, x.size, 1))
    sigma_dz = np.where(np.abs(t-t_ref) < 1e-6, 0., sigma)[None, None, :]*np.ones((y.size, x.size, 1))
    return pc.grid.data().from_dict({'x': x, 'y': y, 'time': t, 'dz': dz, 'sigma_dz': sigma_dz})


def test_prior_reference_time():
    t = np.arange(2018., 2024.01, 0.25)
    assert prior_reference_time(dz_grid(t, 2020.)) == 2020.
    # no zero epoch: the default
    G = dz_grid(t, 2020.5-0.125)
    assert prior_reference_time(G, default=-1) == -1
    # without errors, the zero epoch of dz
    G = dz_grid(t, 2021.)
    G.fields.remove('sigma_dz')
    assert prior_reference_time(G) == 2021.


def tile_grids():
    # grids for a 20-km tile centered at 0, 0, for 2019-2023, referenced to
    # 2021 (epoch 8)
    data = biased_data(N=200)
    data.time += 2021
    S = smooth_fit(data=data, ctr={'x': 0., 'y': 0., 't': 2021.}, W={'x': 2.e4, 'y': 2.e4, 't': 4},
                   spacing={'z0': 1000, 'dz': 1000, 'dt': 0.25}, E_RMS=E_RMS, reference_epoch=8,
                   return_fit_objects=True, VERBOSE=False)
    return S['grids']


def test_prior_op_drops_zero_sigma():
    grids = tile_grids()
    P = dz_grid(np.arange(2019., 2023.01, 0.25), 2020.).as_points()
    P.assign(t=P.time)
    P.index((P.x <= 10000) & np.isfinite(P.dz))
    # a zero error away from the reference epoch would be an invalid constraint
    P.sigma_dz[(P.t == 2019.) & (P.x == 1000.)] = 0
    op = make_prior_op(grids, P, 'test', ref_time=2020.)
    assert np.all(op.expected > 0)
    assert op.expected.size == np.sum(P.sigma_dz > 0)


def test_match_tile_edges_uses_the_neighbors_reference_time(tmp_path):
    # this fit: 2019-2023, referenced to 2021 (epoch 8).  Its neighbor covers
    # 2018-2024 and is referenced to 2020 (its epoch 8, too)
    grids = tile_grids()
    rate = 0.5
    dz_grid(np.arange(2018., 2024.01, 0.25), 2020., rate=rate).to_h5(
        os.path.join(tmp_path, 'E10_N0.h5'), group='dz')
    ops, _ = match_tile_edges(grids, 8, prior_dir=str(tmp_path), tile_spacing=1.e4,
                              edge_include=2000)
    assert len(ops) == 1
    op = ops[0]
    assert op.expected.size > 0 and np.all(op.expected > 0)
    # a model with dz = rate*(t-2021) satisfies m(t) - m(2020) = rate*(t-2020)
    # (with this fit's reference time, the constraint would be off by rate)
    m = np.zeros(op.col_N)
    tg = grids['dz'].ctrs[2]
    m[grids['dz'].col_0:grids['dz'].col_N] = (rate*(tg[None, None, :]-2021.)
                                              * np.ones(grids['dz'].shape)).ravel()
    assert np.allclose(op.toCSR().dot(m), op.prior)
