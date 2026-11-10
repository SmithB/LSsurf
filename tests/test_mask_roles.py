import numpy as np
import pytest
import pointCollection as pc
from LSsurf.smooth_fit import smooth_fit
from LSsurf.grid_functions import resolve_mask_roles

# domain and grids
W = {'x': 1.e4, 'y': 1000., 't': 2.}
CTR = {'x': 0., 'y': 0., 't': 0.}
SPACING = {'z0': 100., 'dz': 200., 'dt': 0.25}
E_RMS = {'d3z_dx2dt': 0.006, 'd2z_dxdt': 0.006*2500, 'd2z0_dx2': 0.03,
         'dz0_dx': 0.03*2500, 'd2z_dt2': 6000}


def make_data():
    rng = np.random.default_rng(1)
    N = 2000
    x = rng.uniform(-W['x']/2, W['x']/2, N)
    y = rng.uniform(-W['y']/2, W['y']/2, N)
    t = np.where(rng.uniform(size=N) > 0.5, 0.99, -0.99)
    z = 10*np.sin(2*np.pi*x/4000) + t
    return pc.data().from_dict({'x': x, 'y': y, 'z': z, 'time': t,
                                'sigma': np.ones(N)})


def make_mask(fn, t=None):
    # 50-m mask covering the domain, 1 where fn(x, y[, t]) is True
    x = np.arange(-W['x']/2-1000, W['x']/2+1000.1, 50)
    y = np.arange(-W['y']/2-1000, W['y']/2+1000.1, 50)
    if t is None:
        xg, yg = np.meshgrid(x, y)
        return pc.grid.data().from_dict({'x': x, 'y': y, 'z': fn(xg, yg).astype(float)})
    yg, xg, tg = np.meshgrid(y, x, t, indexing='ij')
    return pc.grid.data().from_dict({'x': x, 'y': y, 't': t, 'z': fn(xg, yg, tg).astype(float)})


def run(data=None, **kwargs):
    if data is None:
        data = make_data()
    return smooth_fit(data=data, ctr=CTR, W=W, spacing=SPACING, E_RMS=E_RMS,
                      reference_epoch=4, max_iterations=1, VERBOSE=False, **kwargs)


def test_resolve_mask_roles():
    A, B = make_mask(lambda x, y: x < 0), make_mask(lambda x, y: x < 2000)
    # no roles: masks are handled as before roles were introduced
    assert resolve_mask_roles(A) is None
    assert resolve_mask_roles({'z0': A, 'dz': B}) is None
    # a single mask is the default
    assert resolve_mask_roles(A, data_mask='default') == {'z0': 'default', 'dz': 'default', 'data': 'default'}
    # the only mask in a dictionary is used for the roles that are not assigned
    assert resolve_mask_roles({'a': A}, grid_masks={'z0': 'a'}) == {'z0': 'a', 'dz': 'a', 'data': 'a'}
    # otherwise unassigned roles use the default mask
    assert resolve_mask_roles({'default': A, 'b': B}, grid_masks={'dz': 'b'}) == \
        {'z0': 'default', 'dz': 'b', 'data': 'default'}
    with pytest.raises(ValueError):
        resolve_mask_roles({'a': A, 'b': B}, data_mask='a')     # no default for the grids
    with pytest.raises(ValueError):
        resolve_mask_roles({'a': A}, data_mask='c')             # unknown mask
    with pytest.raises(ValueError):
        resolve_mask_roles({'a': A}, grid_masks={'z1': 'a'})    # unknown grid


def test_grid_and_data_masks():
    masks = {'grid': make_mask(lambda x, y: x < 2000), 'data': make_mask(lambda x, y: x <= 0)}
    S = run(mask_data=masks, grid_masks={'z0': 'grid', 'dz': 'grid'}, data_mask='data')
    # data are selected by the data mask
    assert S['data'].x.max() < 50
    assert S['data'].size > 500
    # the grids are masked by the grid mask, which extends beyond the data mask
    z0_mask = S['m']['z0'].mask
    xx = S['m']['z0'].x
    assert np.all(z0_mask[:, xx < 1900])
    assert not np.any(z0_mask[:, xx > 2100])
    dz_mask = S['m']['dz'].mask
    assert np.all(dz_mask[:, S['m']['dz'].x < 1700])
    assert not np.any(dz_mask[:, S['m']['dz'].x > 2100])


def test_unassigned_roles_use_default():
    masks = {'default': make_mask(lambda x, y: np.ones_like(x, dtype=bool)),
             'data': make_mask(lambda x, y: x <= 0)}
    S = run(mask_data=masks, data_mask='data')
    assert S['data'].x.max() < 50
    assert np.all(S['m']['z0'].mask)
    assert np.all(S['m']['dz'].mask)


def test_time_varying_data_mask():
    # the data mask covers x <= 0 before t=0, and x <= 2000 after
    data_mask = make_mask(lambda x, y, t: np.where(t < 0, x <= 0, x <= 2000),
                          t=np.array([-1, -0.01, 0.01, 1]))
    masks = {'default': make_mask(lambda x, y: x < 3000), 'data': data_mask}
    S = run(mask_data=masks, data_mask='data')
    early = S['data'].time < 0
    assert S['data'].x[early].max() < 50
    assert S['data'].x[~early].max() > 1500
    assert S['data'].x[~early].max() < 2050


def test_time_varying_grid_mask():
    # z0 is masked by the union of the epochs of a time-varying mask
    grid_mask = make_mask(lambda x, y, t: np.where(t < 0, x <= 0, x <= 2000),
                          t=np.array([-1, -0.01, 0.01, 1]))
    masks = {'default': make_mask(lambda x, y: x <= 0), 'grid': grid_mask}
    S = run(mask_data=masks, grid_masks={'z0': 'grid', 'dz': 'grid'})
    xx = S['m']['z0'].x
    assert np.all(S['m']['z0'].mask[:, xx < 1900])
    assert S['data'].x.max() < 50
    # the output dz mask is time-varying
    dz = S['m']['dz']
    assert dz.mask.shape == dz.dz.shape
    before, after = dz.time < 0, dz.time > 0
    assert not np.any(dz.mask[:, (dz.x > 400) & (dz.x < 1600)][:, :, before])
    assert np.all(dz.mask[:, (dz.x > 400) & (dz.x < 1600)][:, :, after] == 1)


def test_data_mask_coords():
    # the data mask is evaluated at the fields named in data_mask_coords
    data = make_data()
    data.assign(x_original=data.x + 3000, y_original=data.y.copy())
    masks = {'default': make_mask(lambda x, y: np.ones_like(x, dtype=bool)),
             'data': make_mask(lambda x, y: x <= 0)}
    S = run(data=data, mask_data=masks, data_mask='data',
            data_mask_coords=['x_original', 'y_original', 'time'])
    assert S['data'].x_original.max() < 50
    assert S['data'].x.max() < -2950
