import json
import pytest
import numpy as np
import shapely
import pointCollection as pc
from osgeo import osr
from LSsurf.fd_grid import fd_grid
from LSsurf.lin_op import lin_op
from LSsurf.smooth_fit import smooth_fit
from LSsurf.grid_functions import calc_cell_area
import LSsurf.fine_grid_functions as fgf

rng = np.random.default_rng(0)

SRS_PROJ4 = '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'


def make_grids(reference_epoch=2, spacing=50):
    # coarse grids for a 2 km x 2 km domain, 0-2 yr, with fine grids around a curved strip
    grids = {'z0': fd_grid([[-1000, 1000], [-1000, 1000]], [200, 200], name='z0')}
    grids['dz'] = fd_grid([[-1000, 1000], [-1000, 1000], [0, 2]], [500, 500, 0.5], name='dz',
                          col_0=grids['z0'].N_nodes)
    region = shapely.LineString([(-600, -300), (200, 100), (500, 600)]).buffer(150)
    assert fgf.setup_fine_grids(grids, region, spacing, reference_epoch, grids['dz'].col_N)
    return grids, region


def random_active_field(grid):
    # random values at the active nodes, zero elsewhere, on the full rectangle
    active = grid.active.ravel()
    x_full = rng.normal(size=grid.N_nodes)*active
    m = np.zeros(grid.col_N)
    m[grid.col_map[active]] = x_full[active]
    return x_full, m


def test_center_mask():
    # equations built with a center mask are the rows of the full operator centered in the mask
    grid = fd_grid([[0, 1000], [0, 1500], [0, 2]], [100, 100, 0.5])
    mask = rng.uniform(size=grid.shape[0:2]) > 0.6
    full = lin_op(grid, name='g').grad2_dzdt(DOF='z')
    sub = lin_op(grid, name='g').grad2_dzdt(DOF='z', center_mask=mask)
    sub0 = np.unravel_index(full.ind0-grid.col_0, grid.shape)
    centered = mask[sub0[0], sub0[1]]
    assert sub.N_eq == centered.sum()
    assert np.array_equal(sub.ind0, full.ind0[centered])
    x = rng.normal(size=grid.N_nodes)
    assert np.allclose(full.toCSR(row_N=full.N_eq).dot(x)[centered], sub.toCSR(row_N=sub.N_eq).dot(x))


def test_restrict_to_active():
    # restricted operators match the full operators applied to a field that is zero outside the region
    grids, region = make_grids()
    for grid, make_op in [(grids['z0_fine'], lambda **kw: lin_op(grids['z0_fine'], name='g').grad2(DOF='z0_fine', **kw)),
                          (grids['dz_fine'], lambda **kw: lin_op(grids['dz_fine'], name='g').grad2_dzdt(DOF='dz_fine', **kw)),
                          (grids['dz_fine'], lambda **kw: lin_op(grids['dz_fine'], name='g').d2z_dt2(DOF='dz_fine', **kw))]:
        full = make_op()
        res = make_op(center_mask=grid.region_mask).restrict_to_active()
        x_full, m = random_active_field(grid)
        y_full = full.toCSR(col_N=grid.col_0+grid.N_nodes, row_N=full.N_eq)[:, grid.col_0:].dot(x_full)
        y_res = res.toCSR(col_N=grid.col_N, row_N=res.N_eq).dot(m)
        # match rows by sub-operator and center node
        for key, rows in res.TOC['rows'].items():
            if key == 'g':
                continue
            lut = dict(zip(full.ind0[full.TOC['rows'][key]], y_full[full.TOC['rows'][key]]))
            assert np.allclose([lut[i0] for i0 in res.ind0[rows]], y_res[rows])
        assert np.all(res.c >= grid.col_0) and np.all(res.c < grid.col_N)
    # interpolation: points near the region get only their active nodes
    grid = grids['dz_fine']
    pts = [rng.uniform(-1000, 1000, 2000), rng.uniform(-1000, 1000, 2000), rng.uniform(0, 2, 2000)]
    full = lin_op(grid, name='i').interp_mtx(pts, bounds_error=False)
    res = lin_op(grid, name='i').interp_mtx(pts, bounds_error=False).restrict_to_active(drop_rows=False)
    x_full, m = random_active_field(grid)
    good = (full.v != 0).ravel()
    y_full = np.bincount(full.r.ravel()[good], weights=full.v.ravel()[good]*x_full[full.c.ravel()[good]-grid.col_0],
                         minlength=2000)
    assert np.allclose(y_full, res.toCSR(col_N=grid.col_N, row_N=2000).dot(m))


@pytest.mark.parametrize('spacing, expected', [
    (50, {'z0': 50, 'dz': 50, 'dt': 0.5}),
    ({'z0': 25, 'dz': 100, 'dt': 0.25}, {'z0': 25, 'dz': 100, 'dt': 0.25})])
def test_fine_grid_setup(spacing, expected):
    grids, region = make_grids(reference_epoch=2, spacing=spacing)
    z0f, dzf = grids['z0_fine'], grids['dz_fine']
    assert np.all(z0f.delta == expected['z0'])
    assert np.all(dzf.delta == [expected['dz'], expected['dz'], expected['dt']])
    assert z0f.col_0 == grids['dz'].col_N and dzf.col_0 == z0f.col_N
    for grid in [z0f, dzf]:
        # each grid's nodes are in the region, and the region is padded by two nodes
        assert np.array_equal(grid.region_mask, fgf.region_node_mask(grid, region))
        assert not np.any(grid.region_mask[0:2, :]) and not np.any(grid.region_mask[:, -2:])
    assert z0f.N_active == z0f.region_mask.sum()
    # dz_fine has no columns at the reference epoch (t=1)
    i_ref = np.flatnonzero(dzf.ctrs[2] == 1.)[0]
    assert dzf.N_active == dzf.region_mask.sum()*(dzf.shape[2]-1)
    assert not np.any(dzf.active[:, :, i_ref])
    # values map back to the grid: NaN outside the region, zero at fixed nodes
    vals = dzf.grid_from_cols(np.ones(dzf.N_active))
    assert np.all(np.isnan(vals[~dzf.in_region]))
    assert np.all(vals[:, :, i_ref][dzf.region_mask] == 0)
    assert np.all(vals[dzf.active] == 1)


@pytest.mark.parametrize('E_RMS', [
    {'d2z0_dx2': 1.}, {'dz0_dx': 1.}, {'z0': 10.}, {'d2z0_dx2': 1., 'dz0_dx': 10., 'z0': 100.},
    {'d3z_dx2dt': 0.3}, {'d2z_dxdt': 1.}, {'d2z_dt2': 100.}, {'dz': 10.},
    {'d3z_dx2dt': 0.3, 'd2z_dxdt': 1., 'd2z_dt2': 100., 'dz': 10.}])
def test_fine_constraint_choices(E_RMS):
    # each E_RMS entry gives one constraint operator, scaled by the root of its grid's cell area (or volume)
    grids, region = make_grids()
    ops = []
    fgf.setup_fine_constraints(grids, ops, E_RMS)
    names = {'d2z0_dx2': 'grad2_z0_fine', 'dz0_dx': 'grad_z0_fine', 'z0': 'mag_z0_fine',
             'd3z_dx2dt': 'grad2_dzdt_fine', 'd2z_dxdt': 'grad_dzdt_fine', 'd2z_dt2': 'd2z_dt2_fine', 'dz': 'mag_dz_fine'}
    assert [op.name for op in ops] == [names[key] for key in names if key in E_RMS]
    for key in E_RMS:
        op = ops[[op.name for op in ops].index(names[key])]
        grid = grids['z0_fine'] if 'z0' in op.name else grids['dz_fine']
        assert np.allclose(op.expected, E_RMS[key]/np.sqrt(np.prod(grid.delta)))
        assert np.all(op.c >= grid.col_0) and np.all(op.c < grid.col_N)


def test_fine_E_RMS_validation():
    # each fine grid needs at least one constraint, and unknown entries are errors
    grids, region = make_grids()
    grids = {key: grids[key] for key in ['z0', 'dz']}
    for E_RMS, message in [({'d3z_dx2dt': 0.3}, 'z0_fine'), ({'d2z0_dx2': 1.}, 'dz_fine'),
                           ({'d2z0_dx2': 1., 'd3z_dx2dt': 0.3, 'd2z0dx2': 1.}, 'unknown')]:
        with pytest.raises(ValueError, match=message):
            fgf.setup_fine_fit(None, grids, None, [], {'region': region, 'spacing': 50, 'E_RMS': E_RMS}, 2)


def test_fine_grid_spacing():
    assert fgf.fine_grid_spacing(50) == {'z0': 50, 'dz': 50, 'dt': None}
    assert fgf.fine_grid_spacing({'z0': 50, 'dz': 100}) == {'z0': 50, 'dz': 100, 'dt': None}
    for bad in [{'z0': 50}, {'z0': 50, 'dz': 100, 'x': 1}]:
        with pytest.raises(ValueError):
            fgf.fine_grid_spacing(bad)


def test_region_mask_inputs(tmp_path):
    grid = fd_grid([[-1000, 1000], [-1000, 1000]], [50, 50])
    region = shapely.LineString([(-600, -300), (200, 100), (500, 600)]).buffer(150)
    m_poly = fgf.region_node_mask(grid, region)
    # a rasterized version of the polygon gives the same nodes, as does a
    # time-varying mask that covers the polygon in at least one epoch
    xx = np.arange(-1000, 1001, 10.)
    xg, yg = np.meshgrid(xx, xx)
    inside = shapely.contains_xy(region, xg, yg).astype(float)
    assert np.array_equal(m_poly, fgf.region_node_mask(grid, pc.grid.data().from_dict({'x': xx, 'y': xx, 'z': inside})))
    z3 = np.stack([inside*(xg < 0), inside*(xg >= 0)], axis=2)
    mask_3d = pc.grid.data().from_dict({'x': xx, 'y': xx, 't': np.array([0., 1.]), 'z': z3})
    assert np.array_equal(m_poly, fgf.region_node_mask(grid, mask_3d))
    # buffers grow the region
    assert fgf.region_node_mask(grid, region, buffer=100).sum() > m_poly.sum()

    # a GeoJSON file in lon/lat is transformed to the fit's projection
    xy0 = np.array([-1.5e6, -4.5e5])
    poly_ps = shapely.affinity.translate(region, *xy0)
    src = osr.SpatialReference()
    src.ImportFromProj4(SRS_PROJ4)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    for srs in (src, dst):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    xform = osr.CoordinateTransformation(src, dst)
    lonlat = [xform.TransformPoint(x, y)[0:2] for x, y in poly_ps.exterior.coords]
    with open(tmp_path/'region.geojson', 'w') as fh:
        json.dump({'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': {},
             'geometry': {'type': 'Polygon', 'coordinates': [lonlat]}}]}, fh)
    poly_read = fgf.read_region(str(tmp_path/'region.geojson'), srs_proj4=SRS_PROJ4)
    assert shapely.hausdorff_distance(poly_read, poly_ps) < 0.01

    # a geotif mask gives the same nodes as the polygon
    pc.grid.data().from_dict({'x': xx+xy0[0], 'y': xx+xy0[1], 'z': inside})\
        .to_geotif(str(tmp_path/'region.tif'), srs_proj4=SRS_PROJ4)
    mask_read = fgf.read_region(str(tmp_path/'region.tif'), srs_proj4=SRS_PROJ4)
    grid_ps = fd_grid([[-1000+xy0[1], 1000+xy0[1]], [-1000+xy0[0], 1000+xy0[0]]], [50, 50])
    assert np.array_equal(m_poly, fgf.region_node_mask(grid_ps, mask_read))

    # a region with no nodes adds no grids
    grids, _ = make_grids()
    grids = {key: grids[key] for key in ['z0', 'dz']}
    far_away = shapely.Point(5000, 5000).buffer(100)
    assert not fgf.setup_fine_grids(grids, far_away, 50, 2, grids['dz'].col_N)
    assert 'z0_fine' not in grids


def test_fine_to_coarse_conservation():
    grids, region = make_grids()
    for coarse_name in ['z0', 'dz']:
        coarse, fine = grids[coarse_name], grids[coarse_name+'_fine']
        P = fgf.fine_to_coarse_matrix(coarse, fine)
        # the area-weighted sum over coarse nodes recovers the fine cell area
        A_c = calc_cell_area(coarse)
        if coarse.N_dims == 3:
            A_c = np.tile(A_c[:, :, None], [1, 1, coarse.shape[2]])
        a_f = np.broadcast_to(calc_cell_area(fine)[:, :, None] if fine.N_dims == 3 else calc_cell_area(fine), fine.shape)
        active = fine.col_map >= 0
        assert np.allclose(P.T.dot(A_c.ravel())[fine.col_map[active]], a_f.ravel()[active])


def rift(x, y, t):
    d = y - (-200 + 0.3*x)
    taper = 0.25*(1+np.tanh((x+1200)/100))*(1-np.tanh((x-1200)/100))
    return -(4 + 4*t)*np.exp(-0.5*(d/(50+20*t))**2)*taper


def z_true(x, y, t):
    return 20 + 1e-3*x + np.cos(2*np.pi*x/4000)*np.cos(2*np.pi*y/5000) - 0.5*t + rift(x, y, t)


def run_fine_fit(fine_grid_args=None, compute_E=False):
    # DEM-like data over a 4 km x 4 km domain at five times
    D = []
    for t in [0.1, 0.6, 1.1, 1.4, 1.9]:
        xg, yg = [jj.ravel()+rng.uniform(-20, 20, jj.size) for jj in np.meshgrid(np.arange(-1980, 1980, 40.), np.arange(-1980, 1980, 40.))]
        D.append(pc.data().from_dict({'x': xg, 'y': yg, 'time': np.zeros_like(xg)+t, 'sigma': np.zeros_like(xg)+0.2}))
    data = pc.data().from_list(D)
    data.assign(z=z_true(data.x, data.y, data.time) + rng.normal(size=data.size)*data.sigma)
    whole = pc.grid.data().from_dict({'x': np.array([-3000., 3000]), 'y': np.array([-3000., 3000]), 'z': np.ones((2, 2))})
    E_RMS = {'d2z0_dx2': 0.02, 'dz0_dx': 50, 'd3z_dx2dt': 0.003, 'd2z_dxdt': 7.5, 'd2z_dt2': 5000}
    return smooth_fit(data=data, ctr={'x': 0., 'y': 0., 't': 1.}, W={'x': 4000., 'y': 4000., 't': 2.},
                      spacing={'z0': 200, 'dz': 400, 'dt': 0.25}, E_RMS=E_RMS, reference_epoch=4,
                      max_iterations=3, dzdt_lags=[1], avg_masks={'whole': whole},
                      fine_grid_args=fine_grid_args, compute_E=compute_E, VERBOSE=False)


@pytest.mark.parametrize('spacing, E_RMS_z0', [
    (50, {'d2z0_dx2': 1.0}),
    ({'z0': 50, 'dz': 100}, {'d2z0_dx2': 1.0}),
    (50, {'dz0_dx': 1.0, 'z0': 100.})])
def test_fine_fit(spacing, E_RMS_z0):
    # the region stays more than one coarse cell from the domain edges, so the whole-domain
    # average of the fine field is exactly its area integral divided by the domain area
    region = shapely.LineString([(-1200, -560), (1200, 160)]).buffer(300)
    fine_grid_args = {'region': region, 'spacing': spacing,
                      'E_RMS': E_RMS_z0 | {'d3z_dx2dt': 0.3, 'd2z_dt2': 100}}
    S_c = run_fine_fit()
    S_f = run_fine_fit(fine_grid_args, compute_E=True)

    # the fine grids reduce the model error in the region, and do not degrade it outside
    err = {}
    for label, S in [('coarse', S_c), ('fine', S_f)]:
        D = S['data']
        in_region = shapely.contains_xy(region, D.x, D.y)
        e = D.z_est - z_true(D.x, D.y, D.time)
        err[label] = [np.sqrt(np.mean(e[ii]**2)) for ii in [in_region, ~in_region]]
    assert err['fine'][0] < 0.3*err['coarse'][0]
    assert err['fine'][1] < 1.1*err['coarse'][1]

    m, E, grids = S_f['m'], S_f['E'], S_f['grids']
    for key in ['z0_fine', 'dz_fine', 'z0_total_fine', 'dz_total_fine', 'dzdt_total_fine_lag1']:
        assert key in m and 'sigma_'+key in E
    in_reg = grids['dz_fine'].region_mask
    # dz_fine is zero at the reference epoch in the region, and NaN outside it
    assert np.all(m['dz_fine'].dz_fine[:, :, 4][in_reg] == 0)
    assert np.all(np.isnan(m['dz_fine'].dz_fine[:, :, 4][~in_reg]))
    assert np.all(np.isfinite(E['sigma_dz_total_fine'].sigma_dz_total_fine[:, :, 6][in_reg]))
    # the totals are the coarse model plus the fine model
    k = 8
    xg, yg = np.meshgrid(grids['dz_fine'].ctrs[1], grids['dz_fine'].ctrs[0])
    dz_c = lin_op(grids['dz'], name='i').interp_mtx([yg[in_reg], xg[in_reg], np.zeros(in_reg.sum())+grids['dz'].ctrs[2][k]])\
        .toCSR(row_N=in_reg.sum(), col_N=m['all'].size).dot(m['all'])
    assert np.allclose(m['dz_total_fine'].dz_total_fine[:, :, k][in_reg], dz_c + m['dz_fine'].dz_fine[:, :, k][in_reg])
    # ... on each fine grid's own nodes
    spacing = fgf.fine_grid_spacing(spacing)
    assert np.all(grids['z0_fine'].delta == spacing['z0']) and np.all(grids['dz_fine'].delta[0:2] == spacing['dz'])
    in_reg_z0 = grids['z0_fine'].region_mask
    xg, yg = np.meshgrid(grids['z0_fine'].ctrs[1], grids['z0_fine'].ctrs[0])
    z0_c = lin_op(grids['z0'], name='i').interp_mtx([yg[in_reg_z0], xg[in_reg_z0]])\
        .toCSR(row_N=in_reg_z0.sum(), col_N=m['all'].size).dot(m['all'])
    assert np.allclose(m['z0_total_fine'].z0_total_fine[in_reg_z0], z0_c + m['z0_fine'].z0_fine[in_reg_z0])

    # the whole-domain average includes the area integral of dz_fine
    A_c = grids['dz'].cell_area
    for k in [0, 8]:
        coarse_mean = np.sum(A_c*m['dz'].dz[:, :, k])/np.sum(A_c)
        fine_mean = np.nansum(grids['dz_fine'].cell_area*m['dz_fine'].dz_fine[:, :, k])/np.sum(A_c)
        assert np.abs(fine_mean) > 0.01
        assert np.isclose(m['whole_avg_dz'].whole_avg_dz.ravel()[k], coarse_mean + fine_mean, atol=1.e-6)

    # constraint residuals are reported for the fine grids
    assert 'grad2_dzdt_fine' in S_f['RMS']
