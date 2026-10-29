#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to add fine-resolution grids that cover part of a smooth_fit domain

Within a region (e.g. a set of rifts), the model is
    z = z0(x,y) + dz(x,y,t) + z0_fine(x,y) + dz_fine(x,y,t),
with dz_fine(x,y,t_ref) = 0.  The fine grids are fd_grids that span the
region's bounding box, but only the nodes inside the region have columns in
the model (see fd_grid.set_active_nodes).  Constraint equations are generated
only for nodes in the region, and terms for nodes outside the region are
dropped (lin_op.restrict_to_active), so the fine fields are pulled towards
zero at the edges of the region.

The fine grids are specified with the smooth_fit fine_grid_args argument, a
dict with entries:
    region: shapely geometry or pointCollection.grid.data mask (nonzero inside)
    region_file: vector (e.g. geojson, shp, gpkg) or raster (tif, h5) file
        containing the region, used if 'region' is not given
    region_field, region_group: field and group to read from a raster file (optional)
    region_threshold: interpolated mask values above this are in the region (default 0.5)
    spacing: node spacing for the fine grids (required).  Either a number (the
        horizontal spacing of both fine grids), or a dict, as for the smooth_fit
        spacing argument, with entries 'z0' and 'dz' (horizontal spacings of
        z0_fine and dz_fine) and, optionally, 'dt' (time spacing of dz_fine,
        default: the dz grid's time spacing)
    buffer: distance by which the region is expanded (default 0)
    halo: number of nodes outside the region on which constraint equations
        are also centered (default 0)
    E_RMS: dict of constraint weights for the fine grids (required).  Any
        combination of these entries may be given, with at least one for each grid:
            z0_fine: 'd2z0_dx2' (curvature), 'dz0_dx' (gradient), 'z0' (magnitude)
            dz_fine: 'd3z_dx2dt' (curvature of the rate), 'd2z_dxdt' (gradient
                of the rate), 'd2z_dt2' (time curvature), 'dz' (magnitude)
        As in smooth_fit, each constraint adds the integral of
        (quantity/E_RMS)**2 over the region (and over time, for dz_fine) to the
        misfit, alongside (residual/sigma)**2 for each datum, so each value has
        the units of its quantity (e.g. m**-1 for d2z0_dx2, m for z0, m**-1
        yr**-1 for d3z_dx2dt).  The equations are scaled by the square root of
        the cell area (or volume), so the values do not depend on the grid spacing.
    add_to_coarse_averages: if True (default) the coarse averaging operators
        (avg_dz, avg_dzdt, dzdt_lag, avg_z0 and masked averages) include the fine
        fields

smooth_fit outputs (in S['m'], and as 'sigma_'+key in S['E'] if errors are calculated):
    z0_fine, dz_fine: the fine fields, NaN outside the region.  dz_fine is
        zero at the reference epoch.
    z0_total_fine, dz_total_fine, dzdt_total_fine_lag{N}: the coarse + fine
        model at the fine-grid nodes (each on its own fine grid).  The coarse and
        fine fields trade off against each other, so their errors are
        correlated: use the errors of the totals for the full model.
    S['m']['z0'] and S['m']['dz'] are the coarse fields only.

Provides: read_region, region_node_mask, fine_grid_spacing, setup_fine_grids,
    setup_fine_constraints, setup_fine_fit, parse_fine_grids,
    setup_fine_total_ops, add_fine_to_averaging_ops, fine_constraint_names
"""

import os
import numpy as np
import scipy.sparse as sp
import pointCollection as pc
from LSsurf.fd_grid import fd_grid
from LSsurf.lin_op import lin_op
from LSsurf.grid_functions import calc_cell_area

fine_grid_arg_defaults = {'region': None, 'region_file': None, 'region_field': None,
                          'region_group': None, 'region_threshold': 0.5,
                          'spacing': None, 'buffer': 0, 'halo': 0,
                          'E_RMS': None, 'add_to_coarse_averages': True}

# E_RMS keys for the fine-grid constraints
fine_z0_constraint_keys = ['d2z0_dx2', 'dz0_dx', 'z0']
fine_dz_constraint_keys = ['d3z_dx2dt', 'd2z_dxdt', 'd2z_dt2', 'dz']

# names of the fine-grid constraint equations, for reporting residuals
fine_constraint_names = ['grad2_z0_fine', 'grad_z0_fine', 'mag_z0_fine',
                         'grad2_dzdt_fine', 'grad_dzdt_fine', 'd2z_dt2_fine', 'mag_dz_fine']


def read_region(filename, srs_proj4=None, bounds=None, field=None, group=None):
    '''
    Read a region from a vector or raster file

    Parameters
    ----------
    filename : str
        vector file readable by ogr (e.g. geojson, shp, gpkg), or raster file
        (tif, vrt, h5, nc).
    srs_proj4 : str, optional
        projection of the fit.  Vector geometries are transformed to it if their
        file specifies a spatial reference. Rasters must be in the fit's projection.
    bounds : iterable, optional
        [[xmin, xmax], [ymin, ymax]] subset of a raster to read.
    field, group : str, optional
        field and group to read from an hdf5 raster.

    Returns
    -------
    shapely geometry (union of all features) or pointCollection.grid.data
    '''
    file_format = os.path.splitext(filename)[1][1:].lower()
    if file_format in ('tif', 'tiff', 'geotif', 'geotiff', 'vrt', 'h5', 'hdf', 'hdf5', 'nc'):
        kwargs = {}
        if bounds is not None:
            kwargs['bounds'] = bounds
        if group is not None:
            kwargs['group'] = group
        if field is not None and file_format in ('h5', 'hdf', 'hdf5'):
            kwargs['fields'] = [field]
        return pc.grid.data().from_file(filename, **kwargs)

    import shapely
    from osgeo import ogr, osr
    ds = ogr.Open(filename, 0)
    if ds is None:
        raise ValueError(f'read_region: could not open {filename}')
    dst_srs = None
    if srs_proj4 is not None:
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromProj4(srs_proj4)
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geoms = []
    for i_layer in range(ds.GetLayerCount()):
        layer = ds.GetLayer(i_layer)
        xform = None
        src_srs = layer.GetSpatialRef()
        if dst_srs is not None and src_srs is not None:
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            xform = osr.CoordinateTransformation(src_srs, dst_srs)
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            geom = geom.Clone()
            if xform is not None:
                geom.Transform(xform)
            geoms.append(shapely.from_wkb(bytes(geom.ExportToWkb())))
    ds = None
    if len(geoms) == 0:
        raise ValueError(f'read_region: no geometries found in {filename}')
    return shapely.union_all(geoms)


def region_node_mask(grid, region, buffer=0, threshold=0.5, field=None):
    '''
    Find the nodes of a grid that fall inside a region

    Parameters
    ----------
    grid : LSsurf.fd_grid
        grid whose first two dimensions are (y, x)
    region : shapely geometry or pointCollection.grid.data
        polygon(s) defining the region, or a gridded mask (nonzero inside).  If
        the mask has a time dimension, a node is in the region if it is inside
        the mask at any time.
    buffer : float, optional
        distance by which the region is expanded. The default is 0.
    threshold : float, optional
        interpolated mask values greater than this are inside. The default is 0.5.
    field : str, optional
        mask field. The default is 'z' if present, otherwise the first field.

    Returns
    -------
    mask : numpy boolean array, shape grid.shape[0:2]
    '''
    if isinstance(region, pc.grid.data):
        if field is None:
            field = 'z' if 'z' in region.fields else region.fields[0]
        z = np.nan_to_num(getattr(region, field).astype(float))
        if z.ndim == 3:
            z = np.max(z, axis=region.t_axis)
        temp = pc.grid.data().from_dict({'x': region.x, 'y': region.y, 'z': z})
        mask = temp.interp(grid.ctrs[1], grid.ctrs[0], gridded=True) > threshold
        if buffer > 0:
            from scipy.ndimage import binary_dilation
            ni, nj = [int(np.ceil(buffer/delta)) for delta in grid.delta[0:2]]
            di, dj = np.meshgrid(np.arange(-ni, ni+1)*grid.delta[0],
                                 np.arange(-nj, nj+1)*grid.delta[1], indexing='ij')
            mask = binary_dilation(mask, structure=(di**2+dj**2) <= buffer**2)
        return mask
    # otherwise, assume a shapely geometry
    import shapely
    if buffer > 0:
        region = region.buffer(buffer)
    xg, yg = np.meshgrid(grid.ctrs[1], grid.ctrs[0])
    return shapely.contains_xy(region, xg, yg)


def fine_grid_spacing(spacing):
    '''
    Fine-grid spacing as a dict with entries 'z0', 'dz' and 'dt'

    Parameters
    ----------
    spacing : float or dict
        a number (the horizontal spacing of both fine grids), or a dict with
        entries 'z0' and 'dz' (horizontal spacings) and, optionally, 'dt'

    Returns
    -------
    dict with entries 'z0', 'dz', and 'dt' ('dt' is None if not specified)
    '''
    if not isinstance(spacing, dict):
        return {'z0': spacing, 'dz': spacing, 'dt': None}
    unknown = set(spacing) - {'z0', 'dz', 'dt'}
    if len(unknown) > 0 or 'z0' not in spacing or 'dz' not in spacing:
        raise ValueError(f"fine-grid spacing must be a number or a dict with entries 'z0', 'dz' "
                         f"and, optionally, 'dt'; got {spacing}")
    return {'dt': None} | spacing


def region_rectangle(domain, region, spacing, buffer=0, threshold=0.5, field=None, pad=2):
    '''
    Find the nodes of a grid with a given spacing that cover a region

    The nodes are aligned with the lower-left corner of the domain, and span the
    bounding box of the region, padded by 'pad' nodes.

    Parameters
    ----------
    domain : fd_grid
        grid whose first two dimensions span the domain
    region, buffer, threshold, field : see region_node_mask
    spacing : float
        node spacing
    pad : int, optional
        number of nodes of padding around the region. The default is 2.

    Returns
    -------
    bds_yx : list
        bounds of the rectangle in y and x, or None if no nodes are in the region
    region_mask : numpy boolean array
        True for the nodes of the rectangle that are in the region
    '''
    # nodes of a grid covering the domain
    N_full = [int(np.floor((bd[1]-bd[0])/spacing+1.e-6))+1 for bd in domain.bds[0:2]]
    full = fd_grid([[bd[0], bd[0]+(N-1)*spacing] for bd, N in zip(domain.bds[0:2], N_full)],
                   [spacing, spacing])
    mask_full = region_node_mask(full, region, buffer=buffer, threshold=threshold, field=field)
    if not np.any(mask_full):
        return None, None
    # crop to the region's bounding box
    ii, jj = [np.flatnonzero(np.any(mask_full, axis=1-dim)) for dim in [0, 1]]
    i0, i1 = np.maximum(0, ii[0]-pad), np.minimum(N_full[0]-1, ii[-1]+pad)
    j0, j1 = np.maximum(0, jj[0]-pad), np.minimum(N_full[1]-1, jj[-1]+pad)
    return [full.ctrs[0][[i0, i1]], full.ctrs[1][[j0, j1]]], mask_full[i0:i1+1, j0:j1+1]


def setup_fine_grids(grids, region, spacing, reference_epoch, col_0,
                     buffer=0, threshold=0.5, field=None, pad=2, srs_proj4=None):
    '''
    Define fine z0 and dz grids that cover a region within grids['z0']

    Each fine grid spans the bounding box of the region, padded by 'pad' nodes,
    with nodes aligned with the lower-left corner of the domain.  Only the nodes
    in the region are active, and dz_fine is fixed at zero at the reference epoch.

    Parameters
    ----------
    grids : dict
        grids from setup_grids.  The new grids are added as 'z0_fine' and 'dz_fine'
    region : shapely geometry or pointCollection.grid.data
        region in which the fine grids are defined
    spacing : float or dict
        node spacing of the fine grids (see fine_grid_spacing)
    reference_epoch : int
        epoch of grids['dz'] at which dz is zero
    col_0 : int
        first column for the fine grids
    buffer, threshold, field: see region_node_mask
    pad : int, optional
        number of nodes of padding around the region. The default is 2.
    srs_proj4 : str, optional
        projection of the grids

    Returns
    -------
    bool : True if the region contains nodes of both fine grids
    '''
    spacing = fine_grid_spacing(spacing)
    rect = {}
    for name in ['z0', 'dz']:
        rect[name] = region_rectangle(grids['z0'], region, spacing[name], buffer=buffer,
                                      threshold=threshold, field=field, pad=pad)
        if rect[name][0] is None:
            return False

    dt = spacing['dt']
    if dt is None:
        dt = grids['dz'].delta[2]
    t_bds = grids['dz'].bds[2]
    N_t = int(np.floor((t_bds[1]-t_bds[0])/dt+1.e-6))+1
    t_ref = grids['dz'].ctrs[2][reference_epoch]

    z0_f = fd_grid(rect['z0'][0], [spacing['z0']]*2, name='z0_fine', srs_proj4=srs_proj4)
    dz_f = fd_grid(rect['dz'][0] + [[t_bds[0], t_bds[0]+(N_t-1)*dt]], [spacing['dz'], spacing['dz'], dt],
                   name='dz_fine', srs_proj4=srs_proj4)
    i_ref = np.argmin(np.abs(dz_f.ctrs[2]-t_ref))
    if np.abs(dz_f.ctrs[2][i_ref]-t_ref) > 1.e-6*dt:
        raise ValueError(f'setup_fine_grids: no dz_fine epoch at the reference time {t_ref}')
    fixed = np.zeros(dz_f.shape, dtype=bool)
    fixed[:, :, i_ref] = True

    z0_f.set_active_nodes(rect['z0'][1], col_0=col_0)
    dz_f.set_active_nodes(rect['dz'][1], fixed=fixed, col_0=z0_f.col_N)
    for grid in [z0_f, dz_f]:
        grid.cell_area = calc_cell_area(grid)*grid.region_mask
    grids['z0_fine'] = z0_f
    grids['dz_fine'] = dz_f
    return True


def setup_fine_constraints(grids, constraint_op_list, E_RMS, halo=0):
    '''
    Setup smoothness constraints for the fine grids

    E_RMS may have entries 'd2z0_dx2', 'dz0_dx', 'z0' (for z0_fine) and
    'd3z_dx2dt', 'd2z_dxdt', 'd2z_dt2', 'dz' (for dz_fine), with the same
    meaning as in setup_smoothness_constraints.  Each equation is centered on a
    node in the region, and nodes outside the region enter the equations as
    zero.  If halo > 0, equations are also centered on nodes up to 'halo' nodes
    outside the region, which adds weight to the zero boundary condition.

    Outputs:
    None (appends to constraint_op_list)
    '''
    z0_f, dz_f = [grids['z0_fine'], grids['dz_fine']]
    root_delta_A = np.sqrt(np.prod(z0_f.delta))
    root_delta_V = np.sqrt(np.prod(dz_f.delta))
    center_mask = {grid.name: grid.region_mask for grid in [z0_f, dz_f]}
    if halo > 0:
        from scipy.ndimage import binary_dilation
        center_mask = {name: binary_dilation(mask, iterations=halo) for name, mask in center_mask.items()}

    # constraint key: (operator name, grid, operator method, extra arguments)
    constraint_types = {
        'd2z0_dx2': ('grad2_z0_fine', z0_f, 'grad2', {'DOF': 'z0_fine'}),
        'dz0_dx': ('grad_z0_fine', z0_f, 'grad', {'DOF': 'z0_fine'}),
        'z0': ('mag_z0_fine', z0_f, 'one', {'DOF': 'z0_fine'}),
        'd3z_dx2dt': ('grad2_dzdt_fine', dz_f, 'grad2_dzdt', {'DOF': 'dz_fine', 't_lag': 1}),
        'd2z_dxdt': ('grad_dzdt_fine', dz_f, 'grad_dzdt', {'DOF': 'dz_fine', 't_lag': 1}),
        'd2z_dt2': ('d2z_dt2_fine', dz_f, 'd2z_dt2', {'DOF': 'dz_fine'}),
        'dz': ('mag_dz_fine', dz_f, 'one', {'DOF': 'dz_fine'})}
    for key, (name, grid, method, kwargs) in constraint_types.items():
        if E_RMS.get(key) is None:
            continue
        op = getattr(lin_op(grid, name=name), method)(center_mask=center_mask[grid.name], **kwargs)
        op.restrict_to_active()
        root_delta = root_delta_A if grid is z0_f else root_delta_V
        op.expected = np.zeros(op.N_eq) + E_RMS[key]/root_delta
        constraint_op_list.append(op)


def setup_fine_fit(data, grids, G_data, constraint_op_list, fine_grid_args,
                   reference_epoch, srs_proj4=None):
    '''
    Add fine grids for a region to a smooth_fit problem

    Defines the fine grids, adds their interpolation operators to G_data and
    their constraint equations to constraint_op_list.  The new columns follow
    the existing columns in G_data.

    Parameters
    ----------
    data : pointCollection.data
        data for the fit
    grids : dict
        grids for the fit.  'z0_fine' and 'dz_fine' are added
    G_data : LSsurf.lin_op
        data-fitting operator
    constraint_op_list : list
        constraint operators
    fine_grid_args : dict
        see the module docstring
    reference_epoch : int
        epoch of grids['dz'] at which dz is zero
    srs_proj4 : str, optional
        projection of the fit

    Returns
    -------
    bool : True if fine grids were added
    '''
    unknown = set(fine_grid_args) - set(fine_grid_arg_defaults)
    if len(unknown) > 0:
        raise ValueError(f'setup_fine_fit: unknown fine_grid_args entries: {sorted(unknown)}')
    args = fine_grid_arg_defaults | fine_grid_args
    for key in ['spacing', 'E_RMS']:
        if args[key] is None:
            raise ValueError(f'setup_fine_fit: fine_grid_args must specify {key}')
    unknown = set(args['E_RMS']) - set(fine_z0_constraint_keys + fine_dz_constraint_keys)
    if len(unknown) > 0:
        raise ValueError(f'setup_fine_fit: unknown fine_grid_args["E_RMS"] entries: {sorted(unknown)}')
    # nodes without data are determined only by the constraints, so each grid needs at least one
    for name, keys in [('z0_fine', fine_z0_constraint_keys), ('dz_fine', fine_dz_constraint_keys)]:
        if not any(args['E_RMS'].get(key) is not None for key in keys):
            raise ValueError(f'setup_fine_fit: fine_grid_args["E_RMS"] needs at least one of {keys} '
                             f'to constrain {name}')

    region = args['region']
    if region is None:
        if args['region_file'] is None:
            raise ValueError('setup_fine_fit: fine_grid_args must specify region or region_file')
        spacing = fine_grid_spacing(args['spacing'])
        pad = args['buffer'] + 4*np.maximum(spacing['z0'], spacing['dz'])
        region = read_region(args['region_file'], srs_proj4=srs_proj4,
                             bounds=[grids['z0'].bds[1]+np.array([-pad, pad]),
                                     grids['z0'].bds[0]+np.array([-pad, pad])],
                             field=args['region_field'], group=args['region_group'])
    if not setup_fine_grids(grids, region, args['spacing'], reference_epoch, G_data.col_N,
                            buffer=args['buffer'], threshold=args['region_threshold'],
                            field=args['region_field'], pad=2+args['halo'], srs_proj4=srs_proj4):
        print('setup_fine_fit: the region contains no z0_fine or no dz_fine nodes, no fine grids added')
        return False

    for name, coords in [('z0_fine', ['y', 'x']), ('dz_fine', ['y', 'x', 'time'])]:
        G_data.add(lin_op(grids[name], name='interp_'+name)
                   .interp_mtx(data.coords(coords), bounds_error=False)
                   .restrict_to_active(drop_rows=False))
    setup_fine_constraints(grids, constraint_op_list, args['E_RMS'], halo=args['halo'])
    return True


def parse_fine_grids(m0, grids, sigma=False):
    '''
    Map the fine-grid parts of a model vector (or of its errors) to grids

    Nodes outside the region are NaN, and dz_fine is zero at the reference epoch.
    '''
    m = {}
    for name in ['z0_fine', 'dz_fine']:
        grid = grids[name]
        key = 'sigma_'+name if sigma else name
        m[key] = pc.grid.data().from_dict(
            {coord: ctr for coord, ctr in zip(['y', 'x', 'time'], grid.ctrs)} |
            {key: grid.grid_from_cols(m0[grid.col_0:grid.col_N])})
        if not sigma:
            m[key].assign(cell_area=grid.cell_area, mask=grid.region_mask)
    return m


def setup_fine_total_ops(grids, col_N, dzdt_lags=None):
    '''
    Operators that evaluate the full model (coarse + fine) at the fine-grid nodes in the region

    The operators have dst_grid and dst_ind0 set, as for the averaging
    operators from setup_averaging_ops, so their values and errors are parsed
    with the averaging operators.

    Returns
    -------
    dict of lin_ops:
        'z0_total_fine' : z0 + z0_fine
        'dz_total_fine' : dz + dz_fine
        'dzdt_total_fine_lag{N}': time differences of dz_total_fine, for each N in dzdt_lags
    '''
    ops = {}
    for coarse, coords in [('z0', ['y', 'x']), ('dz', ['y', 'x', 'time'])]:
        fine = grids[coarse+'_fine']
        name = coarse+'_total_fine'
        nodes = np.flatnonzero(fine.in_region.ravel())
        # coarse model interpolated to the fine nodes
        op = lin_op(grids[coarse], name=name, col_N=col_N).interp_mtx(fine.pos_for_nodes(nodes))
        # plus the fine model at those nodes
        fine_cols = fine.col_map[nodes]
        rows = np.flatnonzero(fine_cols >= 0)
        op.r = np.concatenate([op.r.ravel(), rows])
        op.c = np.concatenate([op.c.ravel(), fine_cols[rows]])
        op.v = np.concatenate([op.v.ravel(), np.ones(rows.size)])
        op.col_N = np.maximum(col_N, fine.col_N)
        op.N_eq = nodes.size
        op.ind0 = nodes
        op.__update_size_and_shape__()
        op.dst_grid = fd_grid(fine.bds, fine.delta, name=name, coords=coords, srs_proj4=fine.srs_proj4)
        op.dst_grid.cell_area = fine.cell_area
        op.dst_ind0 = nodes
        ops[name] = op

    if dzdt_lags is None:
        return ops
    dz_op = ops['dz_total_fine']
    fine = grids['dz_fine']
    T = dz_op.toCSR(row_N=dz_op.N_eq, col_N=dz_op.col_N)
    row_for_node = np.zeros(fine.N_nodes, dtype=int)-1
    row_for_node[dz_op.dst_ind0] = np.arange(dz_op.N_eq)
    for lag in dzdt_lags:
        name = f'dzdt_total_fine_lag{lag}'
        sub = np.unravel_index(dz_op.dst_ind0, fine.shape)
        these = sub[2] < fine.shape[2]-lag
        sub = [ss[these] for ss in sub]
        r0 = row_for_node[np.ravel_multi_index(sub, fine.shape)]
        r1 = row_for_node[np.ravel_multi_index([sub[0], sub[1], sub[2]+lag], fine.shape)]
        N = r0.size
        D = sp.coo_matrix((np.r_[-np.ones(N), np.ones(N)]/(lag*fine.delta[2]),
                           (np.r_[np.arange(N), np.arange(N)], np.r_[r0, r1])),
                          shape=(N, dz_op.N_eq)).tocsr()
        M = (D @ T).tocoo()
        op = lin_op(None, name=name, col_0=0, col_N=dz_op.col_N)
        op.r, op.c, op.v = [M.row, M.col, M.data]
        op.N_eq = N
        op.__update_size_and_shape__()
        t_bds = fine.bds[2] + np.array([0.5, -0.5])*lag*fine.delta[2]
        op.dst_grid = fd_grid([fine.bds[0], fine.bds[1], t_bds], fine.delta, name=name,
                              coords=['y', 'x', 'time'], srs_proj4=fine.srs_proj4)
        op.dst_grid.cell_area = fine.cell_area
        op.dst_ind0 = np.ravel_multi_index(sub, op.dst_grid.shape)
        op.ind0 = op.dst_ind0
        ops[name] = op
    return ops


def fine_to_coarse_matrix(coarse, fine):
    '''
    Matrix that spreads a fine-grid field onto the nodes of a coarse grid

    P[k, f] = phi_k(x_f) a_f / A_k, where phi_k is the bilinear basis function
    for coarse node k, a_f is the cell area of fine node f, and A_k is the
    cell area of coarse node k.  This is the transpose of bilinear
    interpolation, scaled by cell area, so sum_k A_k P[k, f] = a_f: the area
    integral of the fine field is preserved.  For 3-D grids, the fine field is
    interpolated in time to the coarse epochs.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (coarse.N_nodes, fine.col_N), mapping the fine
    grid's columns to the coarse grid's nodes
    '''
    coarse_2d = fd_grid(coarse.bds[0:2], coarse.delta[0:2], srs_proj4=coarse.srs_proj4)
    fine_2d = fd_grid(fine.bds[0:2], fine.delta[0:2], srs_proj4=fine.srs_proj4)
    nodes_2d = np.flatnonzero(fine.region_mask.ravel())
    S = lin_op(coarse_2d).interp_mtx(fine_2d.pos_for_nodes(nodes_2d), bounds_error=False)\
        .toCSR(row_N=nodes_2d.size, col_N=coarse_2d.N_nodes)
    P_2d = sp.diags(1/calc_cell_area(coarse_2d).ravel()) @ S.T \
        @ sp.diags(calc_cell_area(fine_2d).ravel()[nodes_2d])
    if coarse.N_dims == 2:
        P = P_2d.tocoo()
        cols = fine.col_map[nodes_2d[P.col]]
    else:
        t_fine = fd_grid([fine.bds[2]], [fine.delta[2]])
        T = lin_op(t_fine).interp_mtx([coarse.ctrs[2]], bounds_error=False)\
            .toCSR(row_N=coarse.shape[2], col_N=t_fine.N_nodes)
        P = sp.kron(P_2d, T).tocoo()
        cols = fine.col_map[nodes_2d[P.col // fine.shape[2]]*fine.shape[2] + P.col % fine.shape[2]]
    keep = cols >= 0
    return sp.csr_matrix((P.data[keep], (P.row[keep], cols[keep])), shape=(coarse.N_nodes, fine.col_N))


def add_fine_to_averaging_ops(averaging_ops, grids):
    '''
    Add the fine-grid contributions to averaging operators built on the coarse grids

    For each operator in averaging_ops whose grid is grids['z0'] or grids['dz'],
    entries are added so that the operator applies the same weights to the fine
    field (spread onto the coarse nodes by fine_to_coarse_matrix) as to the
    coarse field.  Averages over an area then include the area integral of the
    fine field, and operators that sample single coarse nodes (e.g. dzdt_lag)
    include the local, basis-function-weighted mean of the fine field.
    '''
    P = {}
    for op in averaging_ops.values():
        if op.grid is None or op.grid.name not in ('z0', 'dz') \
                or op.grid is not grids[op.grid.name] or op.v.size == 0:
            continue
        coarse = op.grid
        fine = grids[coarse.name+'_fine']
        if coarse.name not in P:
            P[coarse.name] = fine_to_coarse_matrix(coarse, fine)
        r = op.r.ravel()
        M = op.toCSR(row_N=int(np.maximum(op.N_eq, r.max()+1)),
                     col_N=int(np.maximum(op.col_N, coarse.col_0+coarse.N_nodes)))
        F = (M[:, coarse.col_0:coarse.col_0+coarse.N_nodes] @ P[coarse.name]).tocoo()
        op.r = np.concatenate([r, F.row])
        op.c = np.concatenate([op.c.ravel(), F.col])
        op.v = np.concatenate([op.v.ravel(), F.data])
        op.col_N = int(np.maximum(op.col_N, fine.col_N))
        op.__update_size_and_shape__()
