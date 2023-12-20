#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:41:48 2023

@author: ben
"""

import numpy as np
import LSsurf as LS
import os
import re
import pointCollection as pc
import geopandas as gpd


def get_ATC_xform(filename, url_list_file):
    '''
    Calculate the transform between projected and along-track coordinates for a DEM

    Parameters
    ----------
    filename : str
        DEM filename.
    url_list_file : str
        File containing a URL for each DEM filename.

    Returns
    -------
    xform : dict
        dictionary containing origin and basis vectors for the transform to ATC coordinates.
    poly : numpy.array
        Coordinates of the DEM bounding polygon.

    '''

    # read the PGC metadata
    meta=LS.get_pgc(os.path.basename(filename), url_list_file, targets=['meta'])['meta']
    gdf=gpd.GeoDataFrame.from_features([meta])

    # select the projection based on the first y coordinate of the bounding box
    if meta['bbox'][1] < 0:
        epsg_str='epsg:3031'
    else:
        epsg_str='epsg:3413'

    gdf=gdf.set_crs('epsg:4326').to_crs(epsg_str)

    # find the geometry with the largest area
    areas=[geom.area for geom in gdf.geometry[0].geoms]
    biggest=np.argmax(areas)
    poly = np.c_[[*gdf.geometry[0].geoms[biggest].exterior.coords]]
    ctrs=[tuple(geom.centroid.coords)[0] for geom in gdf.geometry[0].geoms]
    xy0=np.array(ctrs[biggest])

    # the along and across-track vectors are the eigenvectors of the polygon boundary segment differences
    dxy=np.diff( poly[:, 0:2], axis=0 )
    vals, vecs = np.linalg.eig(dxy.T@dxy)
    # along track vector is the eigenvector aligned with local north-south
    vec_order=np.argsort(np.abs(xy0@vecs))[::-1]
    xform = {'origin': xy0, 'basis_vectors':vecs[:,vec_order]}

    return xform, poly

def parse_jitter_bias_grids(m0, G_data, grids):
    sensor_bias_re=re.compile('sensor_.*_jitter')
    m={}
    for key, cols in G_data.TOC['cols'].items():
        if sensor_bias_re.search(key) is None:
            continue
        m[key]=pc.grid.data().from_dict({
            'x':grids[key].ctrs[0],\
            'y':np.array([0]),\
            'z':np.reshape(m0[cols], grids[key].shape)})
        m[key].xform=grids[key].xform
    return m

def setup_DEM_jitter_fit(data, filename, col_0=0,
                     url_list_file=None, res=500,
                     sensor=None,
                     expected_rms_grad=1.e-5, expected_rms_bias=2,
                     skip_plane=False,
                     expected_plane_slope=0.02, expected_plane_bias=5,
                     **kwargs):
    """
    Setup equations to fit the jitter variations in a DEM

    Parameters
    ----------
    data : pointCollection.data
        point data against which to compare the DEM.
    filename : str
        DEM filename.
    col_0 : int
        first column for the DOFs in the fit
    url_list_file : str, optional
        File matching PGC urls to DEM filenames. The default is None.
    res : float, optional
        Resolution of the jitter estimates. The default is 500.
    sensor : int, optional
        if specified, only points with data.sensor==sensor are used
    expected_rms_grad : float, optional
        Expected value for the RMS gradient of the solution. The default is 1.e-5.
    expected_rms_bias : float, optional
        Expected value for the RMS of the solution. The default is 2.
    skip_plane : float, optional
        If True, no plane is fit to the residuals
    expected_plane_slope : float, optional
        Expected slope of the mean tilt of the solution. The default is 0.02.
    expected_plane_bias : float, optional
        Expected mean bias of the solution. The default is 5.

    Returns
    -------
    Gd : LSsurf.lin_op
        Matrix mapping bias parameters to each data point.
    Gc : LSsurf.lin_op
        Matrix measuring the magnitude of model parameters
    xform : dict
        Parameters describing the origin and basis vectors of the transform between along-track and projected coordinates
    this_grid : LSsurf.fd_grid
        Grid object for the along-track fit.
    xy_atc : numpy.array
        along-track coordinates for the data points.
    poly : numpy.array
        Polygon for the DEM, in projected coordinates

    """

    if sensor is not None:
        sensor_rows=np.flatnonzero(data.sensor==sensor)
        D = data[sensor_rows]
        name=f'sensor_{sensor}_jitter'
    else:
        D = data
        name='jitter'

    xform, poly = get_ATC_xform(filename, url_list_file)

    xy_atc = (np.c_[D.x, D.y] - xform['origin']) @ xform['basis_vectors']

    XlR = np.round(np.array([ np.min(xy_atc[:,0]), np.max(xy_atc[:,0]) ])/res + np.array([-1, 1]))*res
    grid = LS.fd_grid([XlR], [res], name=name+'_x_atc', xform=xform)
    grid.col_0 = col_0
    grid.col_N = grid.col_0 + grid.N_nodes

    # data fit
    Gd = LS.lin_op( grid = grid, name=name).interp_mtx([D.x, D.y])
    if not skip_plane > 0:
        # plane fit
        G_plane = LS.lin_op( col_0=Gd.col_N, col_N = Gd.col_N+3 )
        rr=np.arange(Gd.shape[0])
        cc=np.zeros_like(rr)+Gd.col_N
        G_plane.r = np.c_[rr, rr, rr]
        G_plane.c = np.c_[cc+0, cc+1, cc+2]
        G_plane.v = np.c_[(D.x-xform['origin'][0])/1000, (D.y-xform['origin'][1])/1000, np.ones_like(rr)]
        G_plane.N_eq=Gd.shape[0]
        G_plane.ravel()
        G_plane.__update_size_and_shape__()
        Gd.add(G_plane)

    # # if the sensor is specified, remap the rows in the fitting matrix
    if sensor is not None:
        Gd.r=sensor_rows[Gd.r.ravel()].reshape(Gd.r.shape)

    # constraint equations
    Gc_zero = LS.lin_op(grid=grid, name=name+'_bias_zero').one()
    Gc_zero.expected = np.zeros(Gc_zero.shape[0])+expected_rms_bias
    Gc_slope = LS.lin_op(grid=grid, name=name+'_bias_grad').grad()
    Gc_slope.expected= np.zeros(Gc_slope.shape[0])+expected_rms_grad
    constraint_list = [Gc_zero, Gc_slope]
    if not skip_plane > 0:
        Gc_plane = LS.lin_op(col_N=Gd.col_N, name=name+'_plane').data_bias(np.arange(3), val=np.ones(3), col=Gd.col_N+[-3, -2, -1])
        Gc_plane.expected = np.array([expected_plane_slope, expected_plane_slope, expected_plane_bias])
        constraint_list += [Gc_plane]


    return Gd, constraint_list, xform, grid, xy_atc, poly
