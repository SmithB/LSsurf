#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:17:43 2022

@author: ben
"""

import numpy as np
from LSsurf.fd_grid import fd_grid
from LSsurf.lin_op import lin_op
from LSsurf.unique_by_rows import unique_by_rows
import pointCollection as pc

''' grid functions
    
    This file contain functions to setup grids for least-squares surface-
    fitting problems
    
    Contains:
        setup_grids, sum_cell_area, calc_cell_area, setup_averaging_ops, setup_avg_mask_ops, setup_mask
'''


def setup_grids(args):
    '''
    setup the grids for the problem.

    Inputs:
    args: (dict) dictionary containing input arguments. Required entries:
        W: dictionary with entries 'x','y','t' specifying the domain width in x, y, and time
        ctr: dictionary with entries 'x','y','t' specifying the domain center in x, y, and time
        spacing: dictionary with entries 'z0','dz' and 'dt' specifying the spacing of the z0 grid, the spacing of the dz grid, and the duration of the epochs
        srs_proj4: a proj4 string specifying the data projection       
        mask_file: the mask file which has 1 for points in the domain (data will be used and strong constraints applied)
        mask_data: pointCollection.data object containing the mask.  If this is specified, mask_file is ignored
    Outputs:
        grids: (dict) a dictionary with entries 'z0', 'dz' and 't', each containing a fd_grid object
        bds: (dict) a dictionary specifying the domain bounds in x, y, and t (2-element vector for each)

    Each grid has an assigned location to which its points are mapped in the solution vector.  In this

    From left to right, the grids are z0, then dz
    '''
    bds={coord:args['ctr'][coord]+np.array([-0.5, 0.5])*args['W'][coord] for coord in ('x','y','t')}
    grids=dict()
    if args['mask_data'] is not None:
        mask_file = None
    else:
        mask_file=args['mask_file']

    grids['z0']=fd_grid( [bds['y'], bds['x']], args['spacing']['z0']*np.ones(2),\
         name='z0', srs_proj4=args['srs_proj4'], mask_file=args['mask_file'],\
         mask_data=args['mask_data'])

    grids['dz']=fd_grid( [bds['y'], bds['x'], bds['t']], \
        [args['spacing']['dz'], args['spacing']['dz'], args['spacing']['dt']], \
         name='dz', col_0=grids['z0'].N_nodes, srs_proj4=args['srs_proj4'], \
         mask_file=mask_file, mask_data=args['mask_data'])
    grids['z0'].col_N=grids['dz'].col_N
    grids['t']=fd_grid([bds['t']], [args['spacing']['dt']], name='t')

    grids['z0'].cell_area=calc_cell_area(grids['z0'])
    if np.any(grids['dz'].delta[0:2]>grids['z0'].delta):
        grids['dz'].cell_area=sum_cell_area(grids['z0'], grids['dz'])
    else:
        grids['dz'].cell_area=calc_cell_area(grids['dz'])
    # last-- multiply the z0 cell area by the z0 mask
    if grids['z0'].mask is not None:
        grids['z0'].cell_area *= grids['z0'].mask

    return grids, bds


def sum_cell_area(grid_f, grid_c, cell_area_f=None, return_op=False, sub0s=None, taper=True):
    # calculate the area of masked cells in a coarse grid within the cells of a fine grid
    if cell_area_f is None:
        cell_area_f = calc_cell_area(grid_f)*grid_f.mask
    n_k=(grid_c.delta[0:2]/grid_f.delta[0:2]+1).astype(int)
    temp_grid = fd_grid((grid_f.bds[0:2]), deltas=grid_f.delta[0:2])
    fine_to_coarse = lin_op(grid=temp_grid).sum_to_grid3( n_k, sub0s=sub0s, taper=True, valid_equations_only=False, dims=[0,1])
    result=fine_to_coarse.toCSR().dot(cell_area_f.ravel()).reshape(grid_c.shape[0:2])
    if return_op:
        return result, fine_to_coarse
    return result

def calc_cell_area(grid):
    xg, yg = np.meshgrid(grid.ctrs[1], grid.ctrs[0])
    if grid.srs_proj4 is not None:
        lat=pc.data().from_dict({'x':xg, 'y':yg}).get_latlon(proj4_string=grid.srs_proj4).latitude
        return pc.ps_scale_for_lat(lat)**2*grid.delta[0]*grid.delta[1]
    else:
        return np.ones(grid.shape[0:2])*grid.delta[0]*grid.delta[1]

def sym_range(N, ni, offset=0.5):
    # calculate a range of indices that are symmetric WRT the center of a grid
    out=np.arange((ni*offset), (N/2), (ni))
    if offset==0:
        out=np.r_[-out[-1:0:-1], out]+int(np.floor(N/2))
    else:
        out=np.r_[-out[-1::-1], out]+int(np.floor(N/2))
    out=np.floor(out[np.abs(out-N/2)<= N/2-ni/2]).astype(int)
    return out

def setup_averaging_ops(grid, col_N, args, cell_area=None):
    # build operators that take the average of of the delta-z grid at large scales.
    # these get used both in the averaging and error-calculation codes

    ops={}
    if args['dzdt_lags'] is not None:
        # build the not-averaged dz/dt operators (these are not masked)
        for lag in args['dzdt_lags']:
            this_name='dzdt_lag'+str(lag)
            op=lin_op(grid, name=this_name, col_N=col_N).dzdt(lag=lag)
            op.dst_grid.cell_area=grid.cell_area
            ops[this_name]=op

    # make the averaged ops
    if args['avg_scales'] is None:
        return ops
    if args['dzdt_lags'] is None:
        return ops

    N_grid=[ctrs.size for ctrs in grid.ctrs]
    for scale in args['avg_scales']:
        this_name='avg_dz_'+str(int(scale))+'m'
        kernel_N=np.floor(np.array([scale/dd for dd in grid.delta[0:2]]+[1])).astype(int)

        # subscripts for the centers of the averaged areas
        # assume that the largest averaging offset takes the mean of the center
        # of the grid.  Otherwise, center the cells on odd muliples of the grid
        # spacing
        if scale==np.max(args['avg_scales']):
            offset=0
        else:
            offset=0.5

        grid_ctr_subs=[sym_range(N_grid[0], kernel_N[0], offset=offset),
                       sym_range(N_grid[1], kernel_N[1], offset=offset),
                       np.arange(grid.shape[2], dtype=int)]

        sub0s = np.meshgrid(*grid_ctr_subs, indexing='ij')

        # make the operator
        op=lin_op(grid, name=this_name, col_N=col_N)\
            .sum_to_grid3(kernel_N+1, sub0s=sub0s, taper=True)

        op.apply_2d_mask(mask=cell_area)
        op.dst_grid.cell_area = sum_cell_area(grid, op.dst_grid, sub0s=sub0s, cell_area_f=cell_area)

        if cell_area is not None:
            # if cell area was specified, normalize each row by the input area
            op.normalize_by_unit_product()
        else:
            # divide the values by the kernel area in cells
            op.v /= (kernel_N[0]*kernel_N[1])
        op.dst_grid.cell_area = sum_cell_area(grid, op.dst_grid, sub0s=sub0s, cell_area_f=cell_area)
        ops[this_name]=op

        for lag in args['dzdt_lags']:
            dz_name='avg_dzdt_'+str(int(scale))+'m'+'_lag'+str(lag)
            op=lin_op(grid, name=this_name, col_N=col_N)\
                .sum_to_grid3(kernel_N+1, sub0s=sub0s, lag=lag, taper=True)\
                    .apply_2d_mask(mask=cell_area)
            op.dst_grid.cell_area = sum_cell_area(grid, op.dst_grid, sub0s=sub0s, cell_area_f=cell_area)
            if cell_area is not None:
                # the appropriate weight is expected number of nonzero elements
                # for each nonzero node, times the weight for each time step
                op.normalize_by_unit_product( wt=2/(lag*grid.delta[2]))
            else:
                op.v /= (kernel_N[0]*kernel_N[1])
            ops[dz_name]=op

    return ops

def setup_avg_mask_ops(grid, col_N, avg_masks, dzdt_lags):
    if avg_masks is None:
        return {}
    avg_ops={}
    for name, mask in avg_masks.items():
        this_name=name+'_avg_dz'
        avg_ops[this_name] = lin_op(grid, col_N=col_N, name=this_name).mean_of_mask(mask, dzdt_lag=None)
        for lag in dzdt_lags:
            this_name=name+f'_avg_dzdt_lag{lag}'
            avg_ops[this_name] = lin_op(grid, col_N=col_N,name=this_name).mean_of_mask(mask, dzdt_lag=lag)
    return avg_ops


def setup_mask(data, grids, valid_data, bds, args):
    '''
    Mark datapoints for which the mask is zero as invalid
    Inputs:
    data: (pc.data) data structure.
    grids: (dict) dictionary of fd_grid objects generated by setup_grids
    valid_data: (numpy boolean array, size(data)) indicates valid data points
    bds: (dict) a dictionary specifying the domain bounds in x, y, and t (2-element vector for each)

    '''

    temp=fd_grid( [bds['y'], bds['x']], [args['spacing']['z0'], args['spacing']['z0']], name='z0', srs_proj4=args['srs_proj4'], mask_file=args['mask_file'], mask_data=args['mask_data'])
    data_mask=lin_op(temp, name='interp_z').interp_mtx(data.coords()[0:2]).toCSR().dot(grids['z0'].mask.ravel())
    data_mask[~np.isfinite(data_mask)]=0
    if np.any(data_mask==0):
        data.index(~(data_mask==0))
        valid_data[valid_data]= ~(data_mask==0)