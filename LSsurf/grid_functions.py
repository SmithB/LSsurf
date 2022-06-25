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
import scipy.sparse as sp
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
    z0_mask_data=None
    if args['mask_data'] is not None:
        mask_file = None
        if isinstance(args['mask_data'], dict):
            z0_mask_data=args['mask_data']['z0']
        elif len(args['mask_data'].shape)==3:
            z0_mask_data=args['mask_data'].copy()
            valid_t = (args['mask_data'].t >= bds['t'][0]) & (args['mask_data'].t <= bds['t'][-1])
            z0_mask_data=pc.grid.data().from_dict({
                'x':args['mask_data'].x,
                'y':args['mask_data'].y,
                'z':np.sum(args['mask_data'].z[:,:,valid_t], axis=2)>0
                })
    else:
        mask_file=args['mask_file']

    grids['z0']=fd_grid( [bds['y'], bds['x']], args['spacing']['z0']*np.ones(2),\
                        name='z0', srs_proj4=args['srs_proj4'], mask_file=args['mask_file'],\
                        mask_data=z0_mask_data)

    mask_interp_threshold=0.95 # interpolated masks should cover a  minimal area
    dz_mask_data=args['mask_data']
    if isinstance(args['mask_data'], dict):
        # if we have specified a dictionary for mask_data, use the 'dz' component
        if args['spacing']['dz']==np.diff(args['mask_data']['dz'].x[0:2]):
            # if the stored grid spacing is the same as that requested here,just
            # copy it
            print("\n---copying dz mask---")
            dz_mask_data=dz_mask_data.z
        else:
            # otherwise, interpolate values from the grid object
            dz_mask_data=args['mask_data']['dz']
            mask_interp_threshold=0.5 # make a faithful copy of the input grid

    grids['dz']=fd_grid( [bds['y'], bds['x'], bds['t']], \
                        [args['spacing']['dz'], args['spacing']['dz'], args['spacing']['dt']], \
                        name='dz', col_0=grids['z0'].N_nodes, srs_proj4=args['srs_proj4'], \
                        mask_file=mask_file, mask_data=dz_mask_data,\
                        mask_interp_threshold = mask_interp_threshold)

    grids['z0'].col_N=grids['dz'].col_N
    grids['t']=fd_grid([bds['t']], [args['spacing']['dt']], name='t')
    grids['z0'].cell_area=calc_cell_area(grids['z0'])

    
    if np.any(grids['dz'].delta[0:2]>grids['z0'].delta):
        if dz_mask_data is not None and dz_mask_data.t is not None and len(dz_mask_data.t) > 1:
            # we have a time-dependent grid
            grids['dz'].cell_area = np.zeros(grids['dz'].shape)
            for t_ind, this_t in enumerate(grids['dz'].ctrs[2]):
                if this_t <= dz_mask_data.t[0]:
                    this_mask=dz_mask_data[:,:,0]
                elif this_t >= dz_mask_data.t[-1]:
                    this_mask=dz_mask_data[:,:,-1]
                else:
                    # find the first time slice of mask_data that is gt this time, do a linear interpolation in time
                    # CHECK IF THESE ARE THE RIGHT INDICES
                    i_t = np.argmin(dz_mask_data.t < this_t)-1
                    di = (this_t - dz_mask_data.t[i_t])/(dz_mask_data.t[i_t+1]-dz_mask_data.t[i_t])
                    this_mask = pc.grid.data().from_dict({'x':dz_mask_data.x,
                                                          'y':dz_mask_data.y,
                                                          'z':((dz_mask_data.z[:,:,i_t]*(1-di)+dz_mask_data.z[:,:,i_t+1]*di)>0.9).astype(float)})
                    temp_grid = fd_grid( [bds['y'], bds['x']], args['spacing']['z0']*np.ones(2),\
                         name='z0', srs_proj4=args['srs_proj4'], \
                        mask_data=this_mask)
                    grids['dz'].cell_area[:,:,t_ind] = sum_cell_area(temp_grid, grids['dz'])
        else:
            grids['dz'].cell_area=sum_cell_area(grids['z0'], grids['dz'])
    else:
        grids['dz'].cell_area=calc_cell_area(grids['dz'])
    # last-- multiply the z0 cell area by the z0 mask
    if grids['z0'].mask is not None:
        grids['z0'].cell_area *= grids['z0'].mask
    return grids, bds

def sum_cell_area(grid_f, grid_c, cell_area_f=None, return_op=False, sub0s=None, taper=True):
    # calculate the area of masked cells in a fine grid within the cells of a coarse grid
    if cell_area_f is None:
        cell_area_f = calc_cell_area(grid_f)*grid_f.mask
    if len(cell_area_f.shape)==3:
        grid_3d=True
        dims=list(range(0,3))
    else:
        grid_3d=False
        dims=list(range(0,2))
    n_k=(grid_c.delta[0:2]/grid_f.delta[0:2]+1).astype(int)
    if grid_3d:
        n_k =np.array(list(n_k)+[1])

    temp_grid = fd_grid([grid_f.bds[ii] for ii in dims], deltas=grid_f.delta[dims])
    fine_to_coarse = lin_op(grid=temp_grid).sum_to_grid3( n_k, sub0s=sub0s, taper=True, valid_equations_only=False, dims=dims)
    result=fine_to_coarse.toCSR().dot(cell_area_f.ravel()).reshape(grid_c.shape[dims])

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
            if grid.mask_3d is not None:
                op=lin_op(grid, name=this_name, col_N=col_N).dzdt(lag=lag).ravel()
                # Map the cell area for the nonzero cells in the operator to the output grid
                temp=sp.coo_matrix(((op.v !=0 ).astype(float),(op.r, op.c-grid.col_0)), \
                                   shape=(np.prod(op.dst_grid.shape), grid.cell_area.size))
                # use this matrix to identify the cells that do not have first and last values
                # within the mask
                num_cells = temp.dot(grid.mask_3d.z.ravel().astype(float))
                op.v[np.in1d(op.r, np.flatnonzero(num_cells<2))]=0
                # recreate the matrix with the updated operator:
                temp=sp.coo_matrix((np.abs(op.v)*0.5*lag*grid.delta[2],(op.r, op.c-grid.col_0)), \
                                   shape=(np.prod(op.dst_grid.shape), grid.cell_area.size))
                # use this matrix to calculate the cell area
                op.dst_grid.cell_area = temp.dot(grid.cell_area.ravel()).reshape(op.dst_grid.shape)
            else:
                # no 3-d mask, just use the mask in the grid
                op=lin_op(grid, name=this_name, col_N=col_N).dzdt(lag=lag).ravel()
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

        op.apply_mask(mask=cell_area)

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
            if grid.mask_3d is not None:
                # 3D mask: need to handle the overlap between time steps
                op=lin_op(grid, name=this_name, col_N=col_N)\
                    .sum_to_grid3(kernel_N+1, sub0s=sub0s, lag=lag, taper=True)\
                        .apply_mask(mask=grid.mask_3d.z, time_step_overlap=2)\
                        .apply_mask(mask=cell_area)
    
                # make an sparse matrix that is like the time-difference operator,
                # but with half the absolute value of the entries.  When this is
                # multiplied by the cell area, it will calculate the area within each
                # coarse cell.  This is similar to what sum_cell_area does, but takes
                # into account the time-varying mask.
                temp=sp.coo_matrix((np.abs(op.v )*0.5*grid.delta[2],(op.r, op.c-grid.col_0)), \
                                   shape=(np.prod(op.dst_grid.shape), grid.cell_area.size))
                op.dst_grid.cell_area = temp.dot(np.ones_like(grid.cell_area.ravel()))\
                    .reshape(op.dst_grid.cell_area)
                #grid.cell_area.ravel()).reshape(op.dst_grid.shape)
                #op.dst_grid.cell_area = sum_cell_area(grid, op.dst_grid, \
                #                                      sub0s=sub0s, \
                #                                      cell_area_f=cell_area, \
                #                                      time_step_overlap=2)
            else:
                #2D mask: just use the mask as is
                op=lin_op(grid, name=this_name, col_N=col_N)\
                    .sum_to_grid3(kernel_N+1, sub0s=sub0s, lag=lag, taper=True)\
                    .apply_2d_mask(mask=cell_area)
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
    Mark datapoints for which the z0 mask is zero as invalid
    Inputs:
    data: (pc.data) data structure.
    grids: (dict) dictionary of fd_grid objects generated by setup_grids
    valid_data: (numpy boolean array, size(data)) indicates valid data points
    bds: (dict) a dictionary specifying the domain bounds in x, y, and t (2-element vector for each)
    args: input arguments

    '''
    temp=fd_grid( [bds['y'], bds['x']], [args['spacing']['z0'], args['spacing']['z0']], name='z0', srs_proj4=args['srs_proj4'], mask_file=args['mask_file'], mask_data=args['mask_data'])
    data_mask=lin_op(temp, name='interp_z').interp_mtx(data.coords()[0:2]).toCSR().dot(grids['z0'].mask.ravel())
    data_mask[~np.isfinite(data_mask)]=0
    if np.any(data_mask==0):
        data.index(~(data_mask==0))
        valid_data[valid_data]= ~(data_mask==0)


def validate_by_dz_mask(data, grids, valid_data):

    '''
    Mark datapoints for which the grids['dz'] mask_3d is zero as invalid
    Inputs:
    data: (pc.data) data structure.
    grids: (dict) dictionary of fd_grid objects generated by setup_grids
    valid_data: (numpy boolean array, size(data)) indicates valid data points

    '''
    temp_grid=grids['dz'].copy()
    temp_grid.col_0=0
    temp_grid.col_N=np.prod(temp_grid.shape)
    if grids['dz'].mask_3d is not None:
        data_mask=lin_op(temp_grid, name='interp_z').interp_mtx(data.coords()).toCSR().\
            dot(grids['dz'].mask_3d.z.ravel().astype(float))
    else:
        data_mask=lin_op(temp_grid, name='interp_z').interp_mtx(data.coords()[0:2]).toCSR().\
            dot(grids['dz'].mask.z.ravel().astype(float))

    data_mask[~np.isfinite(data_mask)]=0
    data_mask = data_mask > 0.5
    if np.any(data_mask==0):
        data.index(~(data_mask==0))
        valid_data[valid_data]= ~(data_mask==0)
