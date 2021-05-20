# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:27:38 2017

@author: ben
"""
import numpy as np
from LSsurf.fd_grid import fd_grid
from LSsurf.lin_op import lin_op
import scipy.sparse as sp
import matplotlib.pyplot as plt
from LSsurf.data_slope_bias import data_slope_bias
#from LSsurf.read_CS2_data import make_test_data
import copy
import sparseqr
from time import time, ctime
from LSsurf.RDE import RDE
from LSsurf.unique_by_rows import unique_by_rows
import os
import h5py
import json
#import LSsurf.op_structure_checks as checks
import pointCollection as pc
#import scipy.sparse.linalg as spl
#from spsolve_tr_upper import spsolve_tr_upper
#from propagate_qz_errors import propagate_qz_errors
from LSsurf.inv_tr_upper import inv_tr_upper

def edit_data_by_subset_fit(N_subset, args):

    W_scale=2./N_subset
    W_subset={'x':args['W']['x']*W_scale, 'y':args['W']['y']*W_scale}
    subset_spacing={key:W_subset[key]/2 for key in list(W_subset)}
    bds={coord:args['ctr'][coord]+np.array([-0.5, 0.5])*args['W'][coord] for coord in ('x','y','t')}

    subset_ctrs=np.meshgrid(np.arange(bds['x'][0]+subset_spacing['x'], bds['x'][1], subset_spacing['x']),\
                            np.arange(bds['y'][0]+subset_spacing['y'], bds['y'][1], subset_spacing['y']))
    valid_data=np.ones_like(args['data'].x, dtype=bool)
    count=0
    for x0, y0 in zip(subset_ctrs[0].ravel(), subset_ctrs[1].ravel()):
        count += 1
        in_bounds= \
            (args['data'].x > x0-W_subset['x']/2) & ( args['data'].x < x0+W_subset['y']/2) & \
            (args['data'].y > y0-W_subset['x']/2) & ( args['data'].y < y0+W_subset['y']/2)
        if in_bounds.sum() < 10:
            valid_data[in_bounds]=False
            continue
        sub_args=copy.deepcopy(args)
        sub_args['N_subset']=None
        sub_args['data']=sub_args['data'][in_bounds]
        sub_args['W_ctr']=W_subset['x']
        sub_args['W'].update(W_subset)
        sub_args['ctr'].update({'x':x0, 'y':y0})
        sub_args['VERBOSE']=False
        sub_args['compute_E']=False
        if 'subset_iterations' in args:
            sub_args['max_iterations']=args['subset_iterations']

        sub_fit=smooth_xytb_fit(**sub_args)

        in_tight_bounds_sub = \
            (sub_args['data'].x > x0-W_subset['x']/4) & (sub_args['data'].x < x0+W_subset['x']/4) & \
            (sub_args['data'].y > y0-W_subset['y']/4) & (sub_args['data'].y < y0+W_subset['y']/4)
        in_tight_bounds_all=\
            (args['data'].x > x0-W_subset['x']/4) & ( args['data'].x < x0+W_subset['x']/4) & \
            (args['data'].y > y0-W_subset['y']/4) & ( args['data'].y < y0+W_subset['x']/4)
        valid_data[in_tight_bounds_all] = valid_data[in_tight_bounds_all] & sub_fit['valid_data'][in_tight_bounds_sub]
    if args['VERBOSE']:
        print("from all subsets, found %d data" % valid_data.sum(), flush=True)
    return valid_data


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
    grids['dz'].cell_area=sum_cell_area(grids['z0'], grids['dz'])

    return grids, bds

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


def assign_bias_ID(data, bias_params=None, bias_name='bias_ID', key_name=None, bias_filter=None, bias_model=None):
    """
    Assign a value to each data point that determines which biases are applied to it.

    parameters:
        data: pointCollection.data instance
        bias_parameters: a list of parameters, each unique combination of which
          defines a different bias
        bias_name: a name for the biases
        key_name: an optional parameter which will be used as the dataset name,
          otherwise a key will be built from the parameter values
        bias_model: a dict containing entries:
            E_bias: a dict of expected bias values for the each biasID, determined from the sigma_corr parameter of the data
            bias_ID_dict: a dict giving the parameter values for each bias_ID (or the key_name if provided)
            bias_param_dict: a dict giving the mapping from parameter values to bias_ID values
    """
    if bias_model is None:
        bias_model={'E_bias':dict(), 'bias_param_dict':dict(), 'bias_ID_dict':dict()}
    bias_ID=np.zeros(data.size)+-9999
    p0=len(bias_model['bias_ID_dict'].keys())
    if bias_params is None:
        # assign all data the same bias
        bias_model['bias_ID_dict'][p0+1]=key_name
        bias_ID=p0+1
        bias_model['E_bias'][p0+1]=np.nanmedian(data.sigma_corr)
    else:
        bias_ID=np.zeros(data.size)
        if bias_filter is not None:
            data_filt=bias_filter(data)
        else:
            data_filt=data
        temp=np.column_stack([getattr(data_filt, bp) for bp in bias_params])

        u_p, i_p=unique_by_rows(temp, return_index=True)
        bias_model['bias_param_dict'].update({param:list() for param in bias_params})
        bias_model['bias_param_dict'].update({'ID':list()})
        for p_num, param_vals in enumerate(u_p):
            this_mask=np.ones(data.size, dtype=bool)
            param_vals_dict={}
            #Identify the data that match the parameter values
            for i_param, param in enumerate(bias_params):
                this_mask = this_mask & (getattr(data, param)==param_vals[i_param])
                param_vals_dict[param]=param_vals[i_param]
                #this_name += '%s%3.2f' % (param, param_vals[i_param])
                bias_model['bias_param_dict'][param].append(param_vals[i_param])
            bias_model['bias_param_dict']['ID'].append(p0+p_num)
            this_ind=np.where(this_mask)[0]
            bias_ID[this_ind]=p0+p_num
            bias_model['bias_ID_dict'][p0+p_num]=param_vals_dict
            bias_model['E_bias'][p0+p_num]=np.nanmedian(data.sigma_corr[this_ind])
    data.assign({bias_name:bias_ID})
    return data, bias_model

def setup_bias_fit(data, bias_model, G_data, constraint_op_list,
                       bias_param_name='data_bias', op_name='data_bias'):
    """
        Setup a set of parameters representing the biases of a set of data

        input arguments:
             data: data for the problem.  Must contain a parameter with the name specified in 'bias_param_name
             bias_model: bias_model dict from assign_bias_params
             G_data: coefficient matrix for least-squares fit.  New bias parameters will be added to right of existing parameters
             constraint_op_list: list of constraint-equation operators
             bias_param_name: name of the parameter used to assign the biases.  Defaults to 'data_bias'
             op_name: the name for the output bias operator.
    """
    # the field in bias_param_name defines the relative column in the bias matrix
    # for the DOF constrained
    col=getattr(data, bias_param_name).astype(int)
    # the new columns are appended to the right of G_data
    col_0=G_data.col_N
    col_N=G_data.col_N+np.max(col)+1
    # The bias matrix is just a 1 in the column for the bias parameter for each
    # data value
    G_bias=lin_op(name=op_name, col_0=col_0, col_N=col_N).data_bias(np.arange(data.size), col=col_0+col)
    # the constraint matrix has a 1 for each column, is zero otherwise
    ii=np.arange(col.max()+1, dtype=int)
    Gc_bias=lin_op(name='constraint_'+op_name, col_0=col_0, col_N=col_N).data_bias(ii,col=col_0+ii)
    for key in bias_model['bias_ID_dict']:
        bias_model['bias_ID_dict'][key]['col']=col_0+key
    # the confidence for each bias parameter being zero is in bias_model['E_bias']
    Gc_bias.expected=np.array([bias_model['E_bias'][ind] for ind in ii])
    if np.any(Gc_bias.expected==0):
        raise(ValueError('found an zero value in the expected biases'))
    constraint_op_list.append(Gc_bias)
    G_data.add(G_bias)

def build_reference_epoch_matrix(G_data, Gc, grids, reference_epoch):
    """
    define the matrix that sets dz[reference_epoch]=0 by removing columns from the solution

    Inputs:
    G_data: (lin_op) matrix representing the mapping between the data and the model
    Gc: (lin_op) matrix representing the constraint equations
    grids: (dict) dictionary of fd_grid objects generated by setup_grids
    reference_epoch: (int) epoch corresponding to the DEM for which dz = 0

    Output:
    (sparse matrix): a matrix that, when multiplied by a matrix or vector including all columns in the full solution,
        produces a matrix that only contains columns that do not correspond to the reference epoch
    """

    # Find the rows and columns that match the reference epoch
    temp_r, temp_c=np.meshgrid(np.arange(0, grids['dz'].shape[0]), np.arange(0, grids['dz'].shape[1]), indexing='ij')
    z02_mask=grids['dz'].global_ind([temp_r.transpose().ravel(), temp_c.transpose().ravel(),\
                  reference_epoch+np.zeros_like(temp_r).ravel()])

    # Identify all of the DOFs that do not include the reference epoch
    cols=np.arange(G_data.col_N, dtype='int')
    include_cols=np.setdiff1d(cols, z02_mask)
    # Generate a matrix that has diagonal elements corresponding to all DOFs except the reference epoch.
    # Multiplying this by a matrix with columns for all model parameters yeilds a matrix with no columns
    # corresponding to the reference epoch.
    return sp.coo_matrix((np.ones_like(include_cols), (include_cols, np.arange(include_cols.size))), \
                       shape=(Gc.col_N, include_cols.size)).tocsc()

def setup_PS_bias(data, G_data, constraint_op_list, grids, bds, args):
    '''
    set up a matrix to fit a smooth POCA-vs-Swath bias
    '''
    grids['PS_bias']=fd_grid( [bds['y'], bds['x']], \
       [args['spacing']['dz'], args['spacing']['dz']],\
       name='PS_bias', srs_proj4=args['srs_proj4'],\
       mask_file=args['mask_file'], mask_data=args['mask_data'], \
       col_0=grids['dz'].col_N)
    ps_mtx=lin_op(grid=grids['PS_bias'], name='PS_bias').\
        interp_mtx(data.coords()[0:2])
    # POCA rows should have zero entries
    temp=ps_mtx.v.ravel()
    temp[np.in1d(ps_mtx.r.ravel(), np.flatnonzero(data.swath==0))]=0
    ps_mtx.v=temp.reshape(ps_mtx.v.shape)
    G_data.add(ps_mtx)
    #Build a constraint matrix for the curvature of the PS bias
    grad2_ps=lin_op(grids['PS_bias'], name='grad2_PS').grad2(DOF='PS_bias')
    grad2_ps.expected=args['E_RMS_d2x_PS_bias']+np.zeros(grad2_ps.N_eq)/\
        np.sqrt(np.prod(grids['dz'].delta[0:2]))
    #Build a constraint matrix for the magnitude of the PS bias
    mag_ps=lin_op(grids['PS_bias'], name='mag_ps').data_bias(\
                ind=np.arange(grids['PS_bias'].N_nodes),
                col=np.arange(grids['PS_bias'].col_0, grids['PS_bias'].col_N))
    mag_ps.expected=args['E_RMS_PS_bias']+np.zeros(mag_ps.N_eq)
    constraint_op_list.append(grad2_ps)
    #constraint_op_list.append(grad_ps)
    constraint_op_list.append(mag_ps)

def setup_smoothness_constraints(grids, constraint_op_list, E_RMS, mask_scale):
    """
    Setup the smoothness constraint operators for dz and z0

    Inputs:
    grids: (dict) dictionary of fd_grid objects generated by setup_grids
    constraint_op_list: (list) list of lin_op objects containing constraint equations that penalize the solution for roughness.
    E_RMS: (dict) constraint weights.  May have entries: 'd2z0_dx2', 'd2z0_dx2', 'd2z0_dx', 'd3z_dx2dt', 'd2z_dxdt', 'd2z_dt2'  Each specifies the penealty for each derivative of the DEM (z0), or the height changes(dz)
    mask_scale: (dict) mapping between mask values (in grids[].mask) and constraint weights.  Keys and values should be floats

    Outputs:
    None (appends to constraint_op_list)
    """
    # make the smoothness constraints for z0
    root_delta_A_z0=np.sqrt(np.prod(grids['z0'].delta))
    grad2_z0=lin_op(grids['z0'], name='grad2_z0').grad2(DOF='z0')
    grad2_z0.expected=E_RMS['d2z0_dx2']/root_delta_A_z0*grad2_z0.mask_for_ind0(mask_scale)
    
    constraint_op_list += [grad2_z0]
    if 'dz0_dx' in E_RMS:
        grad_z0=lin_op(grids['z0'], name='grad_z0').grad(DOF='z0')
        grad_z0.expected=E_RMS['dz0_dx']/root_delta_A_z0*grad_z0.mask_for_ind0(mask_scale)
        constraint_op_list += [grad_z0]

    # make the smoothness constraints for dz
    root_delta_V_dz=np.sqrt(np.prod(grids['dz'].delta))
    if 'd3z_dx2dt' in E_RMS and E_RMS['d3z_dx2dt'] is not None:
        grad2_dz=lin_op(grids['dz'], name='grad2_dzdt').grad2_dzdt(DOF='z', t_lag=1)
        grad2_dz.expected=E_RMS['d3z_dx2dt']/root_delta_V_dz*grad2_dz.mask_for_ind0(mask_scale)
        constraint_op_list += [grad2_dz]
    if 'd2z_dxdt' in E_RMS and E_RMS['d2z_dxdt'] is not None:
        grad_dzdt=lin_op(grids['dz'], name='grad_dzdt').grad_dzdt(DOF='z', t_lag=1)
        grad_dzdt.expected=E_RMS['d2z_dxdt']/root_delta_V_dz*grad_dzdt.mask_for_ind0(mask_scale)
        constraint_op_list += [ grad_dzdt ]
    if 'd2z_dt2' in E_RMS and E_RMS['d2z_dt2'] is not None:
        d2z_dt2=lin_op(grids['dz'], name='d2z_dt2').d2z_dt2(DOF='z')
        d2z_dt2.expected=np.zeros(d2z_dt2.N_eq) + E_RMS['d2z_dt2']/root_delta_V_dz
        constraint_op_list += [d2z_dt2]
    for constraint in constraint_op_list:
        if np.any(constraint.expected==0):
            raise(ValueError(f'found zero value in the expected values for {constraint.name}'))

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
    lat=pc.data().from_dict({'x':xg, 'y':yg}).get_latlon(proj4_string=grid.srs_proj4).latitude
    return pc.ps_scale_for_lat(lat)**2*grid.delta[0]*grid.delta[1]

#def sym_range(N, ni, offset=0.5):
#    out=np.arange(int(ni*offset), int(N/2), int(ni))
#    if offset==0:
#        out=np.r_[-out[-1:0:-1], out]+int(np.floor(N/2))
#    else:
#        out=np.r_[-out[-1::-1], out]+int(np.floor(N/2))
#    out=out[np.abs(out-N/2)<= N/2-ni/2]
#    return out

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
    # build the not-averaged dz/dt operators (these are not masked)
    for lag in args['dzdt_lags']:
        this_name='dzdt_lag'+str(lag)
        op=lin_op(grid, name=this_name, col_N=col_N).dzdt(lag=lag)
        op.dst_grid.cell_area=grid.cell_area
        ops[this_name]=op

    # make the averaged ops
    if args['avg_scales'] is None:
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

def check_data_against_DEM(in_TSE, data, m0, G_data, DEM_tol):
    m1 = m0.copy()
    m1[G_data.TOC['cols']['z0']]=0
    r_DEM=data.z - G_data.toCSR().dot(m1) - data.DEM
    return in_TSE[np.abs(r_DEM[in_TSE]-np.nanmedian(r_DEM[in_TSE]))<DEM_tol]

def iterate_fit(data, Gcoo, rhs, TCinv, G_data, Gc, in_TSE, Ip_c, timing, args,\
                bias_model=None):
    cov_rows=G_data.N_eq+np.arange(Gc.N_eq)

    #print(f"iterate_fit: G.shape={Gcoo.shape}, G.nnz={Gcoo.nnz}, data.shape={data.shape}", flush=True)
    in_TSE_original=np.zeros(data.shape, dtype=bool)
    in_TSE_original[in_TSE]=True

    min_tse_iterations=2
    if args['bias_nsigma_iteration'] is not None:
        min_tse_iterations=np.max([min_tse_iterations, args['bias_nsigma_iteration']+1])

    for iteration in range(args['max_iterations']):
        # build the parsing matrix that removes invalid rows
        Ip_r=sp.coo_matrix((np.ones(Gc.N_eq+in_TSE.size), \
                            (np.arange(Gc.N_eq+in_TSE.size), \
                             np.concatenate((in_TSE, cov_rows)))), \
                           shape=(Gc.N_eq+in_TSE.size, Gcoo.shape[0])).tocsc()

        m0_last = np.zeros(Ip_c.shape[0])

        if args['VERBOSE']:
            print("starting qr solve for iteration %d at %s" % (iteration, ctime()), flush=True)
        # solve the equations
        tic=time();
        m0=Ip_c.dot(sparseqr.solve(Ip_r.dot(TCinv.dot(Gcoo)), Ip_r.dot(TCinv.dot(rhs))));
        timing['sparseqr_solve']=time()-tic

        # calculate the full data residual
        rs_data=(data.z-G_data.toCSR().dot(m0))/data.sigma
        # calculate the robust standard deviation of the scaled residuals for the selected data
        sigma_hat=RDE(rs_data[in_TSE])

        # select the data that have scaled residuals < 3 *max(1, sigma_hat)
        in_TSE_last=in_TSE
        ###testing
        in_TSE = (np.abs(rs_data) < 3.0 * np.maximum(1, sigma_hat))
        
        # if bias_nsigma_edit is specified, check for biases that are more than
        # args['bias_nsigma_edit'] times their expected values.  
        if  args['bias_nsigma_edit'] is not None and iteration >= args['bias_nsigma_iteration']:
            if 'edited' not in bias_model['bias_param_dict']:
                bias_model['bias_param_dict']['edited']=np.zeros_like(bias_model['bias_param_dict']['ID'], dtype=bool)
            bias_dict, slope_bias_dict=parse_biases(m0, bias_model, args['bias_params'])
            bad_bias_IDs=np.array(bias_dict['ID'])\
                [(np.abs(bias_dict['val']) > args['bias_nsigma_edit'] * np.array(bias_dict['expected'])\
                                                                      * np.maximum(1, sigma_hat)) \
                 | bias_model['bias_param_dict']['edited']]
            print(bad_bias_IDs)
            for ID in bad_bias_IDs:
                mask=np.ones(data.size, dtype=bool)
                #Mark the ID as edited (because it will have a bias estimate of zero in subsequent iterations)
                bias_model['bias_param_dict']['edited'][bias_model['bias_param_dict']['ID'].index(ID)]=True
                # mark all data associated with the ID as invalid
                for field, field_val in bias_model['bias_ID_dict'][ID].items():
                    if field in data.fields:
                        mask &= (getattr(data, field).ravel()==field_val)
                in_TSE[mask==1]=0
        if 'editable' in data.fields:
            in_TSE[data.editable==0] = in_TSE_original[data.editable==0]
        in_TSE = np.flatnonzero(in_TSE)

        if args['DEM_tol'] is not None:
            in_TSE = check_data_against_DEM(in_TSE, data, m0, G_data, args['DEM_tol'])

        # quit if the solution is too similar to the previous solution
        if (np.max(np.abs((m0_last-m0)[Gc.TOC['cols']['dz']])) < args['converge_tol_dz']) and (iteration > 2):
            if args['VERBOSE']:
                print("Solution identical to previous iteration with tolerance %3.1f, exiting after iteration %d" % (args['converge_tol_dz'], iteration))
            break
        # select the data that are within 3*sigma of the solution
        if args['VERBOSE']:
            print('found %d in TSE, sigma_hat=%3.3f, dt=%3.0f' % ( in_TSE.size, sigma_hat, timing['sparseqr_solve']), flush=True)
        if iteration > 0:
            if in_TSE.size == in_TSE_last.size and np.all( in_TSE_last == in_TSE ):
                if args['VERBOSE']:
                    print("filtering unchanged, exiting after iteration %d" % iteration)
                break
        if iteration >= min_tse_iterations:
            if sigma_hat <= 1:
                if args['VERBOSE']:
                    print("sigma_hat LT 1, exiting after iteration %d" % iteration, flush=True)
                break
    return m0, sigma_hat, in_TSE, in_TSE_last, rs_data

def parse_biases(m, bias_model, bias_params):
    """
        parse the biases in the ouput model

        inputs:
            m: model vector
            bias_model: the bias model
            bID_dict: the bias parameter dictionary from assign_bias_ID
        output:
            ID_dict: a dictionary giving the parameters and associated bias values for each ibas ID
    """
    slope_bias_dict={}
    b_dict={param:list() for param in bias_params+['val','ID','expected']}
    for item in bias_model['bias_ID_dict']:
        b_dict['val'].append(m[bias_model['bias_ID_dict'][item]['col']])
        b_dict['ID'].append(item)
        b_dict['expected'].append(bias_model['E_bias'][item])
        for param in bias_params:
            b_dict[param].append(bias_model['bias_ID_dict'][item][param])
    if 'slope_bias_dict' in bias_model:

        for key in bias_model['slope_bias_dict']:
            slope_bias_dict[key]={'slope_x':m[bias_model['slope_bias_dict'][key][0]], 'slope_y':m[bias_model['slope_bias_dict'][key][1]]}
    return b_dict, slope_bias_dict

def calc_and_parse_errors(E, Gcoo, TCinv, rhs, Ip_c, Ip_r, grids, G_data, Gc, avg_ops, bias_model, bias_params, dzdt_lags=None, timing={}, error_scale=1):
    tic=time()
    # take the QZ transform of Gcoo  # TEST WHETHER rhs can just be a vector of ones
    z, R, perm, rank=sparseqr.rz(Ip_r.dot(TCinv.dot(Gcoo)), Ip_r.dot(TCinv.dot(rhs)))
    z=z.ravel()
    R=R.tocsr()
    R.sort_indices()
    R.eliminate_zeros()
    timing['decompose_qz']=time()-tic

    E0=np.zeros(R.shape[0])

    # compute Rinv for use in propagating errors.
    # what should the tolerance be?  We will eventually square Rinv and take its
    # row-wise sum.  We care about errors at the cm level, so
    # size(Rinv)*tol^2 = 0.01 -> tol=sqrt(0.01/size(Rinv))~ 1E-4
    tic=time(); RR, CC, VV, status=inv_tr_upper(R, np.int(np.prod(R.shape)/4), 1.e-5);
    # save Rinv as a sparse array.  The syntax perm[RR] undoes the permutation from QZ
    Rinv=sp.coo_matrix((VV, (perm[RR], CC)), shape=R.shape).tocsr(); timing['Rinv_cython']=time()-tic;
    tic=time(); E0=np.sqrt(Rinv.power(2).sum(axis=1)); timing['propagate_errors']=time()-tic;

    # if a scaling for the errors has been provided, mutliply E0 by it
    E0 *= error_scale

    # generate the full E vector.  E0 appears to be an ndarray,
    E0=np.array(Ip_c.dot(E0)).ravel()
    E['sigma_z0']=pc.grid.data().from_dict({'x':grids['z0'].ctrs[1],\
                                     'y':grids['z0'].ctrs[0],\
                                    'sigma_z0':np.reshape(E0[Gc.TOC['cols']['z0']], grids['z0'].shape)})
    E['sigma_dz']=pc.grid.data().from_dict({'x':grids['dz'].ctrs[1],\
                                     'y':grids['dz'].ctrs[0],\
                                    'time':grids['dz'].ctrs[2],\
                                    'sigma_dz': np.reshape(E0[Gc.TOC['cols']['dz']], grids['dz'].shape)})

    # generate the lagged dz errors: [CHECK THIS]
    for key, op in avg_ops.items():
        E['sigma_'+key] = pc.grid.data().from_dict({'x':op.dst_grid.ctrs[1], \
                                          'y':op.dst_grid.ctrs[0], \
                                        'time': op.dst_grid.ctrs[2], \
                                            'sigma_'+key: op.grid_error(Ip_c.dot(Rinv))})

    # generate the grid-mean error for zero lag
    if len(bias_model.keys()) >0:
        E['sigma_bias'], E['sigma_slope_bias'] = parse_biases(E0, bias_model, bias_params)

def parse_model(m, m0, data, R, RMS, G_data, averaging_ops, Gc, Ec, grids, bias_model, args):

    # reshape the components of m to the grid shapes
    m['z0']=pc.grid.data().from_dict({'x':grids['z0'].ctrs[1],\
                                     'y':grids['z0'].ctrs[0],\
                                     'cell_area': grids['z0'].cell_area, \
                                     'mask':grids['z0'].mask, \
                                     'z0':np.reshape(m0[G_data.TOC['cols']['z0']], grids['z0'].shape)})
    m['dz']=pc.grid.data().from_dict({'x':grids['dz'].ctrs[1],\
                                     'y':grids['dz'].ctrs[0],\
                                     'time':grids['dz'].ctrs[2],\
                                     'cell_area':grids['dz'].cell_area, \
                                     'mask':grids['dz'].mask, \
                                     'dz': np.reshape(m0[G_data.TOC['cols']['dz']], grids['dz'].shape)})
    if 'PS_bias' in G_data.TOC['cols']:
        m['dz'].assign({'PS_bias':np.reshape(m0[G_data.TOC['cols']['PS_bias']], grids['dz'].shape[0:2])})

    # calculate height rates and averages
    for key, op  in averaging_ops.items():
        m[key] = pc.grid.data().from_dict({'x':op.dst_grid.ctrs[1], \
                                          'y':op.dst_grid.ctrs[0], \
                                        'time': op.dst_grid.ctrs[2], \
                                        'cell_area':op.dst_grid.cell_area,\
                                            key: op.grid_prod(m0)})

    # report the parameter biases.  Sorted in order of the parameter bias arguments
    if len(bias_model.keys()) > 0:
        m['bias'], m['slope_bias']=parse_biases(m0, bias_model, args['bias_params'])

    # report the entire model vector, just in case we want it.
    m['all']=m0

    # report the geolocation of the output map
    m['extent']=np.concatenate((grids['z0'].bds[1], grids['z0'].bds[0]))

    # parse the resduals to assess the contributions of the total error:
    # Make the C matrix for the constraints
    TCinv_cov=sp.dia_matrix((1./Ec, 0), shape=(Gc.N_eq, Gc.N_eq))
    # scaled residuals
    rc=TCinv_cov.dot(Gc.toCSR().dot(m0))
    # unscaled residuals
    ru=Gc.toCSR().dot(m0)
    for eq_type in ['d2z_dt2','grad2_z0','grad2_dzdt','grad2_PS']:
        if eq_type in Gc.TOC['rows']:
            R[eq_type]=np.sum(rc[Gc.TOC['rows'][eq_type]]**2)
            RMS[eq_type]=np.sqrt(np.mean(ru[Gc.TOC['rows'][eq_type]]**2))
    r=(data.z-data.z_est)[data.three_sigma_edit]
    r_scaled=r/data.sigma[data.three_sigma_edit]
    for ff in ['dz','z0']:
        m[ff].assign({'count':G_data.toCSR()[:,G_data.TOC['cols'][ff]][data.three_sigma_edit,:].T.\
                        dot(np.ones_like(r)).reshape(grids[ff].shape)})
        m[ff].count[m[ff].count==0]=np.NaN
        m[ff].assign({'misfit_scaled_rms':np.sqrt(G_data.toCSR()[:,G_data.TOC['cols'][ff]][data.three_sigma_edit,:].T.dot(r_scaled**2)\
                                        .reshape(grids[ff].shape)/m[ff].count)})
        m[ff].assign({'misfit_rms':np.sqrt(G_data.toCSR()[:,G_data.TOC['cols'][ff]][data.three_sigma_edit,:].T.dot(r**2)\
                                         .reshape(grids[ff].shape)/m[ff].count)})
        if 'tide' in data.fields:
            r_notide=(data.z+np.nan_to_num(data.tide, nan=0.)-data.z_est)[data.three_sigma_edit]
            r_notide_scaled=r_notide/data.sigma[data.three_sigma_edit]
            m[ff].assign({'misfit_notide_rms':np.sqrt(G_data.toCSR()[:,G_data.TOC['cols'][ff]][data.three_sigma_edit,:].T.dot(r_notide**2)\
                                        .reshape(grids[ff].shape)/m[ff].count)})
            m[ff].assign({'misfit_notide_scaled_rms':np.sqrt(G_data.toCSR()[:,G_data.TOC['cols'][ff]][data.three_sigma_edit,:].T.dot(r_notide_scaled**2)\
                                        .reshape(grids[ff].shape)/m[ff].count)})

def smooth_xytb_fit(**kwargs):
    required_fields=('data','W','ctr','spacing','E_RMS')
    args={'reference_epoch':0,
    'W_ctr':1e4,
    'mask_file':None,
    'mask_data':None,
    'mask_scale':None,
    'compute_E':False,
    'max_iterations':10,
    'srs_proj4': None,
    'N_subset': None,
    'bias_params': None,
    'bias_filter':None,
    'repeat_res':None,
    'converge_tol_dz':0.05,
    'DEM_tol':None,
    'repeat_dt': 1,
    'Edit_only': False,
    'dzdt_lags':[1, 4],
    'avg_scales':[],
    'data_slope_sensors':None,
    'E_slope':0.05,
    'E_RMS_d2x_PS_bias':None,
    'E_RMS_PS_bias':None,
    'error_res_scale':None,
    'avg_masks':None,
    'bias_nsigma_edit':None,
    'bias_nsigma_iteration':2,
    'VERBOSE': True}
    args.update(kwargs)
    for field in required_fields:
        if field not in kwargs:
            raise ValueError("%s must be defined", field)
    valid_data = np.isfinite(args['data'].z) #np.ones_like(args['data'].x, dtype=bool)
    timing=dict()

    if args['N_subset'] is not None:
        tic=time()
        valid_data &= edit_data_by_subset_fit(args['N_subset'], args)
        timing['edit_by_subset']=time()-tic
        if args['Edit_only']:
            return {'timing':timing, 'data':args['data'].copy()[valid_data]}
    m={}
    E={}
    R={}
    RMS={}
    tic=time()
    # define the grids
    grids, bds = setup_grids(args)

    #print("\nstarting smooth_xytb_fit")
    #summarize_time(args['data'], grids['dz'].ctrs[2], np.ones(args['data'].shape, dtype=bool))

    # select only the data points that are within the grid bounds
    valid_z0=grids['z0'].validate_pts((args['data'].coords()[0:2]))
    valid_dz=grids['dz'].validate_pts((args['data'].coords()))
    valid_data=valid_data & valid_dz & valid_z0

    if not np.any(valid_data):
        if args['VERBOSE']:
            print("smooth_xytb_fit: no valid data")
        return {'m':m, 'E':E, 'data':None, 'grids':grids, 'valid_data': valid_data, 'TOC':{},'R':{}, 'RMS':{}, 'timing':timing,'E_RMS':args['E_RMS']}

    # subset the data based on the valid mask
    data=args['data'].copy_subset(valid_data)

    #print("\n\nafter validation")
    #summarize_time(data, grids['dz'].ctrs[2], np.ones(data.shape, dtype=bool))

    # if we have a mask file, use it to subset the data
    # needs to be done after the valid subset because otherwise the interp_mtx for the mask file fails.
    if args['mask_file'] is not None or args['mask_data'] is not None:
        setup_mask(data, grids, valid_data, bds, args)

    # Check if we have any data.  If not, quit
    if data.size==0:
        if args['VERBOSE']:
            print("smooth_xytb_fit: no valid data")
        return {'m':m, 'E':E, 'data':data, 'grids':grids, 'valid_data': valid_data, 'TOC':{},'R':{}, 'RMS':{}, 'timing':timing,'E_RMS':args['E_RMS']}

    # define the interpolation operator, equal to the sum of the dz and z0 operators
    G_data=lin_op(grids['z0'], name='interp_z').interp_mtx(data.coords()[0:2])
    G_data.add(lin_op(grids['dz'], name='interp_dz').interp_mtx(data.coords()))

    # define the smoothness constraints
    constraint_op_list=[]
    setup_smoothness_constraints(grids, constraint_op_list, args['E_RMS'], args['mask_scale'])

    if args['E_RMS_d2x_PS_bias'] is not None:
        setup_PS_bias(data, G_data, constraint_op_list, grids, bds, args)

    # if bias params are given, create a set of parameters to estimate them
    if args['bias_params'] is not None:
        data, bias_model = assign_bias_ID(data, args['bias_params'], \
                                          bias_filter=args['bias_filter'])
        setup_bias_fit(data, bias_model, G_data, constraint_op_list,
                       bias_param_name='bias_ID')
    else:
        bias_model={}
    if args['data_slope_sensors'] is not None and len(args['data_slope_sensors'])>0:
        #N.B.  This does not currently work.
        bias_model['E_slope']=args['E_slope']
        G_slope_bias, Gc_slope_bias, Cvals_slope_bias, bias_model = data_slope_bias(data,  bias_model, sensors=args['data_slope_sensors'],  col_0=G_data.col_N)
        G_data.add(G_slope_bias)
        constraint_op_list.append(Gc_slope_bias)
    # put the equations together
    Gc=lin_op(None, name='constraints').vstack(constraint_op_list)

    N_eq=G_data.N_eq+Gc.N_eq

    # put together all the errors
    Ec=np.zeros(Gc.N_eq)
    for op in constraint_op_list:
        try:
            Ec[Gc.TOC['rows'][op.name]]=op.expected
        except ValueError as E:
            print("smooth_xytb_fit:\n\t\tproblem with "+op.name)
            raise(E)
    if args['data_slope_sensors'] is not None and len(args['data_slope_sensors']) > 0:
        Ec[Gc.TOC['rows'][Gc_slope_bias.name]] = Cvals_slope_bias
    Ed=data.sigma.ravel()
    if np.any(Ed==0):
        raise(ValueError('zero value found in data sigma'))
    if np.any(Ec==0):
        raise(ValueError('zero value found in constraint sigma'))
    #print({op.name:[Ec[Gc.TOC['rows'][op.name]].min(),  Ec[Gc.TOC['rows'][op.name]].max()] for op in constraint_op_list})
    # calculate the inverse square root of the data covariance matrix
    TCinv=sp.dia_matrix((1./np.concatenate((Ed, Ec)), 0), shape=(N_eq, N_eq))

    # define the right hand side of the equation
    rhs=np.zeros([N_eq])
    rhs[0:data.size]=data.z.ravel()

    # put the fit and constraint matrices together
    Gcoo=sp.vstack([G_data.toCSR(), Gc.toCSR()]).tocoo()

    # setup operators that take averages of the grid at different scales
    averaging_ops = setup_averaging_ops(grids['dz'], G_data.col_N, args, cell_area=grids['dz'].cell_area)

    # setup masked averaging ops
    averaging_ops.update(setup_avg_mask_ops(grids['dz'], G_data.col_N, args['avg_masks'], args['dzdt_lags']))

    # define the matrix that sets dz[reference_epoch]=0 by removing columns from the solution:
    Ip_c = build_reference_epoch_matrix(G_data, Gc, grids, args['reference_epoch'])

    # eliminate the columns for the model variables that are set to zero
    Gcoo=Gcoo.dot(Ip_c)
    timing['setup']=time()-tic

    # initialize the book-keeping matrices for the inversion
    if "three_sigma_edit" in data.fields:
        in_TSE=np.flatnonzero(data.three_sigma_edit)
    else:
        in_TSE=np.arange(G_data.N_eq, dtype=int)
    in_TSE_last = np.zeros([0])
    if args['VERBOSE']:
        print("initial: %d:" % G_data.r.max(), flush=True)

    # if we've done any iterations, parse the model and the data residuals
    if args['max_iterations'] > 0:
        tic_iteration=time()
        m0, sigma_hat, in_TSE, in_TSE_last, rs_data=iterate_fit(data, Gcoo, rhs, \
                                TCinv, G_data, Gc, in_TSE, Ip_c, timing, args, \
                                    bias_model=bias_model)

        timing['iteration']=time()-tic_iteration
        in_TSE=in_TSE_last
        valid_data[valid_data]=(np.abs(rs_data)<3.0*np.maximum(1, sigma_hat))
        data.assign({'three_sigma_edit':np.abs(rs_data)<3.0*np.maximum(1, sigma_hat)})
        # report the model-based estimate of the data points
        data.assign({'z_est':np.reshape(G_data.toCSR().dot(m0), data.shape)})
        parse_model(m, m0, data, R, RMS, G_data, averaging_ops, Gc, Ec, grids, bias_model, args)
        R['data']=np.sum((((data.z_est[data.three_sigma_edit==1]-data.z[data.three_sigma_edit==1])/data.sigma[data.three_sigma_edit==1])**2))
        RMS['data']=np.sqrt(np.mean((data.z_est[data.three_sigma_edit==1]-data.z[data.three_sigma_edit==1])**2))

    # Compute the error in the solution if requested
    if args['compute_E']:
        # We have generally not done any iterations at this point, so need to make the Ip_r matrix
        cov_rows=G_data.N_eq+np.arange(Gc.N_eq)
        Ip_r=sp.coo_matrix((np.ones(Gc.N_eq+in_TSE.size), (np.arange(Gc.N_eq+in_TSE.size), np.concatenate((in_TSE, cov_rows)))), \
                           shape=(Gc.N_eq+in_TSE.size, Gcoo.shape[0])).tocsc()
        if args['VERBOSE']:
            print("Starting uncertainty calculation", flush=True)
            tic_error=time()
        # recalculate the error scaling from the misfits
        rs=(data.z_est-data.z)/data.sigma
        error_scale=RDE(rs[data.three_sigma_edit==1])
        print(f"scaling uncertainties by {error_scale}")
        calc_and_parse_errors(E, Gcoo, TCinv, rhs, Ip_c, Ip_r, grids, G_data, Gc, averaging_ops, \
                         bias_model, args['bias_params'], dzdt_lags=args['dzdt_lags'], timing=timing, \
                             error_scale=error_scale)
        if args['VERBOSE']:
            print("\tUncertainty propagation took %3.2f seconds" % (time()-tic_error), flush=True)

    TOC=Gc.TOC
    return {'m':m, 'E':E, 'data':data, 'grids':grids, 'valid_data': valid_data, \
            'TOC':TOC,'R':R, 'RMS':RMS, 'timing':timing,'E_RMS':args['E_RMS'], \
                'dzdt_lags':args['dzdt_lags']}

def main():
    W={'x':1.e4,'y':200,'t':2}
    x=np.arange(-W['x']/2, W['x']/2, 100)
    D=pc.data().from_dict({'x':x, 'y':np.zeros_like(x),'z':np.sin(2*np.pi*x/2000.),\
                           'time':np.zeros_like(x)-0.5, 'sigma':np.zeros_like(x)+0.1})
    D1=D
    D1.t=np.ones_like(x)
    data=pc.data().from_list([D, D.copy().assign({'time':np.zeros_like(x)+0.5})])
    E_d3zdx2dt=0.0001
    E_d2z0dx2=0.006
    E_d2zdt2=5
    E_RMS={'d2z0_dx2':E_d2z0dx2, 'dz0_dx':E_d2z0dx2*1000, 'd3z_dx2dt':E_d3zdx2dt, 'd2z_dxdt':E_d3zdx2dt*1000,  'd2z_dt2':E_d2zdt2}

    ctr={'x':0., 'y':0., 't':0.}
    SRS_proj4='+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '
    spacing={'z0':50, 'dz':50, 'dt':0.25}

    S=smooth_xytb_fit(data=data, ctr=ctr, W=W, spacing=spacing, E_RMS=E_RMS,
                     reference_epoch=2, N_subset=None, compute_E=False,
                     max_iterations=2,
                     srs_proj4=SRS_proj4, VERBOSE=True, dzdt_lags=[1])
    return S


def summarize_time(data, t0, ind):
    if 't' in data.fields:
        t=data.t

    if 'time' in data.fields:
        t=data.time
    for ti in range(len(t0)-1):
        N=np.sum((t[ind]>t0[ti]) & (t[ind]<t0[ti+1]))
        print(f"{t0[ti]} to {t0[ti+1]}: {N}")

if __name__=='__main__':
    main()
