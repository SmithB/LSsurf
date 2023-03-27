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
from .data_slope_bias import data_slope_bias
import copy
import sparseqr
from time import time
from LSsurf.RDE import RDE
from LSsurf.unique_by_rows import unique_by_rows
import os
import h5py
import json
#import scipy.sparse.linalg as spl
#from spsolve_tr_upper import spsolve_tr_upper
#from propagate_qz_errors import propagate_qz_errors
from LSsurf.inv_tr_upper import inv_tr_upper

def edit_data_by_subset_fit(N_subset, args):

    W_scale=2./N_subset
    W_subset={'x':args['W']['x']*W_scale, 'y':args['W']['y']*W_scale}
    subset_spacing={key:W_subset[key]/2 for key in list(W_subset)}
    bds={coord:args['ctr'][coord]+np.array([-0.5, 0.5])*args['W'][coord] for coord in ('x','y','t')}

    subset_ctrs=np.meshgrid(np.arange(bds['x'][0]+subset_spacing['x'], bds['x'][1], subset_spacing['x']),  np.arange(bds['y'][0]+subset_spacing['y'], bds['y'][1], subset_spacing['y']))
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
        #tic=time()
        #if args['VERBOSE']:
        #    print("working on subset %d, XR=[%d, %d], YR=[%d, %d], f_tot=%2.2f" % (count, x0-W_subset['x']/2, x0+W_subset['x']/2, y0-W_subset['x']/2, y0+W_subset['x']/2, np.mean(in_bounds)))

        sub_fit=smooth_xyt_fit(**sub_args)
        #t_fit=time()-tic
        #if args['VERBOSE']:
        #    print("dt=%3.2f, t expected for all=%3.2f"  % (t_fit, t_fit*subset_ctrs[0].size))
        in_tight_bounds_sub = \
            (sub_args['data'].x > x0-W_subset['x']/4) & (sub_args['data'].x < x0+W_subset['x']/4) & \
            (sub_args['data'].y > y0-W_subset['y']/4) & (sub_args['data'].y < y0+W_subset['y']/4)
        in_tight_bounds_all=\
            (args['data'].x > x0-W_subset['x']/4) & ( args['data'].x < x0+W_subset['x']/4) & \
            (args['data'].y > y0-W_subset['y']/4) & ( args['data'].y < y0+W_subset['x']/4)
        valid_data[in_tight_bounds_all] = valid_data[in_tight_bounds_all] & sub_fit['valid_data'][in_tight_bounds_sub]
    if args['VERBOSE']:
        print("from all subsets, found %d data" % valid_data.sum())
    return valid_data

def select_repeat_data(data, grids, repeat_dt, resolution, reference_time=None):
    """
        Select data that are repeats

        input arguments:
            data: input data
            grids: grids
            repeat_dt: time interval by which repeats must be separated to count
            resolution: spatial resolution of repeat calculation
    """
    repeat_grid=fd_grid( grids['z0'].bds, resolution*np.ones(2), name='repeat')
    t_coarse=np.round((data.time-grids['dz'].bds[2][0])/repeat_dt)*repeat_dt
    grid_repeat_count=np.zeros(np.prod(repeat_grid.shape))
    for t_val in np.unique(t_coarse):
        # select the data points for each epoch
        ii=t_coarse==t_val
        # use the lin_op.interp_mtx to find the grid points associated with each node
        grid_repeat_count += np.asarray(lin_op(repeat_grid).interp_mtx((data.y[ii], data.x[ii])).toCSR().sum(axis=0)>0.5).ravel()
    data_repeats = lin_op(repeat_grid).interp_mtx((data.y, data.x)).toCSR().dot((grid_repeat_count>1).astype(np.float64))
    if reference_time is None:
        return data_repeats>0.5
    else:
        return (data_repeats > 0.5) | (np.abs(data.time-reference_time ) < repeat_dt)

def assign_bias_ID(data, bias_params=None, bias_name='bias_ID', key_name=None, bias_model=None):
    """
    Assign a value to each data point that determines which biases are applied to it.

    parameters:
        data: pointCollection.data instance
        bias_parameters: a list of parameters, each unique combination of which defines a different bias
        bias_name: a name for the biases
        key_name: an optional parameter which will be used as the dataset name, otherwise a key will be built from the parameter values
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
        temp=np.column_stack([getattr(data, bp) for bp in bias_params])
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

def param_bias_matrix(data,  bias_model, col_0=0, bias_param_name='data_bias', op_name='data_bias'):
    """
        Make a matrix that adds a set of parameters representing the biases of a set of data.

        input arguments:
             data: data for the problem.  Must containa parameter with the name specified in 'bias_param_name
             bias_model: bias_model dict from assign_bias_params
             col_0: the first column of the matrix.
             bias_param_name: name of the parameter used to assign the biases.  Defaults to 'data_bias'
             op_name: the name for the output bias operator.
         output_arguments:
             G_bias: matrix that gives the biases for each parameter
             Gc_bias: matrix that gives the bias values (constraint matrix)
             E_bias: expected value for each bias parameter
             bias_model: bias model dict as defined in assign_bias_ID
    """
    col=getattr(data, bias_param_name).astype(int)
    col_N=col_0+np.max(col)+1
    G_bias=lin_op(name=op_name, col_0=col_0, col_N=col_N).data_bias(np.arange(data.size), col=col_0+col)
    ii=np.arange(col.max()+1, dtype=int)
    Gc_bias=lin_op(name='contstraint_'+op_name, col_0=col_0, col_N=col_N).data_bias(ii,col=col_0+ii)
    E_bias=[bias_model['E_bias'][ind] for ind in ii]
    for key in bias_model['bias_ID_dict']:
        bias_model['bias_ID_dict'][key]['col']=col_0+key
    return G_bias, Gc_bias, E_bias, bias_model

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
    b_dict={param:list() for param in bias_params+['val']}
    for item in bias_model['bias_ID_dict']:
        b_dict['val'].append(m[bias_model['bias_ID_dict'][item]['col']])
        for param in bias_params:
            b_dict[param].append(bias_model['bias_ID_dict'][item][param])
    if 'slope_bias_dict' in bias_model:

        for key in bias_model['slope_bias_dict']:
            slope_bias_dict[key]={'slope_x':m[bias_model['slope_bias_dict'][key][0]], 'slope_y':m[bias_model['slope_bias_dict'][key][1]]}
    return b_dict, slope_bias_dict

def parse_errors(E, Gcoo, TCinv, rhs, Ip_c, Ip_r, grids, G_data, Gc, G_dzbar, bias_model, bias_params, dzdt_lags=None, timing={}):
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
    tic=time(); RR, CC, VV, status=inv_tr_upper(R, int(np.prod(R.shape)/4), 1.e-5);
    # save Rinv as a sparse array.  The syntax perm[RR] undoes the permutation from QZ
    Rinv=sp.coo_matrix((VV, (perm[RR], CC)), shape=R.shape).tocsr(); timing['Rinv_cython']=time()-tic;
    tic=time(); E0=np.sqrt(Rinv.power(2).sum(axis=1)); timing['propagate_errors']=time()-tic;
    
    # generate the full E vector.  E0 appears to be an ndarray,
    E0=np.array(Ip_c.dot(E0)).ravel()
    E['z0']=np.reshape(E0[Gc.TOC['cols']['z0']], grids['z0'].shape)
    E['dz']=np.reshape(E0[Gc.TOC['cols']['dz']], grids['dz'].shape)

    # generate the lagged dz errors:

    for lag in dzdt_lags:
        this_name='dzdt_lag%d' % lag
        E[this_name]=lin_op(grids['dz'], name=this_name, col_N=G_data.col_N).dzdt(lag=lag).grid_error(Ip_c.dot(Rinv))
        # note: this should probably be dotted with the G_dzbar op.  lag op is nlag*nt, G_dzbar is nt*ncols, R is NcolsxNcols RUN ONE LINE AT A TIME
        this_name='dzdt_bar_lag%d' % lag
        this_op=lin_op(grids['t'], name=this_name).diff(lag=lag).toCSR().dot(G_dzbar)
        E[this_name]=np.sqrt((this_op.dot(Ip_c).dot(Rinv)).power(2).sum(axis=1))
         
    # generate the grid-mean error for zero lag  
    E['dz_bar']=np.sqrt((G_dzbar.dot(Ip_c).dot(Rinv)).power(2).sum(axis=1))
    
    E['bias'], E['slope_bias']=parse_biases(E0, bias_model, bias_params)

def parse_model(m, m0, G_data, G_dzbar, TOC, grids, bias_params, bias_model, dzdt_lags=None):

    # reshape the components of m to the grid shapes
    m['z0']=np.reshape(m0[TOC['cols']['z0']], grids['z0'].shape)
    m['dz']=np.reshape(m0[TOC['cols']['dz']], grids['dz'].shape)
    m['count']=np.reshape(np.array(G_data.toCSR().tocsc()[:,TOC['cols']['dz']].sum(axis=0)), grids['dz'].shape)
    
    # calculate height rates
    for lag in dzdt_lags:
        this_name='dzdt_lag%d' % lag
        m[this_name]=lin_op(grids['dz'], name='dzdt', col_N=G_data.col_N).dzdt(lag=lag).grid_prod(m0)
    
    # calculate the grid mean of dz
    m['dz_bar']=G_dzbar.dot(m0)

    # build a matrix that takes the lagged temporal derivative of dzbar (e.g. quarterly dzdt, annual dzdt)
    for lag in dzdt_lags:
        this_name='dzdt_bar_lag%d' % lag
        this_op=lin_op(grids['t'], name=this_name).diff(lag=lag).toCSR()
        # calculate the grid mean of dz/dt
        m[this_name]=this_op.dot(m['dz_bar'].ravel())

    # report the parameter biases.  Sorted in order of the parameter bias arguments
    m['bias'], m['slope_bias']=parse_biases(m0, bias_model, bias_params)

    # report the entire model vector, just in case we want it.
    m['all']=m0

    # report the geolocation of the output map
    m['extent']=np.concatenate((grids['z0'].bds[1], grids['z0'].bds[0]))




def smooth_xyt_fit(**kwargs):
    required_fields=('data','W','ctr','spacing','E_RMS')
    args={'reference_epoch':0,
    'W_ctr':1e4,
    'mask_file':None,
    'mask_scale':None,
    'compute_E':False,
    'max_iterations':10,
    'srs_proj4': None,
    'N_subset': None,
    'bias_params': None,
    'repeat_res':None,
    'converge_tol_dz':0.05,
    'repeat_dt': 1,
    'Edit_only': False,
    'dzdt_lags':[1, 4],
    'data_slope_sensors':None,
    'E_slope':0.05,
    'VERBOSE': True}
    args.update(kwargs)
    for field in required_fields:
        if field not in kwargs:
            raise ValueError("%s must be defined", field)
    valid_data = np.isfinite(args['data'].z) & np.isfinite(args['data'].sigma)
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

    # define the grids
    tic=time()
    bds={coord:args['ctr'][coord]+np.array([-0.5, 0.5])*args['W'][coord] for coord in ('x','y','t')}
    grids=dict()
    grids['z0']=fd_grid( [bds['y'], bds['x']], args['spacing']['z0']*np.ones(2),\
         name='z0', srs_proj4=args['srs_proj4'], mask_file=args['mask_file'])
    grids['dz']=fd_grid( [bds['y'], bds['x'], bds['t']], \
        [args['spacing']['dz'], args['spacing']['dz'], args['spacing']['dt']], \
         name='dz', col_0=grids['z0'].N_nodes, srs_proj4=args['srs_proj4'], \
        mask_file=args['mask_file'])
    grids['z0'].col_N=grids['dz'].col_N
    grids['t']=fd_grid([bds['t']], [args['spacing']['dt']], name='t')

    # select only the data points that are within the grid bounds
    valid_z0=grids['z0'].validate_pts((args['data'].coords()[0:2]))
    valid_dz=grids['dz'].validate_pts((args['data'].coords()))
    valid_data=valid_data & valid_dz & valid_z0
    
    if not np.any(valid_data):
        return {'m':m, 'E':E, 'data':None, 'grids':grids, 'valid_data': valid_data, 'TOC':{},'R':{}, 'RMS':{}, 'timing':timing,'E_RMS':args['E_RMS']}

    # if repeat_res is given, resample the data to include only repeat data (to within a spatial tolerance of repeat_res)
    if args['repeat_res'] is not None:
        N_before_repeat=np.sum(valid_data)   
        valid_data[valid_data]=valid_data[valid_data] & \
            select_repeat_data(args['data'].copy_subset(valid_data), grids, args['repeat_dt'], args['repeat_res'], reference_time=grids['t'].ctrs[0][args['reference_epoch']])
        if args['VERBOSE']:
            print("before repeat editing found %d data" % N_before_repeat)
            print("after repeat editing found %d data" % valid_data.sum())

    # subset the data based on the valid mask
    data=args['data'].copy_subset(valid_data)

    # if we have a mask file, use it to subset the data
    # needs to be done after the valid subset because otherwise the interp_mtx for the mask file fails.
    if args['mask_file'] is not None:
        temp=fd_grid( [bds['y'], bds['x']], [args['spacing']['z0'], args['spacing']['z0']], name='z0', srs_proj4=args['srs_proj4'], mask_file=args['mask_file'])
        data_mask=lin_op(temp, name='interp_z').interp_mtx(data.coords()[0:2]).toCSR().dot(grids['z0'].mask.ravel())
        data_mask[~np.isfinite(data_mask)]=0
        if np.any(data_mask==0):
            data.index(~(data_mask==0))
            valid_data[valid_data]= ~(data_mask==0)

    # Check if we have any data.  If not, quit
    if data.size==0:
        return {'m':m, 'E':E, 'data':data, 'grids':grids, 'valid_data': valid_data, 'TOC':{},'R':{}, 'RMS':{}, 'timing':timing,'E_RMS':args['E_RMS']}

    # define the interpolation operator, equal to the sum of the dz and z0 operators
    G_data=lin_op(grids['z0'], name='interp_z').interp_mtx(data.coords()[0:2])
    G_data.add(lin_op(grids['dz'], name='interp_dz').interp_mtx(data.coords()))

     # define the smoothness constraints
    grad2_z0=lin_op(grids['z0'], name='grad2_z0').grad2(DOF='z0')
    grad2_dz=lin_op(grids['dz'], name='grad2_dzdt').grad2_dzdt(DOF='z', t_lag=1)
    grad_dzdt=lin_op(grids['dz'], name='grad_dzdt').grad_dzdt(DOF='z', t_lag=1)
    constraint_op_list=[grad2_z0, grad2_dz, grad_dzdt]
    if 'd2z_dt2' in args['E_RMS'] and args['E_RMS']['d2z_dt2'] is not None:
        d2z_dt2=lin_op(grids['dz'], name='d2z_dt2').d2z_dt2(DOF='z')
        constraint_op_list.append(d2z_dt2)

    # if bias params are given, create a set of parameters to estimate them
    if args['bias_params'] is not None:
        data, bias_model=assign_bias_ID(data, args['bias_params'])
        G_bias, Gc_bias, Cvals_bias, bias_model=\
            param_bias_matrix(data, bias_model, bias_param_name='bias_ID', 
                              col_0=grids['dz'].col_N)
        G_data.add(G_bias)
        constraint_op_list.append(Gc_bias)

    if args['data_slope_sensors'] is not None:
        bias_model['E_slope']=args['E_slope']
        G_slope_bias, Gc_slope_bias, Cvals_slope_bias, bias_model= data_slope_bias(data,  bias_model, sensors=args['data_slope_sensors'],  col_0=G_data.col_N)
        G_data.add(G_slope_bias)
        constraint_op_list.append(Gc_slope_bias)
    # put the equations together
    Gc=lin_op(None, name='constraints').vstack(constraint_op_list)
    N_eq=G_data.N_eq+Gc.N_eq

    # put together all the errors
    Ec=np.zeros(Gc.N_eq)
    root_delta_V_dz=np.sqrt(np.prod(grids['dz'].delta))
    root_delta_A_z0=np.sqrt(np.prod(grids['z0'].delta))
    Ec[Gc.TOC['rows']['grad2_z0']]=args['E_RMS']['d2z0_dx2']/root_delta_A_z0*grad2_z0.mask_for_ind0(args['mask_scale'])
    Ec[Gc.TOC['rows']['grad2_dzdt']]=args['E_RMS']['d3z_dx2dt']/root_delta_V_dz*grad2_dz.mask_for_ind0(args['mask_scale'])
    Ec[Gc.TOC['rows']['grad_dzdt']]=args['E_RMS']['d2z_dxdt']/root_delta_V_dz*grad_dzdt.mask_for_ind0(args['mask_scale'])
    if 'd2z_dt2' in args['E_RMS'] and args['E_RMS']['d2z_dt2'] is not None:
        Ec[Gc.TOC['rows']['d2z_dt2']]=args['E_RMS']['d2z_dt2']/root_delta_V_dz
    if args['bias_params'] is not None:
        Ec[Gc.TOC['rows'][Gc_bias.name]] = Cvals_bias
    if args['data_slope_sensors'] is not None:
        Ec[Gc.TOC['rows'][Gc_slope_bias.name]] = Cvals_slope_bias
    Ed=data.sigma.ravel()
    # calculate the inverse square root of the data covariance matrix
    TCinv=sp.dia_matrix((1./np.concatenate((Ed, Ec)), 0), shape=(N_eq, N_eq))

    # define the right hand side of the equation
    rhs=np.zeros([N_eq])
    rhs[0:data.size]=data.z.ravel()

    # put the fit and constraint matrices together
    Gcoo=sp.vstack([G_data.toCSR(), Gc.toCSR()]).tocoo()
    cov_rows=G_data.N_eq+np.arange(Gc.N_eq)
     
    # build a matrix that takes the average of the center of the delta-z grid
    # this gets used both in the averaging and error-calculation codes
    XR=np.mean(grids['z0'].bds[0])+np.array([-1., 1.])*args['W_ctr']/2.
    YR=np.mean(grids['z0'].bds[1])+np.array([-1., 1.])*args['W_ctr']/2.
    center_dzbar=lin_op(grids['dz'], name='center_dzbar', col_N=G_data.col_N).vstack([lin_op(grids['dz']).mean_of_bounds((XR, YR, [season, season] )) for season in grids['dz'].ctrs[2]])
    G_dzbar=center_dzbar.toCSR()

    # define the matrix that sets dz[reference_epoch]=0 by removing columns from the solution:
    # Find the rows and columns that match the reference epoch
    temp_r, temp_c=np.meshgrid(np.arange(0, grids['dz'].shape[0]), np.arange(0, grids['dz'].shape[1]))
    z02_mask=grids['dz'].global_ind([temp_r.transpose().ravel(), temp_c.transpose().ravel(),\
                  args['reference_epoch']+np.zeros_like(temp_r).ravel()])

    # Identify all of the DOFs that do not include the reference epoch
    cols=np.arange(G_data.col_N, dtype='int')
    include_cols=np.setdiff1d(cols, z02_mask)
    # Generate a matrix that has diagonal elements corresponding to all DOFs except the reference epoch.
    # Multiplying this by a matrix with columns for all model parameters yeilds a matrix with no columns
    # corresponding to the reference epoch.
    Ip_c=sp.coo_matrix((np.ones_like(include_cols), (include_cols, np.arange(include_cols.size))), \
                       shape=(Gc.col_N, include_cols.size)).tocsc()

    # eliminate the columns for the model variables that are set to zero
    Gcoo=Gcoo.dot(Ip_c)
    timing['setup']=time()-tic

    # initialize the book-keeping matrices for the inversion
    m0=np.zeros(Ip_c.shape[0])
    if "three_sigma_edit" in data.fields:
        inTSE=np.flatnonzero(data.three_sigma_edit)
    else:
        inTSE=np.arange(G_data.N_eq, dtype=int)
    inTSE_last = np.zeros([0])
    if args['VERBOSE']:
        print("initial: %d:" % G_data.r.max())
    tic_iteration=time()
    for iteration in range(args['max_iterations']):
        # build the parsing matrix that removes invalid rows
        Ip_r=sp.coo_matrix((np.ones(Gc.N_eq+inTSE.size), (np.arange(Gc.N_eq+inTSE.size), np.concatenate((inTSE, cov_rows)))), \
                           shape=(Gc.N_eq+inTSE.size, Gcoo.shape[0])).tocsc()

        m0_last=m0
        if args['VERBOSE']:
            print("starting qr solve for iteration %d" % iteration)
        # solve the equations
        tic=time(); 
        m0=Ip_c.dot(sparseqr.solve(Ip_r.dot(TCinv.dot(Gcoo)), Ip_r.dot(TCinv.dot(rhs)))); 
        timing['sparseqr_solve']=time()-tic

        # calculate the full data residual
        rs_data=(data.z-G_data.toCSR().dot(m0))/data.sigma
        # calculate the robust standard deviation of the scaled residuals for the selected data
        sigma_hat=RDE(rs_data[inTSE])
        
        # select the data that have scaled residuals < 3 *max(1, sigma_hat)
        inTSE_last=inTSE
        inTSE = np.flatnonzero(np.abs(rs_data) < 3.0 * np.maximum(1, sigma_hat))
        
        # quit if the solution is too similar to the previous solution
        if (np.max(np.abs((m0_last-m0)[Gc.TOC['cols']['dz']])) < args['converge_tol_dz']) and (iteration > 2):
            if args['VERBOSE']:
                print("Solution identical to previous iteration with tolerance %3.1f, exiting after iteration %d" % (args['converge_tol_dz'], iteration))
            break
        # select the data that are within 3*sigma of the solution
        if args['VERBOSE']:
            print('found %d in TSE, sigma_hat=%3.3f' % ( inTSE.size, sigma_hat ))
        if iteration > 0:
            if inTSE.size == inTSE_last.size and np.all( inTSE_last == inTSE ):
                if args['VERBOSE']:
                    print("filtering unchanged, exiting after iteration %d" % iteration)
                break 
        if iteration >= 2:
            if sigma_hat <= 1:
                if args['VERBOSE']:
                    print("sigma_hat LT 1, exiting after iteration %d" % iteration)
                break             

    # if we've done any iterations, parse the model and the data residuals
    if args['max_iterations'] > 0:
        timing['iteration']=time()-tic_iteration
        inTSE=inTSE_last
        valid_data[valid_data]=(np.abs(rs_data)<3.0*np.maximum(1, sigma_hat))
        data.assign({'three_sigma_edit':np.abs(rs_data)<3.0*np.maximum(1, sigma_hat)})
        # report the model-based estimate of the data points
        data.assign({'z_est':np.reshape(G_data.toCSR().dot(m0), data.shape)})
        parse_model(m, m0, G_data, G_dzbar, Gc.TOC, grids, args['bias_params'], bias_model, dzdt_lags=args['dzdt_lags'])
        # parse the resduals to assess the contributions of the total error:
        # Make the C matrix for the constraints
        TCinv_cov=sp.dia_matrix((1./Ec, 0), shape=(Gc.N_eq, Gc.N_eq))
        rc=TCinv_cov.dot(Gc.toCSR().dot(m0))
        ru=Gc.toCSR().dot(m0)
        for eq_type in ['d2z_dt2','grad2_z0','grad2_dzdt']:
            if eq_type in Gc.TOC['rows']:
                R[eq_type]=np.sum(rc[Gc.TOC['rows'][eq_type]]**2)
                RMS[eq_type]=np.sqrt(np.mean(ru[Gc.TOC['rows'][eq_type]]**2))
    R['data']=np.sum((((data.z_est[data.three_sigma_edit==1]-data.z[data.three_sigma_edit==1])/data.sigma[data.three_sigma_edit==1])**2))
    RMS['data']=np.sqrt(np.mean((data.z_est[data.three_sigma_edit==1]-data.z[data.three_sigma_edit==1])**2))

    # Compute the error in the solution if requested
    if args['compute_E']:
        # We have generally not done any iterations at this point, so need to make the Ip_r matrix
        Ip_r=sp.coo_matrix((np.ones(Gc.N_eq+inTSE.size), (np.arange(Gc.N_eq+inTSE.size), np.concatenate((inTSE, cov_rows)))), \
                           shape=(Gc.N_eq+inTSE.size, Gcoo.shape[0])).tocsc()
        parse_errors(E, Gcoo, TCinv, rhs, Ip_c, Ip_r, grids, G_data, Gc, G_dzbar, \
                         bias_model, args['bias_params'], dzdt_lags=args['dzdt_lags'], timing=timing)

 

    TOC=Gc.TOC
    return {'m':m, 'E':E, 'data':data, 'grids':grids, 'valid_data': valid_data, 'TOC':TOC,'R':R, 'RMS':RMS, 'timing':timing,'E_RMS':args['E_RMS'], 'dzdt_lags':args['dzdt_lags']}



