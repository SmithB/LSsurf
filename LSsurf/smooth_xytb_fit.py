# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:27:38 2017

@author: ben
"""
import numpy as np
import re
from LSsurf.lin_op import lin_op
import scipy.sparse as sp
from LSsurf.data_slope_bias import data_slope_bias
import copy
from LSsurf.fd_grid import fd_grid
from LSsurf.lin_op import lin_op
import sparseqr
from time import time, ctime, sleep
from LSsurf.RDE import RDE
import pointCollection as pc
from LSsurf.inv_tr_upper import inv_tr_upper
from LSsurf.bias_functions import assign_bias_ID, setup_bias_fit, parse_biases
from LSsurf.grid_functions import setup_grids, \
                                    setup_averaging_ops, setup_avg_mask_ops,\
                                    setup_mask
from LSsurf.setup_grid_bias import setup_grid_bias
from LSsurf.constraint_functions import setup_smoothness_constraints, \
                                        build_reference_epoch_matrix

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

def check_data_against_DEM(in_TSE, data, m0, G_data, DEM_tol):
    m1 = m0.copy()
    m1[G_data.TOC['cols']['z0']]=0
    r_DEM=data.z - G_data.toCSR().dot(m1) - data.DEM
    return in_TSE[np.abs(r_DEM[in_TSE]-np.nanmedian(r_DEM[in_TSE]))<DEM_tol]

def iterate_fit(data, Gcoo, rhs, TCinv, G_data, Gc, in_TSE, Ip_c, timing, args,\
                bias_model=None):
    cov_rows=G_data.N_eq+np.arange(Gc.N_eq)

    # save the original state of the in_TSE variable so that we can force the non-editable
    # TSE values to remain in their original state
    in_TSE_original=np.zeros(data.shape, dtype=bool)
    in_TSE_original[in_TSE]=True

    min_tse_iterations=2
    if args['bias_nsigma_iteration'] is not None:
        min_tse_iterations=np.max([min_tse_iterations, args['bias_nsigma_iteration']+1])

    if 'editable' in data.fields:
        N_editable=np.sum(data.editable)
    else:
        N_editable=data.size

    #initialize m0, so that we have a value for a previous iteration
    m0 = np.zeros(Ip_c.shape[0])

    for iteration in range(args['max_iterations']):
        # build the parsing matrix that removes invalid rows
        Ip_r=sp.coo_matrix((np.ones(Gc.N_eq+in_TSE.size), \
                            (np.arange(Gc.N_eq+in_TSE.size), \
                             np.concatenate((in_TSE, cov_rows)))), \
                           shape=(Gc.N_eq+in_TSE.size, Gcoo.shape[0])).tocsc()
        m0_last=m0
        if args['VERBOSE']:
            print("starting qr solve for iteration %d at %s" % (iteration, ctime()), flush=True)
        # solve the equations
        solved=False
        while not solved:
            try:
                tic=time();
                m0=Ip_c.dot(sparseqr.solve(Ip_r.dot(TCinv.dot(Gcoo)), Ip_r.dot(TCinv.dot(rhs))));
                timing['sparseqr_solve']=time()-tic
                solved=True
            except TypeError:
                print("smooth_xytb_fit: spareqr.solve failed, probably because of memory.  Retrying after 5 minutes")
                sleep(300)

        # calculate the full data residual
        rs_data=(data.z-G_data.toCSR().dot(m0))/data.sigma
        # calculate the robust standard deviation of the scaled residuals for the selected data
        sigma_hat=RDE(rs_data[in_TSE])

        # select the data that have scaled residuals < 3 *max(1, sigma_hat)
        in_TSE_last=in_TSE
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
                #mask=np.ones(data.size, dtype=bool)
                #Mark the ID as edited (because it will have a bias estimate of zero in subsequent iterations)
                bias_model['bias_param_dict']['edited'][bias_model['bias_param_dict']['ID'].index(ID)]=True
            in_TSE[np.in1d(data.bias_ID, bad_bias_IDs)]=False
                # mark all data associated with the ID as invalid
                #for field, field_val in bias_model['bias_ID_dict'][ID].items():
                #    if field in data.fields:
                #        mask &= (getattr(data, field).ravel()==field_val)
                #in_TSE[mask==1]=0
        if 'editable' in data.fields:
            in_TSE[data.editable==0] = in_TSE_original[data.editable==0]
            N_editable=np.sum(data.editable==1).astype(float)
        else:
            N_editable=data.x.size
        in_TSE = np.flatnonzero(in_TSE)

        if args['DEM_tol'] is not None:
            in_TSE = check_data_against_DEM(in_TSE, data, m0, G_data, args['DEM_tol'])
        
        max_model_change_dz=np.max(np.abs((m0_last-m0)[Gc.TOC['cols']['dz']]))

        if args['VERBOSE']:
            print('found %d in TSE, sigma_hat=%3.3f, dm_max=%3.3f, dt=%3.0f' % ( in_TSE.size, sigma_hat, max_model_change_dz, timing['sparseqr_solve']), flush=True)
        # quit if the solution is too similar to the previous solution
        if (iteration > np.maximum(2, args['bias_nsigma_iteration'])):
            if (max_model_change_dz < args['converge_tol_dz']) and (iteration > 2):
                if args['VERBOSE']:
                    print("Solution identical to previous iteration with tolerance %3.2f, exiting after iteration %d" % (args['converge_tol_dz'], iteration))
                break

        if iteration > 0:
            # Calculate the number of elements that have changed in in_TSE
            frac_TSE_change = len(np.setxor1d(in_TSE_last, in_TSE))/N_editable
            if frac_TSE_change < args['converge_tol_frac_TSE']:
                if args['VERBOSE']:
                    print("filtering unchanged with tolerance %3.5f, exiting after iteration %d" 
                          % (args['converge_tol_frac_TSE'], iteration))
                break
        if iteration >= min_tse_iterations:
            if sigma_hat <= 1:
                if args['VERBOSE']:
                    print("sigma_hat LT 1, exiting after iteration %d" % iteration, flush=True)
                break
        m0_last=m0
    return m0, sigma_hat, in_TSE, in_TSE_last, rs_data

def parse_biases(m, bias_model, bias_params):
    """
        parse the biases in the ouput model

        inputs:
            m: model vector
            bias_model: the bias model
            bias_params: a list of parameters for which biases are calculated
        output:
            b__dict: a dictionary giving the parameters and associated bias values for each ibas ID
    """
    slope_bias_dict={}
    b_dict={param:list() for param in bias_params+['val','ID','expected']}
    # loop over the keys in bias_model['bias_ID_dict']
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
    tic=time(); RR, CC, VV, status=inv_tr_upper(R, int(np.prod(R.shape)/4), 1.e-5);
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

    if args['grid_bias_model_args'] is not None:
        m['grid_biases']={}
        for temp in args['grid_bias_model_args']:
            this_grid=temp['grid']
            m['grid_biases'][this_grid.name]=pc.grid.data().from_dict({
                'x': this_grid.ctrs[1],
                'y': this_grid.ctrs[0], 
                this_grid.name:np.reshape(m0[this_grid.col_0:this_grid.col_N], this_grid.shape)})

    #if 'PS_bias' in G_data.TOC['cols']:
    #    m['dz'].assign({'PS_bias':np.reshape(m0[G_data.TOC['cols']['PS_bias']], grids['dz'].shape[0:2])})

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

    # calculate the data residuals
    R['data']=np.sum((((data.z_est[data.three_sigma_edit==1]-data.z[data.three_sigma_edit==1])/data.sigma[data.three_sigma_edit==1])**2))
    RMS['data']=np.sqrt(np.mean((data.z_est[data.three_sigma_edit==1]-data.z[data.three_sigma_edit==1])**2))

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
        # convert the NaNs in the count and misfit* fields back to zeros
        for field in ['count', 'misfit_scaled_rms', 'misfit_rms', \
                      'misfit_notide_rms', 'misfit_notide_scaled_rms']:
            if field in m[ff].fields:
                setattr(m[ff], field, np.nan_to_num(getattr(m[ff], field)))

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
    'converge_tol_frac_TSE':0.,
    'DEM_tol':None,
    'repeat_dt': 1,
    'Edit_only': False,
    'dzdt_lags':None,
    'avg_scales':[],
    'data_slope_sensors':None,
    'E_slope':0.05,
    'E_RMS_d2x_PS_bias':None,
    'E_RMS_PS_bias':None,
    'error_res_scale':None,
    'avg_masks':None,
    'grid_bias_model_args':None,
    'bias_nsigma_edit':None,
    'bias_nsigma_iteration':2,
    'bias_edit_vals':None,
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
    
    # setup the smooth biases
    if args['grid_bias_model_args'] is not None:
        for bm_args in args['grid_bias_model_args']:
            setup_grid_bias(data, G_data, constraint_op_list, grids, **bm_args)
    
    #if args['E_RMS_d2x_PS_bias'] is not None:
    #    setup_PS_bias(data, G_data, constraint_op_list, grids, bds, args)

    # if bias params are given, create a set of parameters to estimate them
    if args['bias_params'] is not None:
        data, bias_model = assign_bias_ID(data, args['bias_params'], \
                                          bias_filter=args['bias_filter'])
        setup_bias_fit(data, bias_model, G_data, constraint_op_list,
                       bias_param_name='bias_ID')
        if args['bias_nsigma_edit']:
            bias_model['bias_param_dict']['edited']=np.zeros_like(bias_model['bias_param_dict']['ID'], dtype=bool)

        if args['bias_edit_vals'] is not None:
            edit_bias_list=np.c_[[args['bias_edit_vals'][key] for key in args['bias_edit_vals'].keys()]].T.tolist()
            bias_list=np.c_[[bias_model['bias_param_dict'][key] for key in args['bias_edit_vals'].keys()]].T.tolist()
            for row in edit_bias_list:
                bias_model['bias_param_dict']['edited'][bias_list.index(row)]=True
            # apply the editing to the three_sigma_edit variable
            bad_IDs=[bias_model['bias_param_dict']['ID'][ii] 
                     for ii in np.flatnonzero(bias_model['bias_param_dict']['edited'])]
            data.three_sigma_edit[np.in1d(data.bias_ID, bad_IDs)]=False
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
        # in_TSE_last is the list of points in the last inversion step
        # this needs to be converted from a list of data to a boolean array
        in_TSE=in_TSE_last
        data.assign({'three_sigma_edit':np.zeros_like(data.x, dtype=bool)})
        data.three_sigma_edit[in_TSE]=1
        # copy this into the valid_data array
        valid_data[valid_data]=data.three_sigma_edit

        # report the model-based estimate of the data points
        data.assign({'z_est':np.reshape(G_data.toCSR().dot(m0), data.shape)})
        # reshapethe model vector into the grid outputs
        parse_model(m, m0, data, R, RMS, G_data, averaging_ops, Gc, Ec, grids, bias_model, args)

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
