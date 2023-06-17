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
from LSsurf.setup_sensor_grid_bias import setup_sensor_grid_bias,\
                                            parse_sensor_bias_grids
import sparseqr
from time import time, ctime
from LSsurf.RDE import RDE
import pointCollection as pc
from scipy.stats import scoreatpercentile
from LSsurf.inv_tr_upper import inv_tr_upper
from LSsurf.bias_functions import assign_bias_ID, setup_bias_fit, parse_biases
from LSsurf.grid_functions import setup_grids, \
                                    setup_averaging_ops, setup_avg_mask_ops,\
                                    validate_by_dz_mask, setup_mask
from LSsurf.constraint_functions import setup_smoothness_constraints, \
                                        build_reference_epoch_matrix
from LSsurf.match_priors import match_prior_dz,  match_tile_edges
from LSsurf.calc_sigma_extra import calc_sigma_extra, calc_sigma_extra_on_grid

def check_data_against_DEM(in_TSE, data, m0, G_data, DEM_tol):
    m1 = m0.copy()
    m1[G_data.TOC['cols']['z0']]=0
    r_DEM=data.z - G_data.toCSR().dot(m1) - data.DEM
    temp=in_TSE
    temp[in_TSE] = np.abs(r_DEM[in_TSE]) < DEM_tol
    return temp

def print_TOC(G_data, Gc):
    print(f"G_data : \n\t{len(G_data.v)/1000}K values")
    print(f"\t shape={np.array(G_data.shape)/1000}K")
    print("rows:")
    for name, rr in G_data.TOC['rows'].items():
        print(f'\t{name}: {len(np.unique(rr))/1000}K')
    print("cols:")
    N_sensor_cols=0
    for name, cc in G_data.TOC['cols'].items():
        print(f'\t{name}: {len(np.unique(cc))/1000}K')
        if 'sensor' in name:
            N_sensor_cols += len(np.unique(cc))
    print(f'\t\tSensor cols total: {N_sensor_cols/1000}K')
    print(f"\nGc \n\t{len(Gc.v)/1000}K values")
    print(f"\t shape={np.array(Gc.shape)/1000}K")
    print("rows:")
    N_sensor_rows=0
    for name, rr in Gc.TOC['rows'].items():
        if 'sensor' in name:
            N_sensor_rows += len(np.unique(rr))
        else:
            print(f'\t{name}: {len(np.unique(rr))/1000}K')
    print(f'\t\tSensor constraint total: {N_sensor_rows/1000}K')



def edit_by_bias(data, m0, in_TSE, iteration, bias_model, args):

    if args['bias_nsigma_edit'] is None:
        return False

    # assign the edited field in bias_model['bias_param_dict'] if needed
    if 'edited' not in bias_model['bias_param_dict']:
            bias_model['bias_param_dict']['edited']=np.zeros_like(bias_model['bias_param_dict']['ID'], dtype=bool)
    bias_dict, slope_bias_dict=parse_biases(m0, bias_model, args['bias_params'])
    bias_scaled = np.abs(bias_dict['val']) / np.array(bias_dict['expected'])

    last_edit = bias_model['bias_param_dict']['edited'].copy()
    bad_bias = np.zeros_like(bias_scaled, dtype=bool)
    bad_bias |= last_edit
    if iteration >= args['bias_nsigma_iteration']:
        extreme_bias_scaled_threshold = np.minimum(50, 3*scoreatpercentile(bias_scaled, 95))
        if np.any(bias_scaled > extreme_bias_scaled_threshold):
            bad_bias[ bias_scaled == np.max(bias_scaled) ] = True
        else:
            bad_bias[bias_scaled > args['bias_nsigma_edit']] = True

    bad_bias_IDs=np.array(bias_dict['ID'])[bad_bias]

    for ID in bad_bias_IDs:
        #Mark each bad ID as edited (because it will have a bias estimate of zero in subsequent iterations)
        bias_model['bias_param_dict']['edited'][bias_model['bias_param_dict']['ID'].index(ID)]=True
    if len(bad_bias_IDs)>0:
        print(f"\t have {len(bad_bias_IDs)} bad biases, with {np.sum(np.in1d(data.bias_ID, bad_bias_IDs))} data.")
    else:
        if iteration >= args['bias_nsigma_iteration']:
            print("No bad biases found")
    in_TSE[np.in1d(data.bias_ID, bad_bias_IDs)]=False

    return  ~np.all(bias_model['bias_param_dict']['edited'] == last_edit)

def iterate_fit(data, Gcoo, rhs, TCinv, G_data, Gc, in_TSE, Ip_c, timing, args,
                grids, bias_model=None, sigma_extra_masks=None):
    cov_rows=G_data.N_eq+np.arange(Gc.N_eq)
    E_all = 1/TCinv.diagonal()

    # run edit_by_bias to zero out the edited IDs
    edit_by_bias(data, np.zeros(Ip_c.shape[0]), in_TSE, -1, bias_model, args)

    #print(f"iterate_fit: G.shape={Gcoo.shape}, G.nnz={Gcoo.nnz}, data.shape={data.shape}", flush=True)
    in_TSE_original=np.zeros(data.shape, dtype=bool)
    in_TSE_original[in_TSE]=True

    if 'editable' in data.fields:
        N_editable=np.sum(data.editable)
    else:
        N_editable=data.size

    sigma_extra=0
    N_eq = Gcoo.shape[0]
    last_iteration = False
    m0 = np.zeros(Ip_c.shape[0])
    for iteration in range(args['max_iterations']):

        # augment the errors on the shelf
        E2_plus=E_all**2
        TCinv=sp.dia_matrix((1./np.sqrt(E2_plus),0), shape=(N_eq, N_eq))

        # build the parsing matrix that removes invalid rows
        Ip_r=sp.coo_matrix((np.ones(Gc.N_eq+in_TSE.sum()), \
                            (np.arange(Gc.N_eq+in_TSE.sum()), \
                             np.concatenate((np.flatnonzero(in_TSE), cov_rows)))), \
                           shape=(Gc.N_eq+in_TSE.sum(), Gcoo.shape[0])).tocsc()

        if args['VERBOSE']:
            print("starting qr solve for iteration %d at %s" % (iteration, ctime()), flush=True)
        # solve the equations
        tic=time();
        m0_last=m0
        #print("smooth_xytb_fit_aug:iterate_fit: SETTING TOLERANCE TO -2")
        m0=Ip_c.dot(sparseqr.solve(Ip_r.dot(TCinv.dot(Gcoo)).tocoo(), Ip_r.dot(TCinv.dot(rhs))))#, tolerance=-2))
        timing['sparseqr_solve']=time()-tic

        # calculate the full data residual
        r_data=data.z-G_data.toCSR().dot(m0)
        rs_data=r_data/data.sigma
        if last_iteration:
            break

        if 'sigma_extra_bin_spacing' not in args or args['sigma_extra_bin_spacing'] is None:
            # calculate the additional error needed to make the robust spread of the scaled residuals equal to 1
            sigma_extra=calc_sigma_extra(r_data, data.sigma, in_TSE, sigma_extra_masks)
        else:
            sigma_extra=calc_sigma_extra_on_grid(data.x, data.y, r_data, data.sigma, in_TSE,
                                                 sigma_extra_masks=sigma_extra_masks,
                                                 sigma_extra_max=args['sigma_extra_max'],
                                                 spacing=args['sigma_extra_bin_spacing'])
        sigma_aug=np.sqrt(data.sigma**2 + sigma_extra**2)
        # select the data that have scaled residuals < 3 *max(1, sigma_hat)
        in_TSE_last=in_TSE

        in_TSE = np.abs(r_data/sigma_aug) < 3.0

        # if bias_nsigma_edit is specified, check for biases that are more than
        # args['bias_nsigma_edit'] times their expected values.
        bias_editing_changed=edit_by_bias(data, m0, in_TSE, iteration, bias_model, args)
        if 'editable' in data.fields:
            in_TSE[data.editable==0] = in_TSE_original[data.editable==0]

        if args['DEM_tol'] is not None:
            in_TSE = check_data_against_DEM(in_TSE, data, m0, G_data, args['DEM_tol'])

        # quit if the solution is too similar to the previous solution
        if (np.max(np.abs((m0_last-m0)[Gc.TOC['cols']['dz']])) < args['converge_tol_dz']) and (iteration > args['min_iterations']):
            if args['VERBOSE']:
                print("Solution identical to previous iteration with tolerance %3.1f, exiting after iteration %d" % (args['converge_tol_dz'], iteration))
            last_iteration = True
        # select the data that are within 3*sigma of the solution
        if args['VERBOSE']:
            print('found %d in TSE, dt=%3.0f' % ( in_TSE.sum(), timing['sparseqr_solve']), flush=True)
            if sigma_extra_masks is None:
                print(f'\t median(sigma_extra)={np.median(sigma_extra):3.4f}')
            else:
                for m_key, ii in sigma_extra_masks.items():
                    print(f'\t sigma_extra for {m_key} : {np.median(sigma_extra[ii]):3.4f}', flush=True)
                for m_key, ii in sigma_extra_masks.items():
                    print(f'\t sigma_hat for {m_key} : {RDE(r_data[ii]/sigma_aug[ii]):3.4f}', flush=True)
        if iteration > 0  and iteration > args['bias_nsigma_iteration']:
            # Calculate the number of elements that have changed in in_TSE
            frac_TSE_change = len(np.setxor1d(in_TSE_last, in_TSE))/N_editable
            if frac_TSE_change < args['converge_tol_frac_TSE']:
                if args['VERBOSE']:
                    print("filtering unchanged with tolerance %3.5f, will exit after iteration %d"
                          % (args['converge_tol_frac_TSE'], iteration+1))
                last_iteration=True
        if iteration >= np.maximum(args['min_iterations'], args['bias_nsigma_iteration']+1):
            if np.all(sigma_extra < 0.5 * np.min(data.sigma[in_TSE])) and not bias_editing_changed:
                if args['VERBOSE']:
                    print("sigma_extra is small, performing one additional iteration", flush=True)
                last_iteration=True
        if iteration==args['max_iterations']-2:
            last_iteration=True

    return m0, sigma_extra, in_TSE, rs_data

def calc_and_parse_errors(E, Gcoo, TCinv, rhs, Ip_c, Ip_r, grids, G_data, Gc, avg_ops, bias_model, bias_params, dzdt_lags=None, timing={}, error_res_scale=None):
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
                                     'dz': np.reshape(m0[G_data.TOC['cols']['dz']], grids['dz'].shape),\
                                     'cell_area':grids['dz'].cell_area, \
                                     'mask':grids['dz'].mask})
    if 'lagrangian_dz' in grids:
        m['lagrangian_dz']=pc.grid.data().from_dict({'x':grids['lagrangian_dz'].ctrs[1],\
                                         'y':grids['lagrangian_dz'].ctrs[0],\
                                         'dz': np.reshape(m0[G_data.TOC['cols']['lagrangian_dz']], grids['lagrangian_dz'].shape),\
                                         'cell_area':grids['lagrangian_dz'].cell_area, \
                                         'mask':grids['lagrangian_dz'].mask})

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

    m['sensor_bias_grids']=parse_sensor_bias_grids(m0, G_data, grids)

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
            r_notide=(data.z+data.tide-data.z_est)[data.three_sigma_edit]
            r_notide_scaled=r_notide/data.sigma[data.three_sigma_edit]
            m[ff].assign({'misfit_notide_rms':np.sqrt(G_data.toCSR()[:,G_data.TOC['cols'][ff]][data.three_sigma_edit,:].T.dot(r_notide**2)\
                                        .reshape(grids[ff].shape)/m[ff].count)})
            m[ff].assign({'misfit_notide_scaled_rms':np.sqrt(G_data.toCSR()[:,G_data.TOC['cols'][ff]][data.three_sigma_edit,:].T.dot(r_notide_scaled**2)\
                                        .reshape(grids[ff].shape)/m[ff].count)})

def smooth_fit(**kwargs):
    required_fields=('data','W','ctr','spacing','E_RMS')
    args={'reference_epoch':0,
    'W_ctr':1e4,
    'mask_file':None,
    'mask_data':None,
    'mask_update_function':None,
    'mask_scale':None,
    'compute_E':False,
    'max_iterations':10,
    'min_iterations':2,
    'sigma_extra_bin_spacing':None,
    'sigma_extra_max':None,
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
    'prior_args':None,
    'prior_edge_args':None,
    'avg_scales':[],
    'data_slope_sensors':None,
    'E_slope_bias':0.01,
    'E_RMS_d2x_PS_bias':None,
    'E_RMS_PS_bias':None,
    'error_res_scale':None,
    'avg_masks':None,
    'sigma_extra_masks':None,
    'bias_nsigma_edit':None,
    'bias_nsigma_iteration':2,
    'bias_edit_vals':None,
    'sensor_grid_bias_params':None,
    'ancillary_data':None,
    'lagrangian_coords':None,
    'VERBOSE': True,
    'DEBUG': False}
    args.update(kwargs)
    for field in required_fields:
        if field not in kwargs:
            raise ValueError("%s must be defined", field)
    valid_data = np.isfinite(args['data'].z) #np.ones_like(args['data'].x, dtype=bool)
    timing=dict()

    m={}
    E={}
    R={}
    RMS={}
    averaging_ops={}
    tic=time()
    # define the grids
    grids, bds = setup_grids(args)

    # select only the data points that are within the grid bounds
    valid_z0=grids['z0'].validate_pts((args['data'].coords()[0:2]))
    valid_dz=grids['dz'].validate_pts((args['data'].coords()))
    valid_data=valid_data & valid_dz & valid_z0

    if not np.any(valid_data):
        if args['VERBOSE']:
            print("smooth_xytb_fit_aug: no valid data")
        return {'m':m, 'E':E, 'data':None, 'grids':grids, 'valid_data': valid_data, 'TOC':{},'R':{}, 'RMS':{}, 'timing':timing,'E_RMS':args['E_RMS']}

    # subset the data based on the valid mask
    data=args['data'].copy_subset(valid_data)

    # if we have a mask file, use it to subset the data
    # needs to be done after the valid subset because otherwise the interp_mtx for the mask file fails.
    if args['mask_file'] is not None and grids['dz'].mask_3d is None:
        setup_mask(data, grids, valid_data, bds, args)
    else:
        validate_by_dz_mask(data, grids, valid_data)

    # update the sigma_extra_masks variable to the data mask
    if args['sigma_extra_masks'] is not None:
        for key in args['sigma_extra_masks']:
            args['sigma_extra_masks'][key]=args['sigma_extra_masks'][key][valid_data==1]

    # Check if we have any data.  If not, quit
    if data.size==0:
        return {'m':m, 'E':E, 'data':data, 'grids':grids, 'valid_data': valid_data, 'TOC':{},'R':{}, 'RMS':{}, 'timing':timing,'E_RMS':args['E_RMS']}

    # define the interpolation operator, equal to the sum of the dz and z0 operators
    G_data=lin_op(grids['z0'], name='interp_z').interp_mtx(data.coords()[0:2])
    G_data.add(lin_op(grids['dz'], name='interp_dz').interp_mtx(data.coords()))
    if args['lagrangian_coords'] is not None:
        G_data.add(lin_op(grids['lagrangian_dz'], name='interp_lagrangian_dz').interp_mtx(
            [getattr(data, field) for field in args['lagrangian_coords']], bounds_error=False))

    # define the smoothness constraints
    constraint_op_list=[]
    setup_smoothness_constraints(grids, constraint_op_list, args['E_RMS'], args['mask_scale'])

    ### NB: NEED TO MAKE THIS WORK WITH SETUP_GRID_BIAS
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

    # check that the data_slope sensors are in the data that has passed the filtering steps
    args['data_slope_sensors']=args['data_slope_sensors'][np.in1d(args['data_slope_sensors'], data.sensor)]
    if args['data_slope_sensors'] is not None and len(args['data_slope_sensors'])>0:
        bias_model['E_slope_bias']=args['E_slope_bias']
        G_slope_bias, Gc_slope_bias, bias_model = \
            data_slope_bias(data, bias_model, sensors=args['data_slope_sensors'],\
                            col_0=G_data.col_N, E_rms_bias=args['E_slope_bias'])
        if G_slope_bias is not None:
            G_data.add(G_slope_bias)
            constraint_op_list.append(Gc_slope_bias)

    if args['sensor_grid_bias_params'] is not None:
        for params in args['sensor_grid_bias_params']:
            setup_sensor_grid_bias(data, grids, G_data, \
                                   constraint_op_list, **params)

        # setup priors
    if args['prior_args'] is not None:
        constraint_op_list += match_prior_dz(grids, **args['prior_args'])

    if args['prior_edge_args'] is not None:
        prior_ops, prior_xy = match_tile_edges(grids, args['reference_epoch'], **args['prior_edge_args'])
        constraint_op_list += prior_ops

    for op in constraint_op_list:
        if op.prior is None:
            op.prior=np.zeros_like(op.expected)

    # put the equations together
    Gc = lin_op(None, name='constraints').vstack(constraint_op_list)

    N_eq=G_data.N_eq+Gc.N_eq

    # put together all the errors
    Ec=np.zeros(Gc.N_eq)
    for op in constraint_op_list:
        try:
            Ec[Gc.TOC['rows'][op.name]]=op.expected
        except ValueError as E:
            print("smooth_xytb_fit_aug:\n\t\tproblem with "+op.name)
            raise(E)
    #if args['data_slope_sensors'] is not None and len(args['data_slope_sensors']) > 0:
    #    Ec[Gc.TOC['rows'][Gc_slope_bias.name]] = Cvals_slope_bias
    Ed=data.sigma.ravel()
    if np.any(Ed==0):
        raise(ValueError('zero value found in data sigma'))
    if np.any(Ec==0):
        raise(ValueError('zero value found in constraint sigma'))

    args['DEBUG'] = True
    if args['DEBUG']:
        print_TOC(G_data, Gc)

    # calculate the inverse square root of the data covariance matrix
    TCinv=sp.dia_matrix((1./np.concatenate((Ed, Ec)), 0), shape=(N_eq, N_eq))

    # define the right hand side of the equation
    rhs=np.zeros([N_eq])
    rhs[0:data.size]=data.z.ravel()
    rhs[data.size:]=np.concatenate([op.prior for op in constraint_op_list])

    # put the fit and constraint matrices together
    Gcoo=sp.vstack([G_data.toCSR(), Gc.toCSR()]).tocoo()

    # define the matrix that sets dz[reference_epoch]=0 by removing columns from the solution:
    Ip_c = build_reference_epoch_matrix(G_data, Gc, grids, args['reference_epoch'])

    # eliminate the columns for the model variables that are set to zero
    Gcoo=Gcoo.dot(Ip_c)
    timing['setup']=time()-tic

    # initialize the book-keeping matrices for the inversion
    if "three_sigma_edit" in data.fields:
        in_TSE=data.three_sigma_edit > 0.01
    else:
        in_TSE=np.ones(G_data.N_eq, dtype=bool)

    if args['VERBOSE']:
        print("initial: %d:" % G_data.r.max(), flush=True)

    # if we've done any iterations, parse the model and the data residuals
    if args['max_iterations'] > 0:
        tic_iteration=time()
        m0, sigma_extra, in_TSE, rs_data=iterate_fit(data, Gcoo, rhs, \
                                TCinv, G_data, Gc, in_TSE, Ip_c, timing,
                                args, grids,\
                                bias_model=bias_model, \
                                sigma_extra_masks=args['sigma_extra_masks'])

        timing['iteration']=time()-tic_iteration
        valid_data[valid_data]=in_TSE
        data.assign({'three_sigma_edit':in_TSE})
        data.assign({'sigma_extra':sigma_extra})

        # report the model-based estimate of the data points
        data.assign({'z_est':np.reshape(G_data.toCSR().dot(m0), data.shape)})

        if args['mask_update_function'] is not None:
            parse_model(m, m0, data, R, RMS, G_data, averaging_ops, Gc, Ec, grids, bias_model, args)
            args['mask_update_function'](grids, m, args)

        # setup operators that take averages of the grid at different scales
        averaging_ops=setup_averaging_ops(grids['dz'], grids['z0'].col_N, args, grids['dz'].cell_area)

        # setup masked averaging ops
        averaging_ops.update(setup_avg_mask_ops(grids['dz'], G_data.col_N, args['avg_masks'], args['dzdt_lags']))

        parse_model(m, m0, data, R, RMS, G_data, averaging_ops, Gc, Ec, grids, bias_model, args)
        r_data=data.z_est[data.three_sigma_edit==1]-data.z[data.three_sigma_edit==1]
        R['data']=np.sum(((r_data/data.sigma[data.three_sigma_edit==1])**2))
        RMS['data']=np.sqrt(np.mean((data.z_est[data.three_sigma_edit==1]-data.z[data.three_sigma_edit==1])**2))
    else:
        # still need to make the averaging ops
        averaging_ops=setup_averaging_ops(grids['dz'], grids['z0'].col_N, args, grids['dz'].cell_area)
        # setup masked averaging ops
        averaging_ops.update(setup_avg_mask_ops(grids['dz'], G_data.col_N, args['avg_masks'], args['dzdt_lags']))


    # Compute the error in the solution if requested
    if args['compute_E']:
        # if sigma_extra is not a data field, assume it is zero
        if not 'sigma_extra' in data.fields:
            data.assign({'sigma_extra': np.zeros_like(data.sigma)})
            r_data=data.z_est[data.three_sigma_edit==1]-data.z[data.three_sigma_edit==1]
            data.sigma_extra[data.three_sigma_edit==1] = calc_sigma_extra(r_data, data.sigma[data.three_sigma_edit==1])

        # rebuild TCinv to take into account the extra error
        TCinv=sp.dia_matrix((1./np.concatenate((np.sqrt(Ed**2+data.sigma_extra**2), Ec)), 0), shape=(N_eq, N_eq))

        # We have generally not done any iterations at this point, so need to make the Ip_r matrix
        cov_rows=G_data.N_eq+np.arange(Gc.N_eq)
        Ip_r=sp.coo_matrix((np.ones(Gc.N_eq+in_TSE.sum()), \
                           (np.arange(Gc.N_eq+in_TSE.sum()), \
                            np.concatenate((np.flatnonzero(in_TSE), cov_rows)))), \
                           shape=(Gc.N_eq+in_TSE.sum(), Gcoo.shape[0])).tocsc()
        if args['VERBOSE']:
            print("Starting uncertainty calculation", flush=True)
            tic_error=time()
        calc_and_parse_errors(E, Gcoo, TCinv, rhs, Ip_c, Ip_r, grids, G_data, Gc, averaging_ops, \
                         bias_model, args['bias_params'], dzdt_lags=args['dzdt_lags'], timing=timing, \
                             error_res_scale=args['error_res_scale'])
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

    S=smooth_fit(data=data, ctr=ctr, W=W, spacing=spacing, E_RMS=E_RMS,
                     reference_epoch=2, N_subset=None, compute_E=False,
                     max_iterations=2,
                     srs_proj4=SRS_proj4, VERBOSE=True, dzdt_lags=[1])
    return S


if __name__=='__main__':
    main()
