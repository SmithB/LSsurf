#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:34:24 2022

@author: ben
"""

''' Functions to setup constraints for smooth surface fits
    Provides: setup_smoothness_constraints, build_reference_epoch_matrix

'''

import numpy as np
from LSsurf.fd_grid import fd_grid
from LSsurf.lin_op import lin_op
from LSsurf.unique_by_rows import unique_by_rows
import pointCollection as pc
import scipy.sparse as sp



def setup_smoothness_constraints(grids, constraint_op_list, E_RMS, mask_scale, scaling_masks=None):
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
    if scaling_masks is None:
        scaling_masks={}

    # make the smoothness constraints for z0
    root_delta_A_z0=np.sqrt(np.prod(grids['z0'].delta))
    grad2_z0=lin_op(grids['z0'], name='grad2_z0').grad2(DOF='z0')
    grad2_z0.expected=E_RMS['d2z0_dx2']/root_delta_A_z0*grad2_z0.mask_for_ind0(mask_scale)

    constraint_op_list += [grad2_z0]
    if 'dz0_dx' in E_RMS:
        grad_z0=lin_op(grids['z0'], name='grad_z0').grad(DOF='z0')
        grad_z0.expected=E_RMS['dz0_dx']/root_delta_A_z0*grad_z0.mask_for_ind0(mask_scale)
        constraint_op_list += [grad_z0]

    if 'z0' in E_RMS and E_RMS['z0'] is not None:
        mag_z0=lin_op(grids['z0'], name='mag_z0').one(DOF='z0')
        mag_z0.expected=E_RMS['z0']/root_delta_A_z0*np.ones_like(mag_z0.v.ravel())
        constraint_op_list += [mag_z0]

    # make the smoothness constraints for dz
    root_delta_V_dz=np.sqrt(np.prod(grids['dz'].delta))
    if 'd3z_dx2dt' in E_RMS and E_RMS['d3z_dx2dt'] is not None:
        grad2_dz=lin_op(grids['dz'], name='grad2_dzdt').grad2_dzdt(DOF='z', t_lag=1)
        grad2_dz.expected=E_RMS['d3z_dx2dt']/root_delta_V_dz*grad2_dz.mask_for_ind0(mask_scale)
        if 'd3z_dx2dt' in scaling_masks:
            grad2_dz.expected *= grad2_dz.mask_for_ind0(mask=scaling_masks['d3z_dx2dt'])
        constraint_op_list += [grad2_dz]

    if 'd2z_dxdt' in E_RMS and E_RMS['d2z_dxdt'] is not None:
        grad_dzdt=lin_op(grids['dz'], name='grad_dzdt').grad_dzdt(DOF='z', t_lag=1)
        grad_dzdt.expected=E_RMS['d2z_dxdt']/root_delta_V_dz*grad_dzdt.mask_for_ind0(mask_scale)
        for key in ['d2z_dx2dt','d3z_dx2dt']:
            if key in scaling_masks:
                grad_dzdt.expected *= grad_dzdt.mask_for_ind0(mask=scaling_masks[key])
                break
        constraint_op_list += [ grad_dzdt ]

    if 'd2z_dt2' in E_RMS and E_RMS['d2z_dt2'] is not None:
        d2z_dt2=lin_op(grids['dz'], name='d2z_dt2').d2z_dt2(DOF='z')
        d2z_dt2.expected=np.zeros(d2z_dt2.N_eq) + E_RMS['d2z_dt2']/root_delta_V_dz
        if 'd2z_dt2' in scaling_masks:
            d2z_dt2.expected *= d2z_dt2.mask_for_ind0(mask=scaling_masks['d2z_dt2'])
        constraint_op_list += [d2z_dt2]

    if 'lagrangian_dz' in E_RMS and E_RMS['lagrangian_dz'] is not None:
        root_A_lag=np.sqrt(grids['lagrangian_dz'].delta[0]*grids['lagrangian_dz'].delta[1])
        lag_dz=lin_op(grids['lagrangian_dz'], name='lagrangian_rms').one(DOF='lagrangian_dz')
        lag_dz.expected = np.zeros(lag_dz.N_eq) + E_RMS['lagrangian_dz']/root_A_lag
        constraint_op_list += [lag_dz]

    if 'lagrangian_dzdx' in E_RMS and E_RMS['lagrangian_dzdx'] is not None:
        root_A_lag=np.sqrt(grids['lagrangian_dz'].delta[0]*grids['lagrangian_dz'].delta[1])
        grad_lag_dz=lin_op(grids['lagrangian_dz'], name='lagrangian_rms_grad').grad(DOF='lagrangian_dz')
        grad_lag_dz.expected = np.zeros(grad_lag_dz.N_eq) + E_RMS['lagrangian_dzdx']/root_A_lag
        constraint_op_list += [grad_lag_dz]

    for constraint in constraint_op_list:
        if np.any(constraint.expected==0):
            raise(ValueError(f'found zero value in the expected values for {constraint.name}'))


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
