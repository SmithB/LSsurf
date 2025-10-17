# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 2021

@author: ben
"""
import numpy as np
from LSsurf.fd_grid import fd_grid
from LSsurf.lin_op import lin_op
import scipy.sparse as sp
import copy
import sparseqr
from time import time, ctime
from LSsurf.RDE import RDE
from LSsurf.unique_by_rows import unique_by_rows
import pointCollection as pc
import re

def match_range(b0, b1):
    """
    Find the overlap between two sets of bounds, b0 and b1
    
    inputs:
       iterables b0, b1: iterable of iterables, one for each dimension
    output:
       list b2: list of tuples giving the overlapping region
    """
    return [ (np.maximum(bi[0], bj[0]), np.minimum(bi[1], bj[1]))\
          for bi, bj in zip(b0, b1) ]

def match_dz_op(grids, dz=None, file=None, ref_epoch=0, group='dz', field_mapping={'dz':dz,'sigma_dz':sigma_dz}, \
                    skip={'xy':1,'t':1}, edge_pad={'xy':0, 't':0}):
    """
    Make an operator to match a saved dz model

    inputs: 
        str file: h5 file from which to read the data
        float ref_epoch: Time value to which the saved dz model is referenced
        str group: group in the file in which the dz values are saved (default='dz')
        dict field_mapping: dict mapping between saved fields and fields 'dz and 'sigma_dz'
        dict skip: values to skip in xy and t (one field for each)
        dict edge_pad: cells closer than this distance to the edge will be ignored (one float for xy, one for t)
    outputs:
        'lin_op' m1: matrix mapping model parameters to saved fit parameters
        numpy array d: difference values matching m1
        numpy array sigma_d: uncertainties in the d values
    """
    if file is not None:
        temp=pc.grid.data().from_h5(file, group=group, fields=None)
    else:
        temp=dz.copy()
        
    yc, xc, tc=[np.mean(getattr(temp, field)) for field in ('y','x','t')]
    yw, xw, tw=[np.diff(getattr(temp, field)[[0, -1]] for field in ('y','x','t'))]
    if file is not None:
        dz=pc.grid.data.from_h5(file, group=group, field_mapping=field_mapping,
                                  bounds=[xc+np.array([-1, 1])*(xw/2-edge_pad), yc+np.array([-1, 1])*(xw/2-edge_pad), ])
    else:
        dz=dz.copy().crop(xc+np.array([-1, 1])*(xw/2-edge_pad), yc+np.array([-1, 1])*(xw/2-edge_pad))

    yg, xg, tg = np.meshgrid(dz.y[::skip['xy']],
                             dz.x[::skip['xy']],
                             dz.t[::skip['t']])
    # first matrix interpolates the model to the fit values at the delta-z time
    m1 = lin_op(grids['dz'], name='interp_dz').interp_mtx((yg.ravel(), xg.ravel(), tg.ravel()))
    # second matrix interpolates the model to data's reference epoch.
    m0 = lin_op(grids['dz'], name='interp_dz').interp_mtx((yg.ravel(), xg.ravel(), np.zeros_like(tg.ravel())+dz.t[ref_epoch]))
    # invert the values of m1 (so that the reference epoch is subtracted)
    m0.v *= -1
    m1.add(m0)
    d=dz.dz[::skip['xy']][::skip['xy']][::skip['t']]
    sigma_d=dz.sigma_dz[::skip['xy']][::skip['xy']][::skip['t']]
    return m1, d, sigma_d
