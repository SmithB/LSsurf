# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 2021

@author: ben
"""
import numpy as np
from LSsurf.lin_op import lin_op
import pointCollection as pc

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

def match_prior_dz(grids, dzs=None, filenames=None, ref_epoch=0, group='dz', field_mapping=None, \
                    skip={'xy':1,'t':1}, edge_pad={'xy':0, 't':0}, sigma_scale=1,
                    sigma_max=None):
    """
    Make an operator to match a saved dz model

    inputs:
        iterable filenames: h5 filenames from which to read the data
        iterable dzs: pointCollection.grid.data() objects containing height-change data
        float ref_epoch: Time value to which the saved dz model is referenced
        str group: group in the file in which the dz values are saved (default='dz')
        dict field_mapping: dict mapping between saved fields and fields 'dz and 'sigma_dz'
        dict skip: values to skip in xy and t (one field for each)
        dict edge_pad: cells closer than this distance to the edge will be ignored (one float for xy, one for t)
        float sigma_scale: value by which to scale the uncertainties in the grid
    outputs:
        m_list: list of constraint operators
    """
    if dzs is None:
        dzs=[]
    if filenames is None:
        filenames=[]
    if field_mapping is None:
        field_mapping={'dz':'dz','sigma_dz':'sigma_dz'}
    m_list=[]

    for datasrc in filenames+dzs:
        if datasrc is None:
            continue
        if isinstance(datasrc, (str)):
            dz=pc.grid.data().from_h5(datasrc, group=group, fields=field_mapping)
            src_name=datasrc
        elif isinstance(datasrc, [pc.grid.data]):
            dz=datasrc.copy()
            src_name='internal_prior'
        else:
            raise TypeError('datasource not understood')

        # get the center and width of the range
        yc, xc, tc=[np.mean(getattr(dz, field)) for field in ('y','x','t')]
        print([getattr(dz, field)[[0, -1]] for field in ('y','x','t')])
        yw, xw, tw=[np.diff(getattr(dz, field)[[0, -1]]) for field in ('y','x','t')]

        #Keep only the part of the data that's within the grid
        bds = match_range(grids['dz'].bds, (yc+np.array([-1, 1])*(yw/2-edge_pad['xy']),
                                            xc+np.array([-1, 1])*(xw/2-edge_pad['xy']),
                                            tc+np.array([-1, 1])*tw/2))
        ref_time=dz.t[ref_epoch]

        dz.crop( bds[1], bds[0], bds[2] )

        rows=np.arange(skip['xy']/2, dz.shape[0], dtype='int')
        cols=np.arange(skip['xy']/2, dz.shape[1], dtype='int')
        bands=np.arange(skip['t']/2, dz.shape[2], dtype='int')
        dz.index(rows, cols, band_ind=bands)
        dz=dz.as_points()
        dz=dz[dz.t != ref_time]
        # first matrix interpolates the model to the fit values at the delta-z time
        m1 = lin_op(grids['dz'], name='prior_dz1_'+src_name)\
            .interp_mtx((dz.y, dz.x, dz.t))
        # second matrix interpolates the model to data's reference epoch.
        m0 = lin_op(grids['dz'], name='prior_dz0_'+src_name)\
            .interp_mtx((dz.y, dz.x, np.zeros_like(dz.t)+ref_time))
        # invert the values of m1 (so that the reference epoch is subtracted)
        m0.v *= -1
        m1.add(m0)
        # set the expected misfit to sigma_scale*dz.sigma_d
        m1.expected=sigma_scale*dz.sigma_dz
        # set the prior value to dz.dz
        m1.prior=dz.dz
        m_list += [m1]

    return m_list
