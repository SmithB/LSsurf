# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 2021

@author: ben
"""
import numpy as np
from LSsurf.lin_op import lin_op
import pointCollection as pc
import glob
import re
import os

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

def prior_reference_time(dz, default=None):
    """
    Find the reference time of a saved dz model

    A saved dz model is a difference from its reference epoch, m(t) - m(t_ref),
    so dz (and, if the errors were calculated, sigma_dz) are zero at every node
    at t_ref.  A constraint from the model, m(t) - m(t_ref) = dz(t), needs the
    model's own t_ref, which need not be that of the fit being constrained
    (e.g. a model fit over a different time span with an index reference epoch).

    inputs:
        pointCollection.grid.data dz: the saved model, with a 'dz' field (and
            optionally 'sigma_dz') whose third dimension is time
        float default: time returned if no epoch, or more than one, has dz and
            sigma_dz equal to zero at every node where dz is finite
    output:
        float: the reference time
    """
    t = getattr(dz, 'time', None)
    if t is None:
        t = getattr(dz, 't', None)
    if t is None:
        return default
    t = np.atleast_1d(np.asarray(t, dtype=float))
    z = np.asarray(dz.dz, dtype=float)
    if z.ndim < 3 or z.shape[2] != t.size:
        return default
    sigma = np.asarray(dz.sigma_dz, dtype=float) if 'sigma_dz' in dz.fields else None
    candidates = []
    for k in range(t.size):
        zk = z[:, :, k]
        finite = np.isfinite(zk)
        if not np.any(finite) or np.any(zk[finite] != 0):
            continue
        if sigma is not None:
            sk = sigma[:, :, k]
            if np.any(sk[finite & np.isfinite(sk)] != 0):
                continue
        candidates += [t[k]]
    if len(candidates) == 1:
        return candidates[0]
    return default

def make_prior_op(grids, dz, src_name, ref_time=0, sigma_scale=1):

    if 't' not in dz.fields:
        dz.assign(t=dz.time)
    # the reference epoch gives no constraint (dz = 0 there), and points
    # without a positive, finite error would give invalid constraint equations
    keep = (dz.t != ref_time) & np.isfinite(dz.dz)
    if 'sigma_dz' in dz.fields:
        keep &= np.isfinite(dz.sigma_dz) & (dz.sigma_dz > 0)
    dz=dz[keep]
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
    return m1

def match_prior_dz(grids, dzs=None, filenames=None, ref_epoch=0, group='dz', field_mapping=None, \
                    skip={'xy':1,'t':1}, edge_pad={'xy':0, 't':0}, sigma_scale=1,
                    sigma_max=None):
    """
    Make an operator to match a saved dz model

    inputs:
        dict grids: dict containing LSsurf.fdgrid objects.  Must include a 'dz' entry
        iterable filenames: h5 filenames from which to read the data
        iterable dzs: pointCollection.grid.data() objects containing height-change data
        int ref_epoch: index of the epoch of each saved dz model to which it is
            referenced, used only if the reference epoch cannot be found from the
            model itself (the epoch at which dz and sigma_dz are zero everywhere;
            see prior_reference_time)
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
        elif isinstance(datasrc, pc.grid.data):
            dz=datasrc.copy()
            src_name='internal_prior'
        else:
            raise TypeError('datasource not understood')

        # get the center and width of the range
        yc, xc, tc=[np.mean(getattr(dz, field)) for field in ('y','x','t')]
        yw, xw, tw=[np.diff(getattr(dz, field)[[0, -1]]) for field in ('y','x','t')]

        #Keep only the part of the data that's within the grid
        bds = match_range(grids['dz'].bds, (yc+np.array([-1, 1])*(yw/2-edge_pad['xy']),
                                            xc+np.array([-1, 1])*(xw/2-edge_pad['xy']),
                                            tc+np.array([-1, 1])*tw/2))
        # the prior's own reference time (found from the full grid, before cropping)
        ref_time = prior_reference_time(dz, default=np.atleast_1d(dz.t)[ref_epoch])

        dz.crop( bds[1], bds[0], bds[2] )

        rows=np.arange(skip['xy']/2, dz.shape[0], dtype='int')
        cols=np.arange(skip['xy']/2, dz.shape[1], dtype='int')
        bands=np.arange(skip['t']/2, dz.shape[2], dtype='int')
        dz.index(rows, cols, band_ind=bands)
        dz=dz.as_points()

        m1 = make_prior_op(grids, dz, src_name, ref_time=ref_time, sigma_scale=sigma_scale)
        m_list += [m1]

    return m_list


def match_tile_edges(grids, ref_epoch, prior_dir=None,
                     tile_spacing=4.e4,
                     edge_include=3.e3,
                     group='dz', field_mapping=None, \
                     sigma_scale=1, sigma_max=None, verbose=False):
    '''
    Make a set of operators to match the current tile to adjacent tiles

    Parameters:
        grids (dict): dict containing LSsurf.fdgrid objects.  Must include a 'dz' entry
        xy0 (2-iterable) : center coordinates for the grid to be constrained
        prior_dir (str or iterable): directory containing tiles to be matched
        tile_W (scalar ) : Width of the tiles (default: 8e4)
        tile_spacing (scalar) : distance between tile centers
        edge_include (scalar) : Width of the constraining grids to include at the edges
        ref_epoch (int) : index of this fit's reference epoch.  Each adjacent
            tile's constraints use that tile's own reference time (the epoch at
            which its dz and sigma_dz are zero, see prior_reference_time); this
            fit's reference time is used only if that epoch can't be found.
        group (str) : group in the file in which the dz values are saved (default='dz')
        field_mapping (dict) : dict mapping between saved fields and fields 'dz and 'sigma_dz'
        sigma_scale (scalar) : value by which to scale the uncertainties in the grid
        sigma_max (scalar) : ignore points with sigma values larger than this

    Returns:
        constraint_list (list) : list of LSsurf.lin_op objects containing the constraint equations.
    '''
    xy0 = [np.mean(grids['z0'].bds[dim]) for dim in [1, 0]]
    tile_W = np.diff(grids['z0'].bds[0])

    tile_re=re.compile('E(.*)_N(.*).h5')
    if isinstance(prior_dir, (list, tuple)):
        all_files = []
        for thedir in prior_dir:
            list_file=os.path.join(thedir,'list_of_tiles.txt')
            if os.path.isfile(list_file):
                with open(list_file,'r') as fh:
                    for line in fh:
                        all_files += [os.path.join(thedir, line.rstrip())]
            else:
                all_files += glob.glob(thedir+'/E*N*.h5')
    else:
        list_file=os.path.join(prior_dir,'list_of_tiles.txt')
        if os.path.isfile(list_file):
            with open(list_file,'r') as fh:
                all_files=[os.path.join(prior_dir, line.rstrip()) for line in fh]
        else:
            all_files=glob.glob(prior_dir+'/E*N*.h5')
    HX = tile_W/2
    W_overlap = (tile_W-tile_spacing)/2

    constraint_list=[]
    constraint_xy={}
    for file in all_files:
        try:
            xyc=tile_re.search(os.path.basename(file)).groups()
            xyc=(int(xyc[0])*1000., int(xyc[1])*1000.)
        except Exception:
            print(f"couldn't parse filename {file}")
            continue

        # don't constrain based on the same xy0
        if xyc[0]==xy0[0] and xyc[1]==xy0[1]:
            continue
        # skip tiles that are too far away:
        if np.max(np.abs(np.r_[xy0]-np.r_[xyc]))>(tile_spacing+edge_include):
            continue

        dz=pc.grid.data().from_h5(file, group=group, fields=field_mapping)
        if 'sigma_dz' not in dz.fields:
            print('match_tile_edges: missing sigma_dz for '+dz.filename)
            continue

        if 't' not in dz.fields:
            dz.assign(t=dz.time)

        src_name=file
        # the constraint is m(t) - m(t_ref) = dz(t), with the adjacent tile's
        # reference time: the epoch at which its dz and sigma_dz are zero.  Its
        # time grid and reference epoch need not be this fit's (e.g. a tile fit
        # over a longer time span, with an index reference epoch, whose output
        # was cropped in time).  If the tile's reference epoch can't be found,
        # assume that it is this fit's.
        ref_time = prior_reference_time(dz, default=grids['dz'].ctrs[2][ref_epoch])
        for field in ['cell_area','mask']:
            if field in dz.fields:
                dz.fields.remove(field)
        dz=dz.as_points()

        # TBD: fix dz.as_points to keep the 't' field
        if 't' not in dz.fields:
            dz.assign(t=dz.time)

        if 'sigma_dz' not in dz.fields:
            print(f"match_priors: no sigma_dz found for {dz.filename}")
            continue

        # select points that are close to the edge of the current tile
        ii  =  (dz.x > xy0[0] + HX - edge_include ) | (dz.x < xy0[0] - HX + edge_include )
        ii |= ((dz.y > xy0[1] + HX - edge_include ) | (dz.y < xy0[1] - HX + edge_include ))

        # select points that are within the current tile
        ii &= ((dz.x >= xy0[0] - HX) & (dz.x <= xy0[0] + HX))
        ii &= ((dz.y >= xy0[1] - HX) & (dz.y <= xy0[1] + HX))

        # select points that aren't too far from the prior tile center in either direction
        ii &= (np.abs(dz.x-xyc[0]) < HX-W_overlap)
        ii &= (np.abs(dz.y-xyc[1]) < HX-W_overlap)

        ii &= grids['z0'].validate_pts((dz.y, dz.x))
        if not hasattr(dz,'t'):
            dz.t = dz.time
        ii &= grids['dz'].validate_pts((dz.y, dz.x, dz.t))

        if not np.any(ii):
            continue

        constraint_list += [make_prior_op(grids, dz[ii], src_name,
                                          ref_time=ref_time, sigma_scale=sigma_scale)]
        # make a list of unique constraint points (for debugging)
        u_xy= np.unique(dz.x[ii]+1j*dz.y[ii])
        constraint_xy[file]=(np.real(u_xy), np.imag(u_xy))
        if verbose:
            print(f"sigma_scale={sigma_scale}")
    return constraint_list, constraint_xy
