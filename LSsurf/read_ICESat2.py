#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:00:05 2019

@author: ben
"""
import numpy as np
from ATL11.check_ATL06_hold_list import read_files as read_hold_files
#from PointDatabase.geo_index import geo_index
#from PointDatabase.point_data import point_data
#from PointDatabase.ATL06_filters import segDifferenceFilter
import pointCollection as pc

def segDifferenceFilter(D6, tol=2, setValid=True, toNaN=False, subset=False):
    dAT=20.
    if D6.h_li.shape[0] < 3:
        mask=np.ones_like(D6.h_li, dtype=bool)
        return mask
    EPplus=D6.h_li + dAT*D6.dh_fit_dx
    EPminus=D6.h_li - dAT*D6.dh_fit_dx
    segDiff=np.zeros_like(D6.h_li)
    if len(D6.h_li.shape)>1:
        segDiff[0:-1,:]=np.abs(EPplus[0:-1,:]-D6.h_li[1:, :])
        segDiff[1:,:]=np.maximum(segDiff[1:,:], np.abs(D6.h_li[0:-1,:]-EPminus[1:,:]))
    else:
        segDiff[0:-1]=np.abs(EPplus[0:-1]-D6.h_li[1:])
        segDiff[1:]=np.maximum(segDiff[1:], np.abs(D6.h_li[0:-1]-EPminus[1:]))
    mask=segDiff<tol
    if setValid:
        D6.valid=D6.valid & mask
    if toNaN:
        D6.h_li[mask==0]=np.NaN
    if subset:
        D6.index(np.any(mask==1, axis=1))

    return mask


def read_ICESat2(xy0, W, gI_file, sensor=2, SRS_proj4=None, tiled=True, seg_diff_tol=2, blockmedian_scale=None, cplx_accept_threshold=0.):
    field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude','atl06_quality_summary','segment_id','sigma_geo_h'], 
                'fit_statistics':['dh_fit_dx', 'dh_fit_dy', 'n_fit_photons','w_surface_window_final','snr_significance'],
                'geophysical':['tide_ocean'],
                'ground_track':['x_atc'],
                'orbit_info':['rgt','cycle_number'],
                'derived':['valid', 'BP','LR']}
    if tiled:
        fields=[]
        for key in field_dict:
            fields += field_dict[key]
        fields += ['x','y']
    else:
        fields=field_dict
    px, py=np.meshgrid(np.arange(xy0[0]-W['x']/2, xy0[0]+W['x']/2+1.e4, 1.e4),np.arange(xy0[1]-W['y']/2, xy0[1]+W['y']/2+1.e4, 1.e4) )  
    D0=pc.geoIndex().from_file(gI_file).query_xy((px.ravel(), py.ravel()), fields=fields)
    if D0 is None or len(D0)==0:
        return [None]
    # check the D6 filenames against the blacklist of bad files
    if tiled:
        D0=pc.reconstruct_ATL06_tracks(pc.data(fields=fields).from_list(D0))
    hold_list = read_hold_files()
    hold_list = {(item[0], item[1]) for item in hold_list}
    D1=list()
    for ind, D in enumerate(D0):
        if D.size<2:
            continue
        delete_file = (D.cycle_number[0], D.rgt[0]) in hold_list
        if delete_file==0:
            D1.append(D)

    # D1 is now a filtered version of D0
    D_pt=pc.data().from_list(D1)
    bin_xy=1.e4*np.round((D_pt.x+1j*D_pt.y)/1.e4)
    cplx_bins=[]
    for xy0 in np.unique(bin_xy):
        ii=np.flatnonzero(bin_xy==xy0)
        if np.mean(D_pt.atl06_quality_summary[ii]==0) < cplx_accept_threshold:
            cplx_bins+=[xy0]
    cplx_bins=np.array(cplx_bins)
    sigma_corr=np.zeros(len(D1))
    for ind, D in enumerate(D1):


        valid=segDifferenceFilter(D, setValid=False, toNaN=False, tol=seg_diff_tol)
        
        D.assign({'quality':D.atl06_quality_summary})
        
        cplx_data=np.in1d(1.e4*np.round((D.x+1j*D.y)/1.e4), cplx_bins)
        if np.any(cplx_data):
            D.quality[cplx_data] = (D.snr_significance[cplx_data] > 0.02) | \
                (D.n_fit_photons[cplx_data]/D.w_surface_window_final[cplx_data] < 5)
            valid[cplx_data] |= segDifferenceFilter(D, setValid=False, toNaN=False, tol=2*seg_diff_tol)[cplx_data]
        
        D.h_li[valid==0] = np.NaN
        D.h_li[D.quality==1] = np.NaN
        if blockmedian_scale is not None:
            D.blockmedian(blockmedian_scale, field='h_li')
              
        # rename the h_li field to 'z', and set time to the year
        # note that the extra half day is needed because 2018 is in between two leap years
        D.assign({'z': D.h_li, 'time':D.delta_time/24/3600/365.25+2018+0.5/365.25,
                  'sigma':D.h_li_sigma,'cycle':D.cycle_number})
        # thin to 40 m
        D.index(np.mod(D.segment_id, 2)==0)  
        if 'x' not in D.fields:
            D.get_xy(SRS_proj4)
        #D.index(np.isfinite(D.h_li), list_of_fields=['x','y','z','time','year','sigma','sigma_corr','rgt','cycle'])
                
        D.assign({'sensor':np.zeros_like(D.x)+sensor})
        if np.any(np.isfinite(D.dh_fit_dy)):
            dhdy_med=np.nanmedian(D.dh_fit_dy)
            dhdx_med=np.nanmedian(D.dh_fit_dx)
        else:
            dhdy_med=np.NaN
            dhdx_med=np.NaN
        sigma_geo_x=8
        sigma_corr[ind]=np.sqrt(0.03**2+sigma_geo_x**2*(dhdx_med**2+dhdy_med**2))
        if ~np.isfinite(sigma_corr[ind]):
            sigma_corr[ind]=0.1
        D.assign({'sigma_corr':np.zeros_like(D.h_li)+sigma_corr[ind]})
        D1[ind]=D.copy_subset(np.isfinite(D.h_li), datasets=['x','y','z','time',\
                 'delta_time','sigma','sigma_corr','rgt','cycle',\
                     'sensor', 'BP','LR'])
    return D1

def main():
    import glob
    #gI_file='/Volumes/insar10/ben/IS2_tiles/GL/GeoIndex.h5'
    gI_file=glob.glob('/Volumes/ice2/ben/scf/GL_06/latest/tiles/*/GeoIndex.h5')[0]
    xy0=[-170000.0, -2280000.0]
    dI=pc.data().from_list(read_ICESat2(xy0, {'x':4.e4, 'y':4.e4}, gI_file, cplx_accept_threshold=0.25))
    import matplotlib.pyplot as plt

    plt.scatter(dI.x, dI.y, c=dI.z, linewidth=0); plt.colorbar()
    plt.axis('equal')
    
    
if __name__=='__main__':
    main()