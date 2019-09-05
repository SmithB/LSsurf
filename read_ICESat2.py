#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:00:05 2019

@author: ben
"""
import numpy as np
from PointDatabase.check_ATL06_blacklist import check_rgt_cycle_blacklist
from PointDatabase.geo_index import geo_index
from PointDatabase.point_data import point_data
from PointDatabase.ATL06_filters import segDifferenceFilter
from PointDatabase.ATL06_tiles import reconstruct_tracks
 
def read_ICESat2(xy0, W, gI_file, sensor=2, SRS_proj4=None, tiled=True, seg_diff_tol=2, blockmedian_scale=None, cplx_accept_threshold=0.):
    field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude','atl06_quality_summary','segment_id','sigma_geo_h'], 
                'fit_statistics':['dh_fit_dx', 'n_fit_photons','w_surface_window_final','snr_significance'],
                'ground_track':['x_atc'],
                'geophysical' : ['dac'],
                'orbit_info':['rgt','cycle_number'],
                'derived':['valid','matlab_time', 'BP','LR']}
    if tiled:
        fields=[]
        for key in field_dict:
            fields += field_dict[key]
        fields += ['x','y']
    else:
        fields=field_dict
    px, py=np.meshgrid(np.arange(xy0[0]-W['x']/2, xy0[0]+W['x']/2+1.e4, 1.e4),np.arange(xy0[1]-W['y']/2, xy0[1]+W['y']/2+1.e4, 1.e4) )  
    D0=geo_index().from_file(gI_file).query_xy((px.ravel(), py.ravel()), fields=fields)
    if D0 is None or len(D0)==0:
        return [None]
    # check the D6 filenames against the blacklist of bad files
    if tiled:
        D0=reconstruct_tracks(point_data(list_of_fields=fields).from_list(D0))
    blacklist=None
    D1=list()
    for ind, D in enumerate(D0):
        if D.size<2:
            continue
        delete_file, blacklist=check_rgt_cycle_blacklist(rgt_cycle=[D.rgt[0], D.cycle_number[0]],  blacklist=blacklist)
        if delete_file==0:
            D1.append(D)

    # D1 is now a filtered version of D0
    D_pt=point_data().from_list(D1)
    bin_xy=1.e4*np.round((D_pt.x+1j*D_pt.y)/1.e4)
    cplx_bins=[]
    for xy0 in np.unique(bin_xy):
        ii=np.flatnonzero(bin_xy==xy0)
        if np.mean(D_pt.atl06_quality_summary[ii]==0) < cplx_accept_threshold:
            cplx_bins+=[xy0]
    cplx_bins=np.array(cplx_bins)

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
              
        # rename the h_li field to 'z', and rename the 'matlab_time' to 'day'
        D.assign({'z': D.h_li+D.dac,'time':D.matlab_time,'sigma':D.h_li_sigma,'sigma_corr':D.sigma_geo_h,'cycle':D.cycle_number})
        D.assign({'year':D.delta_time/24./3600./365.25+2018})
        # thin to 40 m
        D.index(np.mod(D.segment_id, 2)==0)  
        if 'x' not in D.list_of_fields:
            D.get_xy(SRS_proj4)
        #D.index(np.isfinite(D.h_li), list_of_fields=['x','y','z','time','year','sigma','sigma_corr','rgt','cycle'])
                
        D.assign({'sensor':np.zeros_like(D.x)+sensor})
        D1[ind]=D.subset(np.isfinite(D.h_li), datasets=['x','y','z','time','delta_time','year','sigma','sigma_corr','rgt','cycle','sensor', 'BP','LR'])
    return D1

def main():
    gI_file='/Volumes/insar10/ben/IS2_tiles/GL/GeoIndex.h5'
    xy0=[-170000.0, -2280000.0]
    dI=point_data().from_list(read_ICESat2(xy0, {'x':2.e4, 'y':2.e4}, gI_file, cplx_accept_threshold=0.25))
    import matplotlib.pyplot as plt
    #plt.plot(dI.x, dI.y,'ro')
    #dI=dI=point_data().from_list(read_ICESat2(xy0, {'x':2.e4, 'y':2.e4}, gI_file, cplx_accept_threshold=0))
    #plt.plot(dI.x, dI.y,'kx')
    #plt.axis('equal')
    plt.scatter(dI.x, dI.y, c=dI.z, linewidth=0); plt.colorbar()
    
    
if __name__=='__main__':
    main()