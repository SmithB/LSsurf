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
 
def read_ICESat2(xy0, W, gI_file, sensor=2, SRS_proj4=None, tiled=True, seg_diff_tol=2, blockmedian_scale=None):
    field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude','atl06_quality_summary','segment_id','sigma_geo_h'], 
                'fit_statistics':['dh_fit_dx'],
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

    for ind, D in enumerate(D1):
        try:
            segDifferenceFilter(D, setValid=False, toNaN=True, tol=seg_diff_tol)
        except TypeError as e:
            print("HERE")          
            print(e)
        D.h_li[D.atl06_quality_summary==1]=np.NaN
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