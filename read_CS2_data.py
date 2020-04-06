#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:22:56 2019

@author: ben
"""

import pointCollection as pc
import numpy as np
import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
from LSsurf.matlab_to_year import matlab_to_year

def read_swath_data(xy0, W, index_file, apply_filters=True):
    fields = ['x','y','time','h', 'power','coherence','AD','error_composite',\
              'R_POCA', 'ambiguity','abs_orbit', 'block_h_spread',\
              'count','phase','R_offnadir','range_surf','seg_ind']

    D=pc.geoIndex().from_file(index_file).query_xy_box(xy0[0]+np.array([-W/2, W/2]), \
                   xy0[1]+np.array([-W/2, W/2]), fields=fields)
    D=pc.data().from_list(D)
    if D is None:
        return D

    D.time=matlab_to_year(D.time)
    #D.assign({'burst':D.pulse_num,
    D.assign({'swath':np.ones_like(D.x, dtype=bool)})

    D.assign({'sigma':np.minimum(5, np.maximum(1, 0.95 -.4816*(np.log10(D.power)+14) +\
                                       1.12*np.sqrt(D.block_h_spread)))})

    if apply_filters:
        if np.any(D.count>1):
            D.index( (D.power > 1e-17) & (D.power < 1e-13) & (D.error_composite==0) & \
                    (D.count > 3) & (D.block_h_spread < 15))
        else:
            D.index( (D.power > 1e-17) & (D.power < 1e-13) & (D.error_composite==0))
    return D

def read_poca_data(xy0, W, index_file, apply_filters=True, DEM=None):
    fields=['x','y','time','h', 'power','coherence','AD','error_composite', \
            'ambiguity','abs_orbit','phase','range_surf']
    D=pc.geoIndex().from_file(index_file).query_xy_box(xy0[0]+np.array([-W/2, W/2]), \
                   xy0[1]+np.array([-W/2, W/2]), fields=fields)
    D=pc.data().from_list(D)
    if D is None:
        return D
    D.time=matlab_to_year(D.time)
    #D.assign({'burst':D.pulse_num,
    D.assign({'swath':np.zeros_like(D.x, dtype=bool)})

    if DEM is not None:
        gx, gy = np.gradient(DEM.z, DEM.x, DEM.y)
        temp=DEM.copy()
        temp.z=np.abs(gx+1j*gy)
        DEM_slope_mag=temp.interp(D.x, D.y)
        D.assign({'sigma':np.maximum(0.5, 50*DEM_slope_mag + np.maximum(0, -0.64*(np.log10(D.power)+14)))})
    else:
        # if no DEM is specified, use a typical value of 0.01 for the slope
        D.assign({'sigma':50*0.01+ np.maximum(0, -0.64*(np.log10(D.power)+14))})

    if apply_filters:
        D.index((D.power > 1e-16) & (D.error_composite == 0) & (D.power < 1e-12))
    return D

def read_cs2_data(xy0, W, index_files, apply_filters=True, DEM_file=None, dem_tol=50):
    if DEM_file is not None:
        DEM=pc.grid.data().from_geotif(DEM_file, bounds=[xy0[0]+np.array([-W/2, W/2]), \
                                    xy0[1]+np.array([-W/2, W/2])])
    else:
        DEM=None

    D=[]
    D = [read_poca_data( xy0, W, file, apply_filters=apply_filters, DEM=DEM) \
         for file in index_files['POCA']]
    D += [ read_swath_data(xy0, W, file, apply_filters=apply_filters)\
          for file in index_files['swath']]
    D=pc.data().from_list(D)

    if DEM_file is not None:

        D.DEM=DEM.interp(D.x, D.y)
        D.index(np.abs(D.h-D.DEM) < dem_tol)

    if apply_filters:
        with open(os.path.join(Path(__file__).parent.absolute(), \
                     'CS2_known_bad_orbits.json'), 'r') as fh:
            bad_orbits=json.load(fh)[1]#['bad orbits']
        D.index( ~np.in1d(D.abs_orbit.astype(int), np.array(bad_orbits, dtype='int')))
    return D

def make_test_data():
    '''
    make test data for CS2 inversions

    inputs:  None (reads test_CS2_data.h5)
    outputs:
        z: simulated elevation data
        bounds: a dict containing the range of the x, y, and time data
        z_func: a function that gives heights as a function of x, y, time and
                (binary) swath
    '''

    data=pc.data().from_h5('CS2_test_data.h5', group='/')
    data.time=matlab_to_year(data.time)
    XR=np.array([np.floor(data.x.min()/1.e4), np.ceil(data.x.max())/1.e4])*1.e4
    YR=np.array([np.floor(data.y.min()/1.e4), np.ceil(data.y.max())/1.e4])*1.e4
    TR=np.array([np.floor(data.time.min()), np.ceil(data.time.max())])
    def z_func(x, y, t, swath, XR, YR, TR):
        if swath is None: 
            swath=np.zeros_like(x, dtype=bool)
        dem=(x-XR[0])*0.05+20*np.sin(2*np.pi*(y-YR[0])/(np.diff(YR)/10))
        t0=TR[0] + np.array([0, 1/3, 2/3, 1])*np.diff(TR)
        t_func=3*np.interp(t, t0, np.array([0, 0, 1, 1]))
        delta_z=2*t_func*np.sin(2*np.pi*(x-XR[0])/(np.diff(XR)/4))
        swath_bias=1.5+np.exp(-((x-np.mean(XR))**2+(y-np.mean(YR))**2)/2/(np.diff(XR)/3)**2)
        return dem+delta_z+swath_bias*swath

    return data, z_func(data.x, data.y, data.time, data.swath, XR, YR, TR), [XR, YR, TR], z_func


def test():
    index_files={'POCA':['/Volumes/insar6/ben/Cryosat/POCA_h5_C/AA_REMA_v1_sigma4/GeoIndex.h5', \
                         '/Volumes/insar6/ben/Cryosat/POCA_h5_C/AA_REMA_v1_sigma4_Jan2020/GeoIndex.h5'], \
                 'swath':['/Volumes/insar6/ben/Cryosat/SW_h5_C/AA_REMA_v1_sigma4/GeoIndex.h5', \
                          '/Volumes/insar6/ben/Cryosat/POCA_h5_C/AA_REMA_v1_sigma4/GeoIndex.h5'] }
    bin_center=     [-1400000.,  -450000.]
    D=read_cs2_data(bin_center, 7.1e4, index_files)
    fig=plt.figure();
    fig.add_subplot(3, 1, 1)
    plt.hist2d(D.x[D.swath==0], D.y[D.swath==0], 25)
    plt.axis('equal');
    plt.colorbar()
    fig.add_subplot(3, 1, 2)
    plt.hist2d(D.x[D.swath==1], D.y[D.swath==1], 25)
    plt.axis('equal');
    plt.colorbar()
    fig.add_subplot(3,1,3)
    plt.hist(D.time, 100)
    plt.show()
    return D

if __name__=='__main__':
    D=test()
