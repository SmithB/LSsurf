#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 18:45:32 2019

@author: ben
"""


import numpy as np
import xarray as xa

import scipy.interpolate as si
import pyproj 
import matplotlib.pyplot as plt

def clip(x, interval):
    if np.any(x< interval[0]):
        x[x<interval[0]]=interval[0]
    if np.any(x > interval[1]):
        x[x>interval[1]]=interval[1]
    return 

def interp_MAR_firn(x, y, year):

    firn_file='/Volumes/ice2/ben/MAR/MARv3.9_Greenland_Daily/MAR-GRq-ERA-7_5km_firnAir.nc'
    season_file='/Volumes/ice2/ben/MAR/MARv3.9_Greenland_Daily/MAR-GRq-ERA-7_5km_season_h_anomaly.nc'

    dh=np.zeros_like(x)+np.NaN
    P_ll=pyproj.Proj(init='EPSG:4326')
    P_ps=pyproj.Proj(init='EPSG:3413')
    t0=1980.0
    # read the firn file
    with xa.open_dataset(firn_file) as fd:
        LON=np.array(fd['LON'][0,:,:])
        LAT=np.array(fd['LAT'][0,:,:])
        t_last=float(fd['year'][-1])
        in_bounds=year-t0 < t_last
        [xg, yg]=np.meshgrid(fd['X12_203'], fd['Y20_377'])
        LND=si.RegularGridInterpolator((fd['year'], fd['Y20_377'], fd['X12_203']), np.array(fd['h_anomaly']))
        if any(year-t0 > t_last):
            fh_end=np.array(fd['h_anomaly'][-1, :, :])
    # transform the input coordinares into model coordinates (use the lat and lon fields in the data file)
    x_ps, y_ps = pyproj.transform(P_ll, P_ps, LON.ravel(), LAT.ravel())
    ix=si.LinearNDInterpolator(np.c_[x_ps, y_ps], xg.ravel())(np.c_[x, y])
    iy=si.LinearNDInterpolator(np.c_[x_ps, y_ps], yg.ravel())(np.c_[x, y])
    good=np.isfinite(ix)
    these= in_bounds & good
    if np.any(these):
        dh[these]=LND.__call__((year[these]-t0, iy[these], ix[these]))
    
    # interpolate the  FDM data  
    if np.any(year-t0  > t_last):
        these=(~in_bounds) & good
        if np.any(these):
            with xa.open_dataset(season_file) as fa:
                LND=si.RegularGridInterpolator((fa['delta_time'], fa['Y20_377'], fa['X12_203']), np.array(fa['mean_dh'])+fh_end)
                t_temp=np.mod(year[these], 1)
                clip(t_temp, [float(np.min(fa['delta_time'])), float(np.max(fa['delta_time']))])               
                dh[these]=LND.__call__((t_temp, iy[these], ix[these]))
    return dh

def __main__():            
    # test function
    xy=(-142833.06277056271, -2224375.2705627708) 
    tt=np.arange(1980, 2019.5, 0.1)
    xx=np.zeros_like(tt)+xy[0]
    yy=np.zeros_like(tt)+xy[1]
    fh=interp_MAR_firn(xx,yy,tt)
    plt.plot(tt, fh)
    
