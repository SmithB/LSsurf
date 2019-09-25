#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:08:42 2019

@author: ben
"""

import numpy as np
from LSsurf.racmo_interp_firn_height import interpolate_racmo_firn
from LSsurf.racmo_extrap_firn_height import extrapolate_racmo_firn
from LSsurf.racmo_interp_downscaled import interpolate_racmo_downscaled
from LSsurf.racmo_extrap_downscaled import extrapolate_racmo_downscaled

from LSsurf.interp_MAR_firn import interp_MAR_firn

def assign_firn_correction(data, firn_correction, hemisphere):
    # if we're rereading data, it already has the firn correction applied
    if firn_correction == 'MAR':
        if hemisphere==1:
            data.assign({'h_firn':interp_MAR_firn(data.x, data.y, data.time)})
            data.z -= data.h_firn
    elif firn_correction == 'RACMO':
        if hemisphere==1:
            h_firn=interpolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN055', data.time, data.x, data.y, VARIABLE='zs', FILL_VALUE=np.NaN)[0]
            # check if any data points are over "unglaciated" values
            bad=~np.isfinite(h_firn)
            if np.any(bad):
                # extrapolate model to coordinates
                h_firn[bad]=extrapolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN055', data.time[bad], data.x[bad], data.y[bad], VARIABLE='zs', FILL_VALUE=np.NaN)[0]
            data.assign({'h_firn':h_firn})
            data.z -= data.h_firn
    elif firn_correction in ["RACMO_fac", "RACMO_fac_smb", "RACMO_smb"]:
        if hemisphere==1 and firn_correction in ["RACMO_fac","RACMO_fac_smb"]:
            h_fac = interpolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN055', data.time, data.x, data.y, VARIABLE='FirnAir', FILL_VALUE=np.NaN)[0]
            # check if any data points are over "unglaciated" values
            bad=~np.isfinite(h_fac)
            if np.any(bad):
                # extrapolate model to coordinates
                h_fac[bad]=extrapolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN055', data.time[bad], data.x[bad], data.y[bad], VARIABLE='FirnAir', FILL_VALUE=np.NaN)[0]
            data.z -= h_fac
            data.assign({'fac':h_fac})
        else:
            h_fac=np.zeros_like(data.z)
        if firn_correction in ["RACMO_fac_smb","RACMO_smb"]:
            # ice density used in RACMO models (kg/m^3)
            rho_model = 910.0
            if hemisphere==1:
                smb = interpolate_racmo_downscaled('/Volumes/ice1/tyler', "EPSG:3413", '3.0', 'SMB', data.time, data.x, data.y, FILL_VALUE=np.NaN)[0]
                # check if any data points are over invalid downscaled values
                bad=~np.isfinite(smb)
                #if np.any(bad):
                    # extrapolate model to coordinates
                #    smb[bad]=extrapolate_racmo_downscaled('/Volumes/ice1/tyler', "EPSG:3413", '3.0', 'SMB', data.time[bad], data.x[bad], data.y[bad], FILL_VALUE=np.NaN)[0]
                data.assign({'smb':smb/rho_model, 'firn':h_fac+smb/rho_model})
                # calculate corrected height (convert smb from mmwe to m)
                data.z -= data.smb
    if firn_correction not in ['RACMO_smb']:
        data.index(np.isfinite(data.z))

def main():
    import matplotlib.pyplot as plt
    from PointDatabase import point_data
    t=np.arange(2002, 2022, 0.1)
    xy_jako=(-170000, -2280000)
    corrections = ['RACMO_fac', 'RACMO_fac_smb', 'RACMO']
    for correction in corrections:
        data=point_data().from_dict({'time':t, 'x':np.ones_like(t)*xy_jako[0], 'y':np.ones_like(t)*xy_jako[1], 'z':np.zeros_like(t)})
        assign_firn_correction(data, correction, 1)
        plt.plot(data.time, data.z)
    plt.legend(corrections)
if __name__=="__main__":
    main()

