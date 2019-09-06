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
    elif firn_correction == "RACMO_fac":
        if hemisphere==1:
            h_fac = interpolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN055', data.time, data.x, data.y, VARIABLE='FirnAir', FILL_VALUE=np.NaN)[0]
            smb = racmo_interp_downscaled('/Volumes/ice1/tyler', "EPSG:3413", '3.0', 'SMB', data.time, data.x, data.y, FILL_VALUE=np.NaN)[0]
            # check if any data points are over "unglaciated" values
            bad=~np.isfinite(h_fac)
            if np.any(bad):
                # extrapolate model to coordinates
                h_fac[bad]=extrapolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN055', data.time[bad], data.x[bad], data.y[bad], VARIABLE='FirnAir', FILL_VALUE=np.NaN)[0]
            # check if any data points are over invalid downscaled values
            bad=~np.isfinite(smb)
            if np.any(bad):
                # extrapolate model to coordinates
                smb[bad]=racmo_extrap_downscaled('/Volumes/ice1/tyler', "EPSG:3413", '3.0', 'SMB', data.time[bad], data.x[bad], data.y[bad], FILL_VALUE=np.NaN)[0]
            data.assign({'fac':h_fac,'smb':smb})
            # ice density used in RACMO models (kg/m^3)
            rho_model = 910.0
            # calculate corrected height (convert smb from mmwe to m)
            data.z -= data.fac + data.smb/rho_model
    data.index(np.isfinite(data.z))
