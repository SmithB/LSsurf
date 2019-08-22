#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:08:42 2019

@author: ben
"""

import numpy as np
from LSsurf.racmo_interp_firn_height import interpolate_racmo_firn
from LSsurf.racmo_extrap_firn_height import extrapolate_racmo_firn

from LSsurf.interp_MAR_firn import interp_MAR_firn

def assign_firn_correction(data, firn_correction, hemisphere):
    # if we're rereading data, it already has the firn correction applied
    if firn_correction == 'MAR':
        if hemisphere==1:
            data.assign({'h_firn':interp_MAR_firn(data.x, data.y, data.time)})
            data.z -= data.h_firn
    elif firn_correction == 'RACMO':
        if hemisphere==1:
            data.assign({'h_firn':interpolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN055', data.time, data.x, data.y, FILL_VALUE=np.NaN)[0]})
            h_firn=interpolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN055', data.time, data.x, data.y, FILL_VALUE=np.NaN)[0]
            bad=~np.isfinite(h_firn)        
            if np.any(bad):
                h_firn[bad]=extrapolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN055', data.time[bad], data.x[bad], data.y[bad], FILL_VALUE=np.NaN)[0]
        data.assign({'h_firn':h_firn})
        data.z -= data.h_firn
    elif firn_correction == "RACMO_fac":
        if hemisphere==1:
            data.assign({'fac':interpolate_racmo_firn('/Volumes/ice1/tyler', "EPSG:3413", 'FGRN11', data.time, data.x, data.y, FILL_VALUE=np.NaN)[1]})
            data.z -= data.fac
    data.index(np.isfinite(data.z))