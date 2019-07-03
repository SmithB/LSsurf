#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:35:41 2019

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
from PointDatabase.mapData import mapData, point_data
import os

def tile_map(xy0, thedir, delta_x=2.e4, t_slice=[0, -1]):
    xyr=np.round(np.array(xy0)/delta_x)*delta_x
    file='%s/E%d_N%d.h5' % (thedir, xyr[0]/1000, xyr[1]/1000)
    if not os.path.isfile(file):
        print("%s not found"%file)
        return
    dz=mapData().from_h5(file, group='/dz/', field_mapping={'z':'dz'})
    h1=plt.subplot(221)
    h_map=plt.imshow((dz.z[:,:,t_slice[1]]-dz.z[:,:,t_slice[0]]), extent=dz.extent,  label=file, origin='lower')
    
    plt.subplot(222, sharex=h1, sharey=h1)
    D=point_data().from_file(file, group='/data/')
    D1=D.subset(D.three_sigma_edit>0.5)
    h_data=plt.scatter(D1.x, D1.y,c=D1.time, linewidth=0, markersize=2); plt.colorbar()
    
    plt.subplot(223, sharex=h1, sharey=h1)
    h_r=plt.scatter(D1.x, D1.y, c=r, linewidth=0, markersize=2); plt.colorbar()
    
    plt.subplot(224)
    plt.plot(dz.t, np.mean(np.mean(dz.z, axis=0), axis=0))
    return h_map, h_data

    

    
    

