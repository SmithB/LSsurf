#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:35:41 2019

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
from PointDatabase.mapData import mapData
from PointDatabase.point_data import point_data
import scipy.stats as sps
import os

def tile_map(file=None, xy0=[], thedir='.', delta_x=4.e4, t_slice=[0, -1]):
    hax={}
    if file is None:
        xyr=np.round(np.array(xy0)/delta_x)*delta_x
        file='%s/E%d_N%d.h5' % (thedir, xyr[0]/1000, xyr[1]/1000)
        print("file= %s" % file)
        if not os.path.isfile(file):
            print("%s not found"%file)
            return
    dz=mapData().from_h5(file, group='/dz/', field_mapping={'z':'dz'})
    hax['dz']=plt.subplot(221)
    plt.imshow((dz.z[:,:,t_slice[1]]-dz.z[:,:,t_slice[0]]), extent=dz.extent,  label=file, origin='lower')
    plt.colorbar()
    z0=mapData().from_h5(file, group='/z0/', field_mapping={'z':'z0'})
    
    hax['time']=plt.subplot(222, sharex=hax['dz'], sharey=hax['dz'])
    D=point_data().from_file(file, group='/data/')
    D1=D.subset(D.three_sigma_edit>0.5)
    plt.scatter(D1.x, D1.y, 2, c=D1.time, linewidth=0); plt.colorbar()
    
    hax['r']=plt.subplot(223, sharex=hax['dz'], sharey=hax['dz'])
    D1.assign({'r':D1.z-D1.z_est})
    order=np.argsort(np.abs(D1.r))
    
    plt.scatter(D1.x[order], D1.y[order], 4, c=D1.r[order], linewidth=0, 
                vmin=sps.scoreatpercentile(D1.r, 1), vmax=sps.scoreatpercentile(D1.r, 99)); plt.colorbar()
    
    hax['z_vs_t']=plt.subplot(224)
    D1.assign({'z0':z0.interp(D1.x, D1.y)})
    months=np.round(D1.time*12)/12
    zbar=[]; zsigma=[];
    zbar_nofirn=[]
    t_month=[]
    for month in np.unique(months):
        these=np.where(months==month)[0]
        if len(months) > 0:
            t_month.append(month)
            delta_z=D1.z[these]-D1.z0[these]
            zbar.append(np.nanmean(delta_z))
            zsigma.append(np.nanstd(delta_z))
            if 'h_firn' in D.list_of_fields:
                delta_z=delta_z+D1.h_firn[these]
                zbar_nofirn.append(np.nanmean(delta_z))
    
    t_mean=np.mean(np.mean(dz.z, axis=0), axis=0 )
    plt.plot(dz.t, t_mean, label='dz grid mean')
    plt.plot(dz.t[t_slice], t_mean[t_slice],'o', label='slices')
    plt.errorbar(np.unique(months), zbar, yerr=zsigma, fmt='o', label='data mean')
    if len(zbar_nofirn)> 0:
        plt.plot(np.unique(months), zbar_nofirn ,'*', label='data without firn')
    hl=plt.legend()
    return D1, hax

def main():
    xy0=np.array([  200000., -1484000.])
    #tile_map(xy0, '/home/ben/temp/', delta_x=4e4, t_slice=[11, 15])
    tile_map(xy0, '/home/ben/temp/', delta_x=4e4)

    
if __name__=='__main__':
    main()

