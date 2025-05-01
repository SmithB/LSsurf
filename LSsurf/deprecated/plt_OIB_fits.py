# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:21:04 2019

@author: ben
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

out_dir='/Volumes/ice2/ben/ATL14_test/d3z_dx2dt=1500m_over3x3km/d2zd2t/'
delta_z_20=list()
delta_z_2025=list()
leg_text=list()
for d2zdt2_val in [1000, 2000, 3000, 5000, 10000, 20000, 30000]:
    out_file=out_dir+'%2.1f_per_year2.h5' % d2zdt2_val
    #out_file='/Volumes/ice2/ben/ATL14_test/d3z_dx2dt/%dm_over_3x3km.h5' % (E_d3zdx2dt_val*3000*3000)
    with h5py.File(out_file,'r') as h5f:
        dz=h5f['/dz/dz']
        delta_z_20.append(dz[:, 15, -2]-dz[:, 15, 0])
        delta_z_2025.append(dz[20, 25,:])
        leg_text.append('scale=%d'% (d2zdt2_val))

        #leg_text.append('scale=%d'% (E_d3zdx2dt_val*3000*3000) )

        
plt.figure(142); plt.clf()
for dz in delta_z_2025:
    plt.plot(dz)
plt.legend(leg_text)

# 5K looks good for d2zdt2
# 1500 looks good for d3zdxdt

leg_list=list()
plt.figure(146); plt.clf()
out_dir='/Volumes/ice2/ben/ATL14_test/d3z_dx2dt=1500m_over3x3km/d2zd2t=5000_peryear2/d2z0dt2/';
# skip the 5000 value
for d2z0_dx2_val in [ 10000, 20000, 30000, 100000, 200000, 300000, 500000, 1000000 ]:
    out_file=out_dir+'/%fm_over_3km2.h5' % d2z0_dx2_val  
    with h5py.File(out_file,'r') as h5f:
        z0=np.array(h5f['/z0/z0'])
        xx=np.array(h5f['/z0/x'])
        xx=np.linspace(xx[0], xx[-1], z0.shape[1])
        yy=np.array(h5f['/z0/y'])
        yy=np.linspace(yy[0], yy[-1], z0.shape[0])
        plt.plot(np.linspace(yy[0], yy[-1], z0.shape[1]), z0[:, 50])
    leg_list.append('%d' % d2z0_dx2_val)
with h5py.File(out_file,'r') as h5f:
    x=np.array(h5f['/data/x'])
    y=np.array(h5f['/data/y'])
    z=np.array(h5f['/data/z'])
    s=np.array(h5f['/data/sensor'])
    v=np.array(h5f['/data/three_sigma_edit'])
iii=np.where((np.abs(x-xx[50])<250) & v==1)[0]
plt.plot(y[iii], z[iii],'k.')
leg_list.append('data')
plt.legend(leg_list)
# 200K is the smallest value that resolves some of the surface bumps.