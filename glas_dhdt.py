#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:13:05 2019

@author: ben
"""

from PointDatabase.geo_index import geo_index
#from PointDatabase import point_data
from PointDatabase.matlabToYear import matlabToYear
from LSsurf.fd_grid import fd_grid
from LSsurf.lin_op import lin_op
from ATL11.RDE import RDE
import numpy as np
import matplotlib.pyplot as plt
from PySPQR import sparseqr
import scipy.sparse as sp
from time import time


def glas_fit(xy0=np.array((-150000, -2000000)), W0, D=None, E_RMS=None, gI=None, giFile='/Data/glas/GL/rel_634/GeoIndex.h5'):
    if gI is None:
        gI=geo_index().from_file(giFile)

    #import_D=False; print("WARNING::::::REUSING D")
    timing=dict()
    xy0=np.array((-150000, -2000000))
    E_RMS={'d2z0_dx2':20000./3000/3000, 'd3z_dx2dt':10./3000/3000, 'd2z_dxdt':100/3000, 'd2z_dt2':1}

    W={'x':W0, 'y':W0,'t':6}
    spacing={'z0':5.e2, 'dzdt':5.e3}

    args={'W':W, 'ctr':ctr, 'spacing':spacing, 'E_RMS':E_RMS, 'max_iterations':25}


    if D is None:
        fields=[ 'IceSVar', 'deltaEllip', 'numPk', 'ocElv', 'reflctUC', 'satElevCorr',  'time',  'x', 'y', 'z']
        ctr={'x':xy0[0], 'y':xy0[1], 't':(2003+2009)/2. }

        D=gI.query_xy_box(xy0[0]+np.array([-W['x']/2, W['x']/2]), xy0[1]+np.array([-W['y']/2, W['y']/2]), fields=fields)

        #plt.plot(xy[0], xy[1],'.')
        #plt.plot(xy0[0], xy0[1],'r*')

        D.assign({'year': matlabToYear(D.time)})
        good=(D.IceSVar < 0.035) & (D.reflctUC >0.05) & (D.satElevCorr < 1) & (D.numPk==1)
        D.subset(good, datasets=['x','y','z','year'])

        D.assign({'sigma':np.zeros_like(D.x)+0.2, 'time':D.year})
        plt.plot(D.x, D.y,'m.')

    bds={coord:args['ctr'][coord]+np.array([-0.5, 0.5])*args['W'][coord] for coord in ('x','y')}
    grids=dict()
    grids['z0']=fd_grid( [bds['y'], bds['x']], args['spacing']['z0']*np.ones(2), name='z0')
    grids['dzdt']=fd_grid( [bds['y'], bds['x']],  args['spacing']['dzdt']*np.ones(2), \
         col_0=grids['z0'].col_N+1, name='dzdt')

    valid_z0=grids['z0'].validate_pts((D.coords()[0:2]))
    valid_dz=grids['dzdt'].validate_pts((D.coords()))
    valid_data=valid_dz & valid_z0
    D=D.subset(valid_data)

    G_data=lin_op(grids['z0'], name='interp_z').interp_mtx(D.coords()[0:2])
    G_dzdt=lin_op(grids['dzdt'], name='dzdt').interp_mtx(D.coords()[0:2])
    G_dzdt.v *= (D.year[G_dzdt.r.astype(int)]-ctr['t'])
    G_data.add(G_dzdt)

    grad2_z0=lin_op(grids['z0'], name='grad2_z0').grad2(DOF='z0')
    grad_z0=lin_op(grids['z0'], name='grad_z0').grad(DOF='z0')
    grad2_dzdt=lin_op(grids['dzdt'], name='grad2_dzdt').grad2(DOF='dzdt')
    grad_dzdt=lin_op(grids['dzdt'], name='grad_dzdt').grad2(DOF='dzdt')
    Gc=lin_op(None, name='constraints').vstack((grad2_z0, grad_z0, grad2_dzdt, grad_dzdt))
    Ec=np.zeros(Gc.N_eq)
    root_delta_A_z0=np.sqrt(np.prod(grids['z0'].delta))
    Ec[Gc.TOC['rows']['grad2_z0']]=args['E_RMS']['d2z0_dx2']/root_delta_A_z0
    Ec[Gc.TOC['rows']['grad2_dzdt']]=args['E_RMS']['d3z_dx2dt']/root_delta_A_z0
    Ec[Gc.TOC['rows']['grad_z0']]=1.e4*args['E_RMS']['d2z0_dx2']/root_delta_A_z0
    Ec[Gc.TOC['rows']['grad_dzdt']]=1.e4*args['E_RMS']['d3z_dx2dt']/root_delta_A_z0

    Ed=D.sigma.ravel()

    N_eq=G_data.N_eq+Gc.N_eq

    # calculate the inverse square root of the data covariance matrix
    TCinv=sp.dia_matrix((1./np.concatenate((Ed, Ec)), 0), shape=(N_eq, N_eq))

    # define the right hand side of the equation
    rhs=np.zeros([N_eq])
    rhs[0:D.x.size]=D.z.ravel()

    # put the fit and constraint matrices together
    Gcoo=sp.vstack([G_data.toCSR(), Gc.toCSR()]).tocoo()
    cov_rows=G_data.N_eq+np.arange(Gc.N_eq)

    # initialize the book-keeping matrices for the inversion
    m0=np.zeros(Gcoo.shape[1])
    inTSE=np.arange(G_data.N_eq, dtype=int)

    for iteration in range(args['max_iterations']):
        # build the parsing matrix that removes invalid rows
        Ip_r=sp.coo_matrix((np.ones(Gc.N_eq+inTSE.size), (np.arange(Gc.N_eq+inTSE.size), np.concatenate((inTSE, cov_rows)))), shape=(Gc.N_eq+inTSE.size, Gcoo.shape[0])).tocsc()

        m0_last=m0
        # solve the equations
        tic=time();
        m0=sparseqr.solve(Ip_r.dot(TCinv.dot(Gcoo)), Ip_r.dot(TCinv.dot(rhs)));
        timing['sparseqr_solve']=time()-tic

        # quit if the solution is too similar to the previous solution
        if np.max(np.abs((m0_last-m0)[Gc.TOC['cols']['dzdt']])) < 0.05:
            break

        # calculate the full data residual
        rs_data=(D.z-G_data.toCSR().dot(m0))/D.sigma
        # calculate the robust standard deviation of the scaled residuals for the selected data
        sigma_hat=RDE(rs_data[inTSE])
        inTSE_last=inTSE
        # select the data that are within 3*sigma of the solution
        inTSE=np.where(np.abs(rs_data)<3.0*sigma_hat)[0]
        print('found %d in TSE, sigma_hat=%3.3f' % (inTSE.size, sigma_hat))
        if sigma_hat <= 1 or( inTSE.size == inTSE_last.size and np.all( inTSE_last == inTSE )):
            break
    m=dict()
    m['z0']=m0[Gc.TOC['cols']['z0']].reshape(grids['z0'].shape)
    m['dzdt']=m0[Gc.TOC['cols']['dzdt']].reshape(grids['dzdt'].shape)
    if DOPLOT:
        plt.subplot(121)
        plt.imshow(m['z0'])
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(m['dzdt'])
        plt.colorbar()

    if False:
        plt.figure()
        Dfinal=D.subset(inTSE)
        ii=np.argsort(Dfinal.z)
        plt.scatter(Dfinal.x[ii], Dfinal.y[ii], c=Dfinal.z[ii]); plt.colorbar()


    return grids, m, D, inTSE, sigma_hat

def fit_GL():
    giFile='/Data/glas/GL/rel_634/GeoIndex.h5'
    gI=geo_index().from_file(giFile, read_file=False)
