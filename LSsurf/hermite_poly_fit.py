#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:03:42 2020

@author: ben
"""

import numpy as np
import scipy.sparse as sp

def assign_bins(t, t0):
    ind=np.searchsorted(t0.ravel(), t.ravel(), side='right')-1
    ind[t.ravel()==t0[-1]]=t0.size-2
    return ind
  
def hermite(u):
    coeffs=np.array([[2, -3, 0, 1], [1, -2, 1, 0], [-2, 3, 0, 0,], [1, -1, 0, 0,]]).T;
    h=sp.hstack([u.power(3), u.power(2), u, np.ones(u.shape)], format='csr').dot(coeffs)
    return h

def hermite_design_matrix(x_in, x0_in, c=0):

    theta=(1-c);

    x=x_in.ravel()
    valid=np.flatnonzero((x.ravel()>=x0_in[0]) & (x.ravel()<=x0_in[-1]))
    x=x[valid]
    x0=x0_in.ravel()

    dx0=np.diff(x0)
    c_herm=assign_bins(x, x0)
    r_herm=np.arange(x.size)
    x_rel=sp.coo_matrix(((x-x0[c_herm])/dx0[c_herm], (r_herm, c_herm))).tocsr()
    
    G=sp.coo_matrix((x_in.size, x0_in.size))
    # force the first segment to have zero second derivative at 0
    rows=np.flatnonzero(c_herm==0)
    if np.any(rows):
        h=hermite(x_rel[rows,0]); 
        P0=np.array([1, 0, 0]).reshape((1,3))
        P1=np.array([0, 1, 0]).reshape((1,3))
        P2=np.array([0, 0, 1]).reshape((1,3))
        m1=theta*0.5*(dx0[0]*(P2-P1)/dx0[1]+(P1-P0));
        m0=-3/2*P0+3/2*P1-1/2*m1;
        sh=(h.shape[0], 1)
        Gsub=h[:,0].reshape(sh).dot(P0)+h[:,1].reshape(sh).dot(m0) + h[:,2].reshape(sh).dot(P1) + h[:,3].reshape(sh).dot(m1);
        ind=np.where(Gsub)
        G += sp.coo_matrix((Gsub[ind], (valid[rows[ind[0]]], ind[1])), shape=G.shape)

    #force the last segment to have zero second derivative at 1
    rows=np.flatnonzero(c_herm>=len(x0)-2)
    if len(rows)>0:
        h=hermite(x_rel[rows,-1])
        Pm1=np.array([1, 0, 0]).reshape((1,3))
        P0=np.array([0, 1, 0]).reshape((1,3))
        P1=np.array([0, 0, 1]).reshape((1,3))
        m0=theta*0.5*(dx0[-1]*(P0-Pm1)/dx0[-2]+ P1-P0);
        m1=-3/2*P0-1/2*m0+3/2*P1;
        # this is the solution for P2 to give P''(end)=0    
        sh=(h.shape[0], 1)

        Gsub=h[:,0].reshape(sh).dot(P0)+h[:,1].reshape(sh).dot(m0) + h[:,2].reshape(sh).dot(P1) + h[:,3].reshape(sh).dot(m1);
        ind=np.where(Gsub)
        G += sp.coo_matrix((Gsub[ind], (valid[rows[ind[0]]],  x0.size-3+ind[1])), shape=G.shape)


    # fill in the interior of the interval
    Pm1=np.array([1, 0, 0, 0]).reshape((1,4))
    P0=np.array([0, 1, 0, 0]).reshape((1,4))
    P1=np.array([0, 0, 1, 0]).reshape((1,4))
    P2=np.array([0, 0, 0, 1]).reshape((1,4))


    for k in range(1, x0.size-2):
        rows = np.flatnonzero(c_herm==k)
        if len(rows) > 0:
            h=hermite(x_rel[rows,k])  
            m0=theta*0.5*(dx0[k]*(P0-Pm1)/dx0[k-1]+ P1-P0);
            m1=theta*0.5*((P1-P0)+dx0[k]*(P2-P1)/dx0[k+1]);
            sh=(h.shape[0],1)
            Gsub=h[:,0].reshape(sh).dot(P0)+h[:,1].reshape(sh).dot(m0) + h[:,2].reshape(sh).dot(P1) + h[:,3].reshape(sh).dot(m1);
            ind=np.where(Gsub)
            G += sp.coo_matrix((Gsub[ind], (valid[rows[ind[0]]], k+ind[1]-1)), shape=G.shape)
    return G


def test_spline():
    
    x0=np.arange(0., 110., 10.)
    y0=np.zeros_like(x0)+x0/10.
    y0[5]=1.
    y0[-1]=0.8
    y0[0]=5
    import matplotlib.pyplot as plt
    
    xx=np.arange(0, 101.)
    hh=hermite_design_matrix(xx, x0)
    yy=hh.dot(y0.reshape(( len(y0), 1)))
    plt.plot(x0, y0,'rs')
    plt.plot(xx, yy )
    plt.show()

if __name__=="__main__":
    test_spline()
    
