#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:43:14 2022

@author: ben
"""
import numpy as np
import scipy.optimize as scipyo
from LSsurf.RDE import RDE


def calc_sigma_extra(r, sigma, mask, sigma_extra_masks=None):
    '''
    calculate the error needed to be added to the data to achieve RDE(rs)==1

    Parameters
    ----------
    r : numpy array
        model residuals
    sigma : numpy array
        estimated errors
    sigma_extra_masks: dict, optional
        masks into which to collect data before calculating sigma_extra
    Returns
    -------
    sigma_extra.

    '''
    if sigma_extra_masks is None:
        sigma_extra_masks = {'all': np.ones_like(r, dtype=bool)}
    sigma_extra=np.zeros_like(r)
    for key, ii in sigma_extra_masks.items():
        if np.sum(ii & mask) < 10:
            continue
        this_r=r[ii & mask]
        this_sigma=sigma[ii & mask]
        sigma_hat=RDE(this_r)
        sigma_aug_minus_1_sq = lambda sigma1: (RDE(this_r/np.sqrt(sigma1**2+this_sigma**2))-1)**2
        sigma_extra[ii]=scipyo.minimize_scalar(sigma_aug_minus_1_sq, method='bounded', bounds=[0, sigma_hat])['x']
    return sigma_extra


def calc_sigma_extra_on_grid(x, y, r, sigma, in_TSE, sigma_extra_masks=None, 
                             spacing=1.e4, L_avg=None, sigma_extra_max=None):
    ''' calculate sigma_extra based on an overlapping grid of points
    

    Parameters
    ----------
    x, y : numpy arrays
        data x and y coordinates
    r : numpy array
        model residuals
    sigma : numpy array
        estimated errors
    in_TSE : numpy array, boolean
        DESCRIPTION.
    sigma_extra_masks: dict, optional
        masks into which to collect data before calculating sigma_extra. If not
        specified, all data will be collected together
    spacing : float, optional
        spacing of bin centers into which to collect the data. 
        The default is 1.e4.
    L_avg : float, optional
        width of bins into which to collect the data. If None, 2*spacing is used
        The default is None
    sigma_extra_max: float, optional
        the maximum value for sigma_extra.  
        The default is None

    Returns
    -------
    sigma_extra.

    '''
    
    if L_avg is None:
        L_avg = 2*spacing
    
    sigma_extra_full = calc_sigma_extra(r, sigma, in_TSE, sigma_extra_masks=sigma_extra_masks)
    
    uXY = np.unique(np.round((x+1j*y)/spacing)*spacing)
    
    sum_W_sigma=np.zeros_like(r)
    sum_W = np.zeros_like(r)
    
    for xyi in uXY:
        dx = np.abs(np.real(xyi)-x)
        dy = np.abs(np.imag(xyi)-y)
        in_bin = (dx < L_avg/2) & (dy < L_avg/2)
        these = in_TSE & in_bin
        if np.sum(these) < 10:
            continue
        this_sigma_extra = calc_sigma_extra(r, sigma, these, sigma_extra_masks=sigma_extra_masks)
        if sigma_extra_max is not None:
            this_sigma_extra = np.minimum(this_sigma_extra, sigma_extra_max)
        W = in_bin * (1-dx/(L_avg/2)) * (1-dy/(L_avg/2))
        sum_W_sigma += W*this_sigma_extra
        sum_W += W
    
    sigma_extra=np.zeros_like(r) + sigma_extra_full
    these=sum_W > 0
    sigma_extra[these] = sum_W_sigma[ these ]/sum_W[ these ]

    return sigma_extra
