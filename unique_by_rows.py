# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:40:52 2018

@author: ben
"""

import numpy as np

def unique_by_rows(x, return_index=False, return_inverse=False):
    ind=np.zeros(x.shape[0])
    scale=1.
    for col in range(x.shape[1]):
        z, ii=np.unique(x[:,col], return_inverse=True)
        scale /= (np.max(ii).astype(float)+1.)
        ind+= ii * scale     
    u_ii, index, inverse=np.unique(ind, return_index=True, return_inverse=True)
    uX=x[index,:]
    if return_index is True and return_inverse is False:
        return uX, index
    elif return_index is False and return_inverse is True:
        return uX, inverse
    elif return_inverse is True and return_index is True:
        return uX, index, inverse
    else:
        return uX

            
    