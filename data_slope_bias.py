#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:26:03 2019

@author: ben
"""
import lin_op
import numpy as np

def data_slope_bias(data,  bias_model, col_0=0, sensors=[], op_name='data_slope'):
    """
        Make a matrix that adds a set of parameters representing the biases of a set of data.

        input arguments:
             data: data for the problem.  Must contain parameters 'x' and 'y'
             bias_model: bias_model dict from assign_bias_params
             col_0: the first column of the matrix.
             op_name: the name for the output bias operator.
         output_arguments:
             G_bias: matrix that gives the biases for each parameter
             Gc_bias: matrix that gives the bias values (constraint matrix)
             E_bias: expected value for each bias parameter
             bias_model: bias model dict as defined in assign_bias_ID
    """
    if ['slope_bias_dict'] not in bias_model:
        bias_model['slope_bias_dict']={}

    col_N=col_0+2*len(sensors)
    rr, cc, vv=[[], [], []]

    for d_col_1, sensor in enumerate(sensors):
        rows=np.flatnonzero(data.sensor==sensor)
        bias_model['slope_bias_dict'][sensor]=col_0+np.array([0, 1])
        for d_col_2, var in enumerate('x', 'y'):
            delta=getattr(data, var)
            delta -= np.nanmean(delta)
            rr += [rows]
            cc += [np.ones_like(rows, dtype=int) + int(col_0+d_col_1+d_col_2)]
            vv += [delta]

    G_bias = lin_op(col_0=col_0, col_N=col_N)
    G_bias.r = np.concatenate(rr)
    G_bias.c = np.concatenate(cc)
    G_bias.v = np.concatenate(vv)

    ii=np.arange(col_0, col_N, dtype=int)
    Gc_bias=lin_op(name='constraint_'+op_name, col_0=col_0, col_N=col_N).data_bias(ii,col=col_0+ii)
    E_bias=bias_model['E_bias_slope']+np.zeros(2*len(sensors))

    return G_bias, Gc_bias, E_bias, bias_model