#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:26:23 2022

@author: ben
"""

import numpy as np
from LSsurf.fd_grid import fd_grid
from LSsurf.lin_op import lin_op
from LSsurf.unique_by_rows import unique_by_rows
import pointCollection as pc

'''  Functions to assign and manipulate biase estimates for surface fits

    Contains:
        assign_bias_ID
        setup_bias_fit
        parse_biases
'''



def assign_bias_ID(data, bias_params=None, bias_name='bias_ID', key_name=None, bias_filter=None, bias_model=None):
    """
    Assign a value to each data point that determines which biases are applied to it.

    Parameters:
        data: pointCollection.data instance
        bias_parameters: a list of parameters, each unique combination of which
          defines a different bias
        bias_name: a name for the biases
        key_name: an optional parameter which will be used as the dataset name,
          otherwise a key will be built from the parameter values
        bias_model: a dict containing entries:
            E_bias: a dict of expected bias values for the each biasID, determined from the sigma_corr parameter of the data
            bias_ID_dict: a dict giving the parameter values for each bias_ID (or the key_name if provided)
            bias_param_dict: a dict giving the mapping from parameter values to bias_ID values
    """
    if bias_model is None:
        bias_model={'E_bias':dict(), 'bias_param_dict':dict(), 'bias_ID_dict':dict()}
    bias_ID=np.zeros(data.size)+-9999
    p0=len(bias_model['bias_ID_dict'].keys())
    if bias_params is None:
        # assign all data the same bias
        bias_model['bias_ID_dict'][p0+1]=key_name
        bias_ID=p0+1
        bias_model['E_bias'][p0+1]=np.nanmedian(data.sigma_corr)
        data.assign({bias_name:bias_ID})
        return data, bias_model
    bias_ID=np.zeros(data.size)
    if bias_filter is not None:
        data_filt=bias_filter(data)
    else:
        data_filt=data
    temp=np.column_stack([getattr(data_filt, bp) for bp in bias_params])
    # set NaN param values to -9999 so that they are put into the same category
    # Could update pc.unique_by_rows to do this
    NDV=-9999
    temp=np.nan_to_num(temp, nan=NDV)
    u_p, this_param_dict=pc.unique_by_rows(temp, return_dict=True)
    bias_model['bias_param_dict'].update({param:list() for param in bias_params})
    bias_model['bias_param_dict'].update({'ID':list()})
    for p_num, param_vals in enumerate(u_p):
        this_ind=this_param_dict[tuple(param_vals)]
        # report invalid parameter values as NaN
        out_param_vals=param_vals.copy()
        out_param_vals[out_param_vals==NDV] = np.nan
        param_vals_dict={}
        #Identify the data that match the parameter values
        for i_param, param in enumerate(bias_params):
            param_vals_dict[param] = out_param_vals[i_param]
            #this_name += '%s%3.2f' % (param, param_vals[i_param])
            bias_model['bias_param_dict'][param].append(param_vals[i_param])
        bias_model['bias_param_dict']['ID'].append(p0+p_num)
        bias_ID[this_ind]=p0+p_num
        bias_model['bias_ID_dict'][p0+p_num]=param_vals_dict
        bias_model['E_bias'][p0+p_num]=np.nanmedian(data.sigma_corr[this_ind])
    data.assign({bias_name:bias_ID})
    return data, bias_model

def setup_bias_fit(data, bias_model, G_data, constraint_op_list,
                       bias_param_name='data_bias', op_name='data_bias'):
    """
        Setup a set of parameters representing the biases of a set of data

        input arguments:
             data: data for the problem.  Must contain a parameter with the name specified in 'bias_param_name
             bias_model: bias_model dict from assign_bias_params
             G_data: coefficient matrix for least-squares fit.  New bias parameters will be added to right of existing parameters
             constraint_op_list: list of constraint-equation operators
             bias_param_name: name of the parameter used to assign the biases.  Defaults to 'data_bias'
             op_name: the name for the output bias operator.
    """
    # the field in bias_param_name defines the relative column in the bias matrix
    # for the DOF constrained
    col=getattr(data, bias_param_name).astype(int)
    # the new columns are appended to the right of G_data
    col_0=G_data.col_N
    col_N=G_data.col_N+np.max(col)+1
    # The bias matrix is just a 1 in the column for the bias parameter for each
    # data value
    G_bias=lin_op(name=op_name, col_0=col_0, col_N=col_N).data_bias(np.arange(data.size), col=col_0+col)
    # the constraint matrix has a 1 for each column, is zero otherwise
    ii=np.arange(col.max()+1, dtype=int)
    Gc_bias=lin_op(name='constraint_'+op_name, col_0=col_0, col_N=col_N).data_bias(ii,col=col_0+ii)
    for key in bias_model['bias_ID_dict']:
        bias_model['bias_ID_dict'][key]['col']=col_0+key
    # the confidence for each bias parameter being zero is in bias_model['E_bias']
    Gc_bias.expected=np.array([bias_model['E_bias'][ind] for ind in ii])
    if np.any(Gc_bias.expected==0):
        raise(ValueError('found an zero value in the expected biases'))
    constraint_op_list.append(Gc_bias)
    G_data.add(G_bias)


def parse_biases(m, bias_model, bias_params, data=None):
    """
        parse the biases in the ouput model

        inputs:
            m: model vector
            bias_model: the bias model
            bias_params: a list of parameters for which biases are calculated
        output:
            b_dict: a dictionary giving the parameters and associated bias values for each ibas ID
            slope_bias_dict:  a dictionary giving the parameters and assicated biase values for each slope bias ID
    """
    slope_bias_dict={}
    b_dict={param:list() for param in bias_params+['val','ID','expected']}
    if data is not None:
        b_dict['rms_data_raw']=[]
        b_dict['rms_data_edited']=[]
        b_dict['N_raw']=[]
        b_dict['N_edited']=[]
        r = data.z-data.z_est
    # loop over the keys in bias_model['bias_ID_dict']
    for item in bias_model['bias_ID_dict']:
        b_dict['val'].append(m[bias_model['bias_ID_dict'][item]['col']])
        b_dict['ID'].append(item)
        b_dict['expected'].append(bias_model['E_bias'][item])
        for param in bias_params:
            b_dict[param].append(bias_model['bias_ID_dict'][item][param])
        if data is not None:
            jj = data.bias_ID==item
            r1=r[jj]
            if len(r1)==0:
                b_dict['rms_data_raw'].append(np.nan)
            else:
                b_dict['rms_data_raw'].append(np.nanstd(r1))
            b_dict['N_raw'].append(len(r1))
            r2 = r[jj & (data.three_sigma_edit==1)]
            b_dict['N_edited'].append(len(r2))
            if len(r2)==0:
                b_dict['rms_data_edited'].append(np.nan)
            else:
                b_dict['rms_data_edited'].append(np.nanstd(r2))

    if 'slope_bias_dict' in bias_model:
        for key in bias_model['slope_bias_dict']:
            slope_bias_dict[key]={'slope_x':m[bias_model['slope_bias_dict'][key][0]], 'slope_y':m[bias_model['slope_bias_dict'][key][1]]}

    return b_dict, slope_bias_dict
