#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:34:58 2020

@author: ben
"""

import numpy as np
def op_field_rc(G, row_field, col_fields):
    """

    Parameters
    ----------
    G : linear operator (as defined in LSsurf.lin_op)
        Matrix to check
    row_field : string
        Fieldname to check
    col_fields : list of strings
        Fieldnames for the columns into which the column entries of G[rows,:]
        must fall, where rows = G.TOC[row_field]

    Returns
    -------
    Boolean
        True if all entries in G[rows,:] fall in columns designated by 
        col_fields
    """


    tr=G.TOC['rows']
    tc=G.TOC['cols']
    rows=tr[row_field]
    cols=np.concatenate([tc[field].ravel() for field in col_fields])
    ri, ci = G.toCSR()[rows,:].nonzero()
    return np.all(np.in1d(np.unique(ci), cols))

def all_row(G):
    """
    Check if every row of G contains a nonzero entry

    Parameters
    ----------
    G : lin_op
        operator to check

    Returns
    -------
    Boolean
        True if every row of G contains a nonzero entry

    """
    ri, ci = G.toCSR().nonzero()
    return np.all(np.in1d(np.arange(G.shape[0]), np.unique(ri)))

def all_col(G):
    """
    Check if every column of G contains a nonzero entry

    Parameters
    ----------
    G : lin_op
        operator to check

    Returns
    -------
    Boolean
        True if every column of G contains a nonzero entry

    """
    ri, ci = G.toCSR().nonzero()
    return np.all(np.in1d(np.arange(G.shape[1]), np.unique(ci)))

def complete_TOC_col(G, TOC_fields):
    """
    Check if fields in TOC_fields cover the columns of G

    Parameters
    ----------
    G : lin_op
        operator to check
    TOC_fields : list of strings
        List of fields that we expect to cover G exactly
    Returns
    -------
    Boolean: True if TOC_fields exactly cover the columns of G
    np.array: Number of fields that cover each column of G (should be all ones)

    """
    col_mask=np.zeros(G.shape[1])
    for field in TOC_fields:
        col_mask[G.TOC['cols'][field]] += 1
    return np.all(col_mask==1), col_mask

def complete_TOC_row(G, TOC_fields):
    """
    Check if fields in TOC_fields cover the columns of G

    Parameters
    ----------
    G : lin_op
        operator to check
    TOC_fields : list of strings
        List of fields that we expect to cover G exactly
    Returns
    -------
    Boolean: True if TOC_fields exactly cover the rows of G
    np.array: Number of fields that cover each row of G (should be all ones)

    """
    row_mask=np.zeros(G.shape[0])
    for field in TOC_fields:
        row_mask[G.TOC['rows'][field]] += 1
    return np.all(row_mask==1), row_mask
