#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:12:44 2019

@author: ben
"""

def matlab_to_year(t):
    # approximate conversion of matlab date to year.  Uses the matlab conversion
    # datestr('jan 1 2000') -> 730486

    return (t-730486.)/365.25+2000.
