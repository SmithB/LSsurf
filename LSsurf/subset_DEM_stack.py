#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:33:41 2019

@author: ben
"""
import numpy as np
from LSsurf.unique_by_rows import unique_by_rows
from LSsurf import matlab_to_year
import pointCollection as pc

def subset_DEM_stack(D0, xyc, W, bin_width=400, min_coverage_w=5e3):
    '''
    Find the minimal set of DEMS that cover a grid completely once per year.
    '''
    if isinstance(D0, list):
        D=pc.data(fields=['x','y','time','sensor','z']).from_list(D0)
    elif isinstance(D0, pc.data):
        D=D0
    else:
        raise TypeError('type of D0 not understood')
    if np.any(D.time > 1e4):
        D.time=matlab_to_year(D.time)
    xy0=[xyc[0]-W/2, xyc[1]-W/2]
    N_bins=W/bin_width + 1
    row=np.round((D.y-xy0[1])/bin_width).astype(int)
    col=np.round((D.x-xy0[0])/bin_width).astype(int)
    cal_year=np.floor(D.time).astype(int)
    # find the unique combinations of rows, columns, and years.  rcy_dict is a dictionary containing the
    # data indices for each combination
    rcy, rcy_dict=unique_by_rows(np.c_[row, col, cal_year], return_dict=True)
    sensor_stats={}
    # for each row, column, and year, find the median height, and calculate the scaled difference
    # between the data from each sensor and the median (difference / sigma)
    for key in rcy_dict:
        year=key[2]
        # skip the bin if it's not in bounds
        if key[0] < 0 or key[0] > N_bins or key[1] < 0 or key[1] > N_bins:
            continue
        if year not in sensor_stats:
            sensor_stats[year]={}
        z0=np.nanmedian(D.z[rcy_dict[key]])
        these_z=D.z[rcy_dict[key]]
        these_sensors=D.sensor[rcy_dict[key]].astype(int)
        # calculate the MAD for data in the bin
        zs=np.maximum(np.nanmedian(np.abs(D.z[rcy_dict[key]]-z0)), 0.5)
        for sensor in np.unique(these_sensors):
            if sensor not in sensor_stats[year]:
                sensor_stats[year][sensor]={'devs':[], 'pts':[]}
            # append the scaled differences for the current sensor
            sensor_stats[year][sensor]['devs'] += (np.abs(these_z[these_sensors==sensor]-z0)/zs).tolist()
            # save the bin locations for each sensor
            sensor_stats[year][sensor]['pts'] += [(key[0], key[1])]
    best_sensors=[]
    for year in sensor_stats.keys():
        sensors=np.array(list(sensor_stats[year].keys()))
        score = np.zeros(len(sensors))
        #zero_count = np.zeros(len(sensors))
        for ii, sensor in enumerate(sensors):
            # the score for each sensor is the sum of the inverse differences from the median
            # so being close the median is good, and more points is good
            score[ii] = np.sum(1./np.maximum(np.array(sensor_stats[year][sensor]['devs']), 0.25))
        bins=[]
        # loop over sensors from worst to best
        for sensor in sensors[np.argsort(score)[::-1]]:
            # see which bins a sensor covers that haven't been covered already
            new_bins = [ ii for ii in sensor_stats[year][sensor]['pts'] if ii not in bins]
            new_bins_count = len(new_bins)
            # if the sensor covers more than (min_coverage_width/bin_dist)^2 new bins, use it
            if new_bins_count > (min_coverage_w/bin_width)**2:
                best_sensors += [int(sensor)]
                bins += new_bins
        #year_best_sensors = sensors[np.argsort(score)[-max_per_year:]].tolist()
        #year_best_sensors += sensors[zero_count > (5.e3/bin_width)**2].tolist()
        #best_sensors += np.unique(year_best_sensors).tolist()
    return best_sensors

def main():
    # test code
    thefile='Volumes/ice2/ben/ATL14_test/Jako_d2zdt2=5000_d3z=0.00001_d2zdt2=1500_RACMO//centers/E480_N-1200.h5'
    D=pc.data().from_h5(thefile, group='/data/')
    D.index(D.sensor>=4)
    xy0=np.array([  480000., -1200000.])
    best_sensors=subset_DEM_stack(D, xy0, 4.e4)


    