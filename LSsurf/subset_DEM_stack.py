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
import pdb

def make_bin_stats(D, xyc, W, bin_width, year_offset, pad_z0=True):
    xy0=[xyc[0]-W/2, xyc[1]-W/2]
    N_bins=W/bin_width + 1
    row=np.round((D.y-xy0[1])/bin_width).astype(int)
    col=np.round((D.x-xy0[0])/bin_width).astype(int)
    # apply year_offset (allows time bins centered on half years in Antarctica, avoids time-adjacent DEMs)
    cal_year=np.floor(D.time - year_offset).astype(int)
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
        bin_z=D.z[rcy_dict[key]]
        z_aug = bin_z.copy()
        if pad_z0:
            for offset in [-1, 0, 1]:
                this_key = (key[0], key[1], year+offset)
                if this_key in rcy_dict:
                    z_aug = np.concatenate([z_aug, D.z[rcy_dict[this_key]]])
        z0=np.nanmedian(z_aug)
        bin_sensors=D.sensor[rcy_dict[key]].astype(int)
        # calculate the MAD for data in the bin
        zs=np.maximum(np.nanmedian(np.abs(z_aug-z0)), 0.5)
        for sensor in np.unique(bin_sensors):
            if sensor not in sensor_stats[year]:
                sensor_stats[year][sensor]={'devs':[], 'pts':[]}
            # append the scaled differences for the current sensor
            sensor_stats[year][sensor]['devs'] += [np.median(np.abs(bin_z[bin_sensors==sensor]-z0)/zs)]#.tolist()
            # save the bin locations for each sensor
            sensor_stats[year][sensor]['pts'] += [(key[0], key[1])]
    return sensor_stats

def select_DEMs(sensor_stats, best_sensors, min_coverage_w, max_overlap_w, bin_width):
    sensors=np.array(list(sensor_stats.keys()))

    #zero_count = np.zeros(len(sensors))
    unchecked_sensors = list(sensor_stats.keys())
    for sensor in sensor_stats:
        bij = np.r_[sensor_stats[sensor]['pts']]
        bij = bij[:,0]+1j*bij[:,1]
        sensor_stats[sensor]['b_ij'] = bij
        sensor_stats[sensor]['free'] =np.arange(len(bij), dtype=int)

    bins=np.array([])
    num_unchecked_sensors = len(unchecked_sensors)
    sensor_total_bins = {sensor: len(sensor_stats[sensor]['devs']) for sensor in unchecked_sensors}
    while num_unchecked_sensors > 0:
        # count the unchecked bins:
        for ii, sensor in enumerate(unchecked_sensors.copy()):


            sensor_stats[sensor]['free'] = sensor_stats[sensor]['free'][~np.in1d(sensor_stats[sensor]['b_ij'][ sensor_stats[sensor]['free']], bins)]
            if len(sensor_stats[sensor]['free']) < (min_coverage_w/bin_width)**2:
                unchecked_sensors.remove(sensor)
        score = np.zeros(len(unchecked_sensors))
        for ii, sensor in enumerate(unchecked_sensors):
            # the score for each sensor is the sum of the inverse differences from the median
            # so being close the median is good, and more points is good
            new_devs = [ sensor_stats[sensor]['devs'][ii] for ii in sensor_stats[sensor]['free'] ]
            score[ii] = np.sum(1./np.maximum(np.array(new_devs), 0.25))
        #pdb.set_trace()
        # loop over sensors from best to worst
        for sensor in np.array(unchecked_sensors)[np.argsort(score)[::-1]]:
            # see which bins a sensor covers that haven't been covered already
            new_bins_count = len(sensor_stats[sensor]['free'])
            old_bins_count = sensor_total_bins[sensor] - new_bins_count

            # if the sensor covers more than (min_coverage_width/bin_dist)^2 new bins, and it
            # covers fewer more than (max_overlap_w/bin_width)**2 old bins, use it.
            unchecked_sensors.remove(sensor)
            if new_bins_count > (min_coverage_w/bin_width)**2 and old_bins_count < (max_overlap_w/bin_width)**2:
                best_sensors += [int(sensor)]
                bins = np.concatenate([bins, sensor_stats[sensor]['b_ij'][sensor_stats[sensor]['free']]])
                break
        #print(unchecked_sensors, flush=True)

        num_unchecked_sensors = len(unchecked_sensors)

def subset_DEM_stack(D0, xyc, W, bin_width=400, min_coverage_w=5e3, max_overlap_w=1.e4, year_offset=0 ):
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


    sensor_stats = make_bin_stats(D, xyc, W, bin_width, year_offset)

    best_sensors=[]
    for year in sensor_stats.keys():
        select_DEMs(sensor_stats[year], best_sensors, min_coverage_w, max_overlap_w, bin_width)

    return best_sensors

def main():
    # test code
    thefile='Volumes/ice2/ben/ATL14_test/Jako_d2zdt2=5000_d3z=0.00001_d2zdt2=1500_RACMO//centers/E480_N-1200.h5'
    D=pc.data().from_h5(thefile, group='/data/')
    D.index(D.sensor>=4)
    xy0=np.array([  480000., -1200000.])
    best_sensors=subset_DEM_stack(D, xy0, 4.e4)
