#!/usr/bin/env python
u"""
racmo_interp_downscaled.py
Written by Tyler Sutterley (09/2019)
Interpolates and extrapolates downscaled RACMO products to times and coordinates

CALLING SEQUENCE:
    python racmo_interp_downscaled.py --directory=<path> --version=3.0 \
        --product=SMB,PRECIP,RUNOFF --coordinate=[-39e4,-133e4],[-39e4,-133e4] \
        --date=2016.1,2018.1

COMMAND LINE OPTIONS:
    -D X, --directory=X: Working data directory
    --version=X: Downscaled RACMO Version
        1.0: RACMO2.3/XGRN11
        2.0: RACMO2.3p2/XGRN11
        3.0: RACMO2.3p2/FGRN055
    --product: RACMO product to calculate
        SMB: Surface Mass Balance
        PRECIP: Precipitation
        RUNOFF: Melt Water Runoff
        SNOWMELT: Snowmelt
        REFREEZE: Melt Water Refreeze
    --coordinate=X: Polar Stereographic X and Y of point
    --date=X: Date to interpolate in year-decimal format
    --csv=X: Read dates and coordinates from a csv file
    --fill-value: Replace invalid values with fill value
        (default uses original fill values from data file)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        http://www.numpy.org
        http://www.scipy.org/NumPy_for_Matlab_Users
    scipy: Scientific Tools for Python
        http://www.scipy.org/
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

UPDATE HISTORY:
    Updated 09/2019: read subsets of DS1km netCDF4 file to save memory
    Written 09/2019
"""
from __future__ import print_function

import sys
import os
import re
import pyproj
import getopt
import netCDF4
import numpy as np
import scipy.spatial
import scipy.interpolate

#-- PURPOSE: read and interpolate downscaled RACMO products
def interpolate_racmo_downscaled(base_dir, EPSG, VERSION, PRODUCT, tdec, X, Y,
    FILL_VALUE=None):

    #-- Full Directory Setup
    DIRECTORY = 'SMB1km_v{0}'.format(VERSION)

    #-- netcdf variable names
    input_products = {}
    input_products['SMB'] = 'SMB_rec'
    input_products['PRECIP'] = 'precip'
    input_products['RUNOFF'] = 'runoff'
    input_products['SNOWMELT'] = 'snowmelt'
    input_products['REFREEZE'] = 'refreeze'
    #-- version 1 was in separate files for each year
    if (VERSION == '1.0'):
        RACMO_MODEL = ['XGRN11','2.3']
        VARNAME = input_products[PRODUCT]
        SUBDIRECTORY = '{0}_v{1}'.format(VARNAME,VERSION)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY, SUBDIRECTORY)
    elif (VERSION == '2.0'):
        RACMO_MODEL = ['XGRN11','2.3p2']
        var = input_products[PRODUCT]
        VARNAME = var if PRODUCT in ('SMB','PRECIP') else '{0}corr'.format(var)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY)
    elif (VERSION == '3.0'):
        RACMO_MODEL = ['FGRN055','2.3p2']
        var = input_products[PRODUCT]
        VARNAME = var if (PRODUCT == 'SMB') else '{0}corr'.format(var)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY)
    #-- input cumulative netCDF4 file
    args = (RACMO_MODEL[0],RACMO_MODEL[1],VERSION,PRODUCT)
    input_file = '{0}_RACMO{1}_DS1km_v{2}_{3}_cumul.nc'.format(*args)

    #-- convert projection from input coordinates (EPSG) to model coordinates
    proj1 = pyproj.Proj("+init={0}".format(EPSG))
    proj2 = pyproj.Proj("+init=EPSG:{0:d}".format(3413))
    ix,iy = pyproj.transform(proj1, proj2, X, Y)

    #-- Open the RACMO NetCDF file for reading
    fileID = netCDF4.Dataset(os.path.join(input_dir,input_file), 'r')
    #-- input shape of RACMO data
    nt = fileID[VARNAME].shape[0]
    #-- Get data from each netCDF variable and remove singleton dimensions
    d = {}
    #-- cell origins on the bottom right
    dx = np.abs(fileID.variables['x'][1]-fileID.variables['x'][0])
    dy = np.abs(fileID.variables['y'][1]-fileID.variables['y'][0])
    #-- x and y arrays at center of each cell
    d['x'] = fileID.variables['x'][:].copy() - dx/2.0
    d['y'] = fileID.variables['y'][:].copy() - dy/2.0
    #-- extract time (decimal years)
    d['TIME'] = fileID.variables['TIME'][:].copy()
    
    # choose a subset of model variables that span the input data
    xr = [ix.min()-dx, ix.max()+dx]
    yr = [iy.min()-dy, iy.max()+dy]
    cols = np.flatnonzero( (d['x'] >= xr[0]) & (d['x'] <= xr[1]) )
    rows = np.flatnonzero( (d['y'] >= yr[0]) & (d['y'] <= yr[1]) )
    ny = rows.size
    nx = cols.size
    #-- mask object for interpolating data
    d['MASK'] = np.array(fileID.variables['MASK'][rows, cols], dtype=np.bool)
    d['x'] = d['x'][cols]
    d['y'] = d['y'][rows]
    i,j = np.nonzero(d['MASK'])

    #-- check that input points are within convex hull of valid model points
    #xg,yg = np.meshgrid(d['x'],d['y'])
    #points = np.concatenate((xg[i,j,None],yg[i,j,None]),axis=1)
    #triangle = scipy.spatial.Delaunay(points.data, qhull_options='Qt Qbb Qc Qz')
    #interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
    #valid = (triangle.find_simplex(interp_points) >= 0)  
    # Check ix and iy against the bounds of d['x'] and d['y']
    valid = (ix >= d['x'].min()) & (ix <= d['x'].max()) & (iy >= d['y'].min()) & (iy <= d['y'].max())
    
    MI = scipy.interpolate.RegularGridInterpolator(
            (d['y'],d['x']), d['MASK'])
    # check valid points against the mask:
    valid[valid] = MI.__call__(np.c_[iy[valid],ix[valid]])
    
    #-- output interpolated arrays of variable
    interp_var = np.zeros_like(tdec,dtype=np.float)
    #-- type designating algorithm used (1: interpolate, 2: backward, 3:forward)
    interp_type = np.zeros_like(tdec,dtype=np.uint8)
    #-- interpolation mask of invalid values
    interp_mask = np.zeros_like(tdec,dtype=np.bool)

    #-- find days that can be interpolated
    if np.any((tdec >= d['TIME'].min()) & (tdec <= d['TIME'].max()) & valid):
        #-- indices of dates for interpolated days
        ind, = np.nonzero((tdec >= d['TIME'].min()) &
            (tdec <= d['TIME'].max()) & valid)
        #-- determine which subset of time to read from the netCDF4 file
        f = scipy.interpolate.interp1d(d['TIME'], np.arange(nt), kind='linear',
            fill_value=(0,nt-1), bounds_error=False)
        date_indice = f(tdec[ind]).astype(np.int)
        #-- months to read
        months = np.arange(date_indice.min(),np.minimum(date_indice.max()+2, d['TIME'].size))
        nm = len(months)
        #-- extract variable for months of interest
        d[VARNAME] = np.zeros((nm,ny,nx))
        for i,m in enumerate(months):
            d[VARNAME][i,:,:] = fileID.variables[VARNAME][m,rows,cols].copy()

        #-- create an interpolator for variable
        RGI = scipy.interpolate.RegularGridInterpolator(
            (d['TIME'][months],d['y'],d['x']), d[VARNAME])
        #-- create an interpolator for input mask
        #MI = scipy.interpolate.RegularGridInterpolator(
        #    (d['y'],d['x']), d['MASK'])

        #-- interpolate to points
        interp_var[ind] = RGI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        interp_mask[ind] = MI.__call__(np.c_[iy[ind],ix[ind]])
        #-- set interpolation type (1: interpolated)
        interp_type[ind] = 1

    #-- check if needing to extrapolate backwards in time
    count = np.count_nonzero((tdec < d['TIME'].min()) & valid)
    if (count > 0):
        #-- indices of dates before RACMO model
        ind, = np.nonzero((tdec < d['TIME'].min()) & valid)
        #-- set interpolation type (2: extrapolated backwards)
        interp_type[ind] = 2
        #-- calculate a regression model for calculating values
        #-- read first 10 years of data to create regression model
        N = 120
        #-- spatially interpolate variable to coordinates
        VAR = np.zeros((count,N))
        T = np.zeros((N))
        #-- spatially interpolate mask to coordinates
        mspl = scipy.interpolate.RectBivariateSpline(d['x'], d['y'],
            d['MASK'].T, kx=1, ky=1)
        interp_mask[ind] = mspl.ev(ix[ind],iy[ind])
        #-- create interpolated time series for calculating regression model
        for k in range(N):
            #-- time at k
            T[k] = d['TIME'][k]
            #-- spatially interpolate variable
            spl = scipy.interpolate.RectBivariateSpline(d['x'], d['y'],
                fileID.variables[VARNAME][k,rows,cols].T, kx=1, ky=1)
            #-- create numpy masked array of interpolated values
            VAR[:,k] = spl.ev(ix[ind],iy[ind])

        #-- calculate regression model
        for n,v in enumerate(ind):
            interp_var[v] = regress_model(T, VAR[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])

    #-- check if needing to extrapolate forward in time
    count = np.count_nonzero((tdec > d['TIME'].max()) & valid)
    if (count > 0):
        #-- indices of dates after RACMO model
        ind, = np.nonzero((tdec > d['TIME'].max()) & valid)
        #-- set interpolation type (3: extrapolated forward)
        interp_type[ind] = 3
        #-- calculate a regression model for calculating values
        #-- read last 10 years of data to create regression model
        N = 120
        #-- spatially interpolate variable to coordinates
        VAR = np.zeros((count,N))
        T = np.zeros((N))
        #-- spatially interpolate mask to coordinates
        mspl = scipy.interpolate.RectBivariateSpline(d['x'], d['y'],
            d['MASK'].T, kx=1, ky=1)
        interp_mask[ind] = mspl.ev(ix[ind],iy[ind])
        #-- create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            #-- time at k
            T[k] = d['TIME'][kk]
            #-- spatially interpolate variable
            spl = scipy.interpolate.RectBivariateSpline(d['x'], d['y'],
                fileID.variables[VARNAME][kk,rows, cols].T, kx=1, ky=1)
            #-- create numpy masked array of interpolated values
            VAR[:,k] = spl.ev(ix[ind],iy[ind])

        #-- calculate regression model
        for n,v in enumerate(ind):
            interp_var[v] = regress_model(T, VAR[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])

    #-- replace fill value if specified
    if FILL_VALUE:
        ind, = np.nonzero(~interp_mask)
        interp_var[ind] = FILL_VALUE
        fv = FILL_VALUE
    else:
        fv = 0.0

    #-- close the NetCDF files
    fileID.close()

    #-- return the interpolated values
    return (interp_var,interp_type,fv)

#-- PURPOSE: calculate a regression model for extrapolating values
def regress_model(t_in, d_in, t_out, ORDER=2, CYCLES=None, RELATIVE=None):

    #-- remove singleton dimensions
    t_in = np.squeeze(t_in)
    d_in = np.squeeze(d_in)
    t_out = np.squeeze(t_out)
    #-- check dimensions of output
    if (np.ndim(t_out) == 0):
        t_out = np.array([t_out])

    #-- CREATING DESIGN MATRIX FOR REGRESSION
    DMAT = []
    MMAT = []
    #-- add polynomial orders (0=constant, 1=linear, 2=quadratic)
    for o in range(ORDER+1):
        DMAT.append((t_in-RELATIVE)**o)
        MMAT.append((t_out-RELATIVE)**o)
    #-- add cyclical terms (0.5=semi-annual, 1=annual)
    for c in CYCLES:
        DMAT.append(np.sin(2.0*np.pi*t_in/np.float(c)))
        DMAT.append(np.cos(2.0*np.pi*t_in/np.float(c)))
        MMAT.append(np.sin(2.0*np.pi*t_out/np.float(c)))
        MMAT.append(np.cos(2.0*np.pi*t_out/np.float(c)))

    #-- Calculating Least-Squares Coefficients
    #-- Standard Least-Squares fitting (the [0] denotes coefficients output)
    beta_mat = np.linalg.lstsq(np.transpose(DMAT), d_in, rcond=-1)[0]

    #-- return modeled time-series
    return np.dot(np.transpose(MMAT),beta_mat)


#-- PURPOSE: interpolate RACMO products to a set of coordinates and times
#-- wrapper function to extract EPSG and print to terminal
def racmo_interp_downscaled(base_dir, VERSION, PRODUCT, COORDINATES=None,
    DATES=None, CSV=None, FILL_VALUE=None):

    #-- this is the projection of the coordinates being interpolated into
    EPSG = "EPSG:{0:d}".format(3413)

    #-- read coordinates and dates from a csv file (X,Y,year decimal)
    if CSV:
        X,Y,tdec = np.loadtxt(CSV,delimiter=',').T
    else:
        #-- regular expression pattern for extracting x and y coordinates
        numerical_regex = '([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)'
        regex = re.compile('\[{0},{0}\]'.format(numerical_regex))
        #-- number of coordinates
        npts = len(regex.findall(COORDINATES))
        #-- x and y coordinates of interpolation points
        X = np.zeros((npts))
        Y = np.zeros((npts))
        for i,XY in enumerate(regex.findall(COORDINATES)):
            X[i],Y[i] = np.array(XY, dtype=np.float)
        #-- convert dates to ordinal days (count of days of the Common Era)
        tdec = np.array(DATES, dtype=np.float)

    #-- read and interpolate/extrapolate RACMO2.3 products
    vi,itype,fv = interpolate_racmo_downscaled(base_dir, EPSG, VERSION, PRODUCT,
        tdec, X, Y, FILL_VALUE=FILL_VALUE)
    interpolate_types = ['invalid','interpolated','backward','forward']
    for v,t in zip(vi,itype):
        print(v,interpolate_types[t])

#-- PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
    print(' -D X, --directory=X\tWorking data directory')
    print(' --version=X\t\tDownscaled RACMO Version')
    print('\t1.0: RACMO2.3/XGRN11')
    print('\t2.0: RACMO2.3p2/XGRN11')
    print('\t3.0: RACMO2.3p2/FGRN055')
    print(' --product:\t\tRACMO product to calculate')
    print('\tSMB: Surface Mass Balance')
    print('\tPRECIP: Precipitation')
    print('\tRUNOFF: Melt Water Runoff')
    print('\tSNOWMELT: Snowmelt')
    print('\tREFREEZE: Melt Water Refreeze')
    print(' --coordinate=X\t\tPolar Stereographic X and Y of point')
    print(' --date=X\t\tDates to interpolate in year-decimal format')
    print(' --csv=X\t\tRead dates and coordinates from a csv file')
    print(' --fill-value\t\tReplace invalid values with fill value\n')

#-- Main program that calls racmo_interp_downscaled()
def main():
    #-- Read the system arguments listed after the program
    long_options = ['help','directory=','version=','product=','coordinate=',
        'date=','csv=','fill-value=']
    optlist,arglist = getopt.getopt(sys.argv[1:], 'hD:', long_options)

    #-- data directory
    base_dir = os.getcwd()
    #-- Downscaled version
    VERSION = '3.0'
    #-- Products to calculate cumulative
    PRODUCTS = ['SMB']
    #-- coordinates and times to run
    COORDINATES = None
    DATES = None
    #-- read coordinates and dates from csv file
    CSV = None
    #-- invalid value (default is nan)
    FILL_VALUE = np.nan
    #-- extract parameters
    for opt, arg in optlist:
        if opt in ('-h','--help'):
            usage()
            sys.exit()
        elif opt in ("-D","--directory"):
            base_dir = os.path.expanduser(arg)
        elif opt in ("--version"):
            VERSION = arg
        elif opt in ("--product"):
            PRODUCTS = arg.split(',')
        elif opt in ("--coordinate"):
            COORDINATES = arg
        elif opt in ("--date"):
            DATES = arg.split(',')
        elif opt in ("--csv"):
            CSV = os.path.expanduser(arg)
        elif opt in ("--fill-value"):
            FILL_VALUE = eval(arg)

    #-- data product longnames
    longname = {}
    longname['SMB'] = 'Cumulative Surface Mass Balance Anomalies'
    longname['PRECIP'] = 'Cumulative Precipitation Anomalies'
    longname['RUNOFF'] = 'Cumulative Runoff Anomalies'
    longname['SNOWMELT'] = 'Cumulative Snowmelt Anomalies'
    longname['REFREEZE'] = 'Cumulative Melt Water Refreeze Anomalies'

    #-- for each product
    for p in PRODUCTS:
        #-- check that product was entered correctly
        if p not in longname.keys():
            raise IOError('{0} not in valid RACMO products'.format(p))
        #-- run program with parameters
        racmo_interp_downscaled(base_dir, VERSION, p, COORDINATES=COORDINATES,
            DATES=DATES, CSV=CSV, FILL_VALUE=FILL_VALUE)

#-- run main program
if __name__ == '__main__':
    main()
