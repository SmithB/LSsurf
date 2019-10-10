#!/usr/bin/env python
u"""
merra_hybrid_interp.py
Written by Tyler Sutterley (10/2019)
Interpolates and extrapolates MERRA-2 firn variables to times and coordinates
	MERRA-2 Hybrid firn model outputs provided by Brooke Medley at GSFC

CALLING SEQUENCE:
	python merra_hybrid_interp.py --directory=<path> --region=gris \
		--coordinate=[-39e4,-133e4],[-39e4,-133e4] --date=2016.1,2018.1

COMMAND LINE OPTIONS:
	-D X, --directory=X: Working data directory
	-R X, --region=X: Region to interpolate (gris, ais)
	--coordinate=X: Polar Stereographic X and Y of point
	--date=X: Date to interpolate in year-decimal format
	--csv=X: Read dates and coordinates from a csv file
	--sigma=X: Standard deviation for Gaussian kernel
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
	Written 10/2019
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
import scipy.ndimage
import scipy.interpolate

#-- PURPOSE: set the projection parameters based on the region name
def set_projection(REGION):
	if (REGION == 'ais'):
		projection_flag = 'EPSG:3031'
	elif (REGION == 'gris'):
		projection_flag = 'EPSG:3413'
	return projection_flag

#-- PURPOSE: read and interpolate MERRA-2 hybrid firn corrections
def interpolate_merra_hybrid(base_dir, EPSG, REGION, tdec, X, Y,
	VARIABLE='FAC', SIGMA=1.5, FILL_VALUE=None):

	#-- set the input netCDF4 file for the variable of interest
	if VARIABLE in ('FAC'):
		hybrid_file='gsfc_{0}_{1}.nc'.format(VARIABLE,REGION.lower())
	elif VARIABLE in ('p_minus_e','melt'):
		hybrid_file='m2_hybrid_{0}_cumul_{1}.nc'.format(VARIABLE,REGION.lower())
	#-- Open the MERRA-2 Hybrid NetCDF file for reading
	fileID = netCDF4.Dataset(os.path.join(base_dir,hybrid_file), 'r')
	#-- Get data from each netCDF variable and remove singleton dimensions
	fd = {}
	fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][:].copy())
	xg = fileID.variables['x'][:,:].copy()
	yg = fileID.variables['y'][:,:].copy()
	fd['time'] = fileID.variables['time'][:].copy()
	#-- invalid data value
	fv = np.float(fileID.variables[VARIABLE]._FillValue)
	#-- input shape of MERRA-2 Hybrid firn data
	nt,nx,ny = np.shape(fd[VARIABLE])
	#-- close the NetCDF files
	fileID.close()
	#-- time is year decimal at time step 5 days
	time_step = 5.0/365.25

	#-- indices of specified ice mask
	i,j = np.nonzero(fd[VARIABLE][0,:,:] != fv)
	#-- create mask object for interpolating data
	fd['mask'] = np.zeros((nx,ny))
	fd['mask'][i,j] = 1.0
	#-- extract x and y coordinate arrays from grids
	fd['x'],fd['y'] = (xg[:,0],yg[0,:])

	#-- use a gaussian filter to smooth mask
	gs = {}
	gs['mask'] = scipy.ndimage.gaussian_filter(fd['mask'], SIGMA,
		mode='constant', cval=0)
	#-- indices of smoothed ice mask
	ii,jj = np.nonzero(np.ceil(gs['mask']) == 1.0)
	#-- use a gaussian filter to smooth each firn field
	gs[VARIABLE] = np.ma.zeros((nt,nx,ny), fill_value=fv)
	gs[VARIABLE].mask = (gs['mask'] == 0.0)
	for t in range(nt):
		#-- replace fill values before smoothing data
		temp1 = np.zeros((nx,ny))
		#-- reference to first firn field
		temp1[i,j] = fd[VARIABLE][t,i,j] - fd[VARIABLE][0,i,j]
		#-- smooth firn field
		temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
			mode='constant', cval=0)
		#-- scale output smoothed firn field
		gs[VARIABLE][t,ii,jj] = temp2[ii,jj]/gs['mask'][ii,jj]
		#-- replace valid firn values with original
		gs[VARIABLE][t,i,j] = temp1[i,j]

	#-- convert projection from input coordinates (EPSG) to model coordinates
	#-- MERRA-2 Hybrid models are rotated pole latitude and longitude
	MODEL_EPSG = set_projection(REGION)
	proj1 = pyproj.Proj("+init={0}".format(EPSG))
	proj2 = pyproj.Proj("+init={0}".format(MODEL_EPSG))
	ix,iy = pyproj.transform(proj1, proj2, X, Y)

	#-- check that input points are within convex hull of smoothed model points
	points = np.concatenate((xg[ii,jj,None],yg[ii,jj,None]),axis=1)
	triangle = scipy.spatial.Delaunay(points.data, qhull_options='Qt Qbb Qc Qz')
	interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
	valid = (triangle.find_simplex(interp_points) >= 0)

	#-- output interpolated arrays of firn variable (height or firn air content)
	interp_firn = np.full_like(tdec,fv,dtype=np.float)
	#-- type designating algorithm used (1: interpolate, 2: backward, 3:forward)
	interp_type = np.zeros_like(tdec,dtype=np.uint8)
	#-- interpolation mask of invalid values
	interp_mask = np.zeros_like(tdec,dtype=np.bool)

	#-- find days that can be interpolated
	if np.any((tdec >= fd['time'].min()) & (tdec <= fd['time'].max()) & valid):
		#-- indices of dates for interpolated days
		ind, = np.nonzero((tdec >= fd['time'].min()) &
			(tdec <= fd['time'].max()) & valid)
		#-- create an interpolator for firn height or air content
		RGI = scipy.interpolate.RegularGridInterpolator(
			(fd['time'],fd['x'],fd['y']), gs[VARIABLE])
		#-- create an interpolator for input mask
		MI = scipy.interpolate.RegularGridInterpolator(
			(fd['x'],fd['y']), (1.0 - gs['mask']))

		#-- interpolate to points
		interp_firn[ind] = RGI.__call__(np.c_[tdec[ind],ix[ind],iy[ind]])
		interp_mask[ind] = MI.__call__(np.c_[ix[ind],iy[ind]]).astype(np.uint8)
		#-- set interpolation type (1: interpolated)
		interp_type[ind] = 1

	#-- check if needing to extrapolate backwards in time
	count = np.count_nonzero((tdec < fd['time'].min()) & valid)
	if (count > 0):
		#-- indices of dates before firn model
		ind, = np.nonzero((tdec < fd['time'].min()) & valid)
		#-- set interpolation type (2: extrapolated backwards)
		interp_type[ind] = 2
		#-- calculate a regression model for calculating values
		#-- read first 10 years of data to create regression model
		N = np.int(10.0/time_step)
		#-- spatially interpolate firn elevation or air content to coordinates
		FIRN = np.zeros((count,N))
		T = np.zeros((N))
		#-- spatially interpolate mask to coordinates
		mspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
			(1.0 - gs['mask']), kx=1, ky=1)
		interp_mask[ind] = mspl.ev(ix[ind],iy[ind]).astype(np.uint8)
		#-- create interpolated time series for calculating regression model
		for k in range(N):
			#-- time at k
			T[k] = fd['time'][k]
			#-- spatially interpolate firn elevation or air content
			fspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
				gs[VARIABLE][k,:,:], kx=1, ky=1)
			#-- create numpy masked array of interpolated values
			FIRN[:,k] = fspl.ev(ix[ind],iy[ind])

		#-- calculate regression model
		for n,v in enumerate(ind):
			interp_firn[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
				CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])

	#-- check if needing to extrapolate forward in time
	count = np.count_nonzero((tdec > fd['time'].max()) & valid)
	if (count > 0):
		#-- indices of dates after firn model
		ind, = np.nonzero((tdec > fd['time'].max()) & valid)
		#-- set interpolation type (3: extrapolated forward)
		interp_type[ind] = 3
		#-- calculate a regression model for calculating values
		#-- read last 10 years of data to create regression model
		N = np.int(10.0/time_step)
		#-- spatially interpolate firn elevation or air content to coordinates
		FIRN = np.zeros((count,N))
		T = np.zeros((N))
		#-- spatially interpolate mask to coordinates
		mspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
			(1.0 - gs['mask']), kx=1, ky=1)
		interp_mask[ind] = mspl.ev(ix[ind],iy[ind]).astype(np.uint8)
		#-- create interpolated time series for calculating regression model
		for k in range(N):
			kk = nt - N + k
			#-- time at k
			T[k] = fd['time'][kk]
			#-- spatially interpolate firn elevation or air content
			fspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
				gs[VARIABLE][kk,:,:], kx=1, ky=1)
			#-- create numpy masked array of interpolated values
			FIRN[:,k] = fspl.ev(ix[ind],iy[ind])

		#-- calculate regression model
		for n,v in enumerate(ind):
			interp_firn[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
				CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])

	#-- replace fill value if specified
	if FILL_VALUE:
		ind, = np.nonzero((interp_firn == fv) | interp_mask)
		interp_firn[ind] = FILL_VALUE
		fv = FILL_VALUE

	#-- return the interpolated values
	return (interp_firn,interp_type,fv)

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

#-- PURPOSE: interpolate MERRA-2 firn height to a set of coordinates and times
#-- wrapper function to extract EPSG and print to terminal
def merra_hybrid_interp(base_dir, REGION, COORDINATES=None,
	DATES=None, CSV=None, SIGMA=None, FILL_VALUE=None):

	#-- get projection information from region
	EPSG = set_projection(REGION)

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

	#-- read and interpolate/extrapolate MERRA-2 Hybrid firn corrections
	air,itype,fv = interpolate_merra_hybrid(base_dir, EPSG, REGION, tdec, X, Y,
		VARIABLE='FAC', SIGMA=SIGMA, FILL_VALUE=FILL_VALUE)
	interpolate_types = ['invalid','interpolated','backward','forward']
	for a,t in zip(air,itype):
		print(a,interpolate_types[t])

#-- PURPOSE: help module to describe the optional input parameters
def usage():
	print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
	print(' -D X, --directory=X\tWorking data directory')
	print(' -R X, --region=X\tRegion of firn model to interpolate')
	print(' --coordinate=X\t\tPolar Stereographic X and Y of point')
	print(' --date=X\t\tDates to interpolate in year-decimal format')
	print(' --csv=X\t\tRead dates and coordinates from a csv file')
	print(' --sigma=X\t\tStandard deviation for Gaussian kernel')
	print(' --fill-value\t\tReplace invalid values with fill value\n')

#-- Main program that calls merra_hybrid_interp()
def main():
	#-- Read the system arguments listed after the program
	long_options = ['help','directory=','region=','coordinate=','date=','csv=',
		'sigma=','fill-value=']
	optlist,arglist = getopt.getopt(sys.argv[1:], 'hD:R:', long_options)

	#-- data directory
	base_dir = os.getcwd()
	#-- region of firn model
	REGION = 'gris'
	#-- coordinates and times to run
	COORDINATES = None
	DATES = None
	#-- read coordinates and dates from csv file
	CSV = None
	#-- standard deviation for gaussian kernel
	SIGMA = 1.5
	#-- invalid value (default is from MERRA-2 Hybrid file)
	FILL_VALUE = None
	#-- extract parameters
	for opt, arg in optlist:
		if opt in ('-h','--help'):
			usage()
			sys.exit()
		elif opt in ("-D","--directory"):
			base_dir = os.path.expanduser(arg)
		elif opt in ("-R","--region"):
			REGION = arg.lower()
		elif opt in ("--coordinate"):
			COORDINATES = arg
		elif opt in ("--date"):
			DATES = arg.split(',')
		elif opt in ("--csv"):
			CSV = os.path.expanduser(arg)
		elif opt in ("--sigma"):
			SIGMA = np.float(arg)
		elif opt in ("--fill-value"):
			FILL_VALUE = eval(arg)

	#-- run program with parameters
	merra_hybrid_interp(base_dir, REGION, COORDINATES=COORDINATES,
		DATES=DATES, CSV=CSV, SIGMA=SIGMA, FILL_VALUE=FILL_VALUE)

#-- run main program
if __name__ == '__main__':
	main()
