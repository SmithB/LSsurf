#!/usr/bin/env python
u"""
racmo_extrap_downscaled.py
Written by Tyler Sutterley (09/2019)
Interpolates and extrapolates downscaled RACMO products to times and coordinates

Uses fast nearest-neighbor search algorithms
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
and inverse distance weighted interpolation to extrapolate spatially

CALLING SEQUENCE:
	python racmo_extrap_downscaled.py --directory=<path> --version=3.0 \
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
	scikit-learn: Machine Learning in Python
		http://scikit-learn.org/stable/index.html
		https://github.com/scikit-learn/scikit-learn

UPDATE HISTORY:
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
import scipy.interpolate
from sklearn.neighbors import KDTree, BallTree

#-- PURPOSE: read and interpolate downscaled RACMO products
def extrapolate_racmo_downscaled(base_dir, EPSG, VERSION, PRODUCT, tdec, X, Y,
	SEARCH='BallTree', N=10, POWER=2.0, FILL_VALUE=None):

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

	#-- Open the RACMO NetCDF file for reading
	fileID = netCDF4.Dataset(os.path.join(input_dir,input_file), 'r')
	#-- Get data from each netCDF variable and remove singleton dimensions
	d = {}
	d[VARNAME] = np.squeeze(fileID.variables[VARNAME][:].copy())
	#-- cell origins on the bottom right
	dx = np.abs(fileID.variables['x'][1]-fileID.variables['x'][0])
	dy = np.abs(fileID.variables['y'][1]-fileID.variables['y'][0])
	#-- latitude and longitude arrays at center of each cell
	d['LON'] = fileID.variables['LON'][:,:].copy()
	d['LAT'] = fileID.variables['LAT'][:,:].copy()
	#-- extract time (decimal years)
	d['TIME'] = fileID.variables['TIME'][:].copy()
	#-- mask object for interpolating data
	d['MASK'] = np.array(fileID.variables['MASK'][:],dtype=np.bool)
	i,j = np.nonzero(d['MASK'])
	#-- input shape of RACMO data
	nt,ny,nx = np.shape(d[VARNAME])
	#-- close the NetCDF files
	fileID.close()

	#-- convert RACMO latitude and longitude to input coordinates (EPSG)
	proj1 = pyproj.Proj("+init={0}".format(EPSG))
	proj2 = pyproj.Proj("+init=EPSG:{0:d}".format(4326))
	xg,yg = pyproj.transform(proj2, proj1, d['LON'], d['LAT'])

	#-- construct search tree from original points
	#-- can use either BallTree or KDTree algorithms
	xy1 = np.concatenate((xg[i,j,None],yg[i,j,None]),axis=1)
	tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

	#-- output extrapolated arrays of variable
	extrap_var = np.zeros_like(tdec,dtype=np.float)
	#-- type designating algorithm used (1: interpolate, 2: backward, 3:forward)
	extrap_type = np.zeros_like(tdec,dtype=np.uint8)

	#-- find days that can be extrapolated
	if np.any((tdec >= d['TIME'].min()) & (tdec <= d['TIME'].max())):
		#-- indices of dates for interpolated days
		ind,=np.nonzero((tdec >= d['TIME'].min()) & (tdec < d['TIME'].max()))
		f = scipy.interpolate.interp1d(d['TIME'], np.arange(nt), kind='linear')
		date_indice = f(tdec[ind]).astype(np.int)
		#-- set interpolation type (1: interpolated in time)
		extrap_type[ind] = 1
		#-- for each unique RACMO date
		#-- linearly interpolate in time between two RACMO maps
		#-- then then inverse distance weighting to extrapolate in space
		for k in np.unique(date_indice):
			kk, = np.nonzero(date_indice==k)
			count = np.count_nonzero(date_indice==k)
			#-- query the search tree to find the N closest points
			xy2 = np.concatenate((X[kk,None],Y[kk,None]),axis=1)
			dist,indices = tree.query(xy2, k=N, return_distance=True)
			#-- normalized weights if POWER > 0 (typically between 1 and 3)
			#-- in the inverse distance weighting
			power_inverse_distance = dist**(-POWER)
			s = np.sum(power_inverse_distance, axis=1)
			w = power_inverse_distance/np.broadcast_to(s[:,None],(count,N))
			#-- RACMO variables for times before and after tdec
			var1 = d[VARNAME][k,i,j]
			var2 = d[VARNAME][k+1,i,j]
			#-- linearly interpolate to date
			dt = (tdec[kk] - d['TIME'][k])/(d['TIME'][k+1] - d['TIME'][k])
			#-- spatially extrapolate using inverse distance weighting
			extrap_var[kk] = (1.0-dt)*np.sum(w*var1[indices],axis=1) + \
				dt*np.sum(w*var2[indices], axis=1)

	#-- check if needing to extrapolate backwards in time
	count = np.count_nonzero((tdec < d['TIME'].min()))
	if (count > 0):
		#-- indices of dates before RACMO
		ind, = np.nonzero(tdec < d['TIME'].min())
		#-- set interpolation type (2: extrapolated backwards in time)
		extrap_type[ind] = 2
		#-- query the search tree to find the N closest points
		xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
		dist,indices = tree.query(xy2, k=N, return_distance=True)
		#-- normalized weights if POWER > 0 (typically between 1 and 3)
		#-- in the inverse distance weighting
		power_inverse_distance = dist**(-POWER)
		s = np.sum(power_inverse_distance, axis=1)
		w = power_inverse_distance/np.broadcast_to(s[:,None],(count,N))
		#-- calculate a regression model for calculating values
		#-- read first 10 years of data to create regression model
		N = 120
		#-- spatially interpolate variables to coordinates
		VAR = np.zeros((count,N))
		T = np.zeros((N))
		#-- create interpolated time series for calculating regression model
		for k in range(N):
			#-- time at k
			T[k] = d['TIME'][k]
			#-- spatially extrapolate variables
			var1 = d[VARNAME][k,i,j]
			VAR[:,k] = np.sum(w*var1[indices],axis=1)

		#-- calculate regression model
		for n,v in enumerate(ind):
			extrap_var[v] = regress_model(T, VAR[n,:], tdec[v], ORDER=2,
				CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])

	#-- check if needing to extrapolate forward in time
	count = np.count_nonzero((tdec > d['TIME'].max()))
	if (count > 0):
		#-- indices of dates after RACMO
		ind, = np.nonzero(tdec >= d['TIME'].max())
		#-- set interpolation type (3: extrapolated forward in time)
		extrap_type[ind] = 3
		#-- query the search tree to find the N closest points
		xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
		dist,indices = tree.query(xy2, k=N, return_distance=True)
		#-- normalized weights if POWER > 0 (typically between 1 and 3)
		#-- in the inverse distance weighting
		power_inverse_distance = dist**(-POWER)
		s = np.sum(power_inverse_distance, axis=1)
		w = power_inverse_distance/np.broadcast_to(s[:,None],(count,N))
		#-- calculate a regression model for calculating values
		#-- read last 10 years of data to create regression model
		N = 120
		#-- spatially interpolate variables to coordinates
		FIRN = np.zeros((count,N))
		T = np.zeros((N))
		#-- create interpolated time series for calculating regression model
		for k in range(N):
			kk = nt - N + k
			#-- time at k
			T[k] = d['TIME'][kk]
			#-- spatially extrapolate variables
			var1 = d[VARNAME][kk,i,j]
			VAR[:,k] = np.sum(w*var1[indices],axis=1)

		#-- calculate regression model
		for n,v in enumerate(ind):
			extrap_var[v] = regress_model(T, VAR[n,:], tdec[v], ORDER=2,
				CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])

	#-- replace fill value if specified
	if FILL_VALUE:
		ind, = np.nonzero(extrap_type == 0)
		extrap_var[ind] = FILL_VALUE
		fv = FILL_VALUE
	else:
		fv = 0.0

	#-- return the extrapolated values
	return (extrap_var,extrap_type,fv)

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
def racmo_extrap_downscaled(base_dir, VERSION, PRODUCT, COORDINATES=None,
	DATES=None, CSV=None, FILL_VALUE=None):

	#-- this is the projection of the coordinates being extrapolated into
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
	vi,itype,fv = extrapolate_racmo_downscaled(base_dir, EPSG, VERSION, PRODUCT,
		tdec, X, Y, FILL_VALUE=FILL_VALUE)
	interpolate_types = ['invalid','extrapolated','backward','forward']
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

#-- Main program that calls racmo_extrap_downscaled()
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
		racmo_extrap_downscaled(base_dir, VERSION, p, COORDINATES=COORDINATES,
			DATES=DATES, CSV=CSV, FILL_VALUE=FILL_VALUE)

#-- run main program
if __name__ == '__main__':
	main()
