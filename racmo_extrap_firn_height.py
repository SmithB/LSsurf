#!/usr/bin/env python
u"""
racmo_extrap_firn_height.py
Written by Tyler Sutterley (10/2019)
Interpolates and extrapolates firn heights to times and coordinates

Uses fast nearest-neighbor search algorithms
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
and inverse distance weighted interpolation to extrapolate spatially

CALLING SEQUENCE:
	python racmo_extrap_firn_height.py --directory=<path> --smb=FGRN055 \
		--coordinate=[-39e4,-133e4],[-39e4,-133e4] --date=2016.1,2018.1

COMMAND LINE OPTIONS:
	-D X, --directory=X: Working data directory
	-S X, --smb=X: Firn model outputs to interpolate
		FGRN055: 1km interpolated Greenland RACMO2.3p2
		FGRN11: 11km Greenland RACMO2.3p2
		XANT27: 27km Antarctic RACMO2.3p2
		ASE055: 5.5km Amundsen Sea Embayment RACMO2.3p2
		XPEN055: 5.5km Antarctic Peninsula RACMO2.3p2
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
	scikit-learn: Machine Learning in Python
		http://scikit-learn.org/stable/index.html
		https://github.com/scikit-learn/scikit-learn

UPDATE HISTORY:
	Updated 10/2019: Gaussian average firn fields before interpolation
	Updated 09/2019: use scipy interpolate to find date indices
	Forked 08/2019 from racmo_interp_firn_height.py
	Updated 08/2019: convert to model coordinates (rotated pole lat/lon)
		and interpolate using N-dimensional functions
		added rotation parameters for Antarctic models (XANT27,ASE055,XPEN055)
		added option to change the fill value for invalid points
	Written 07/2019
"""
from __future__ import print_function

import sys
import os
import re
import pyproj
import getopt
import netCDF4
import numpy as np
import scipy.ndimage
import scipy.interpolate
from sklearn.neighbors import KDTree, BallTree

#-- PURPOSE: set the projection parameters based on the firn model shortname
#-- these are the default projections of the coordinates being interpolated into
#-- and not the projection of the models (interpolate into polar stereographic)
def set_projection(FIRN_MODEL):
	if FIRN_MODEL in ('XANT27','ASE055','XPEN055'):
		projection_flag = 'EPSG:3031'
	elif FIRN_MODEL in ('FGRN11','FGRN055'):
		projection_flag = 'EPSG:3413'
	return projection_flag

#-- PURPOSE: read and interpolate RACMO2.3 firn corrections
def extrapolate_racmo_firn(base_dir, EPSG, MODEL, tdec, X, Y, SEARCH='BallTree',
	N=10, POWER=2.0, SIGMA=1.5, VARIABLE='zs', FILL_VALUE=None):

	#-- set parameters based on input model
	FIRN_FILE = {}
	if (MODEL == 'FGRN11'):
		#-- filename and directory for input FGRN11 file
		FIRN_FILE['zs'] = 'FDM_zs_FGRN11_1960-2016.nc'
		FIRN_FILE['FirnAir'] = 'FDM_FirnAir_FGRN11_1960-2016.nc'
		FIRN_DIRECTORY = ['RACMO','FGRN11_1960-2016']
	elif (MODEL == 'FGRN055'):
		#-- filename and directory for input FGRN055 file
		FIRN_FILE['zs'] = 'FDM_zs_FGRN055_1960-2017_interpol.nc'
		FIRN_FILE['FirnAir'] = 'FDM_FirnAir_FGRN055_1960-2017_interpol.nc'
		FIRN_DIRECTORY = ['RACMO','FGRN055_1960-2017']
	elif (MODEL == 'XANT27'):
		#-- filename and directory for input XANT27 file
		FIRN_FILE['zs'] = 'FDM_zs_ANT27_1979-2016.nc'
		FIRN_FILE['FirnAir'] = 'FDM_FirnAir_ANT27_1979-2016.nc'
		FIRN_DIRECTORY = ['RACMO','XANT27_1979-2016']
	elif (MODEL == 'ASE055'):
		#-- filename and directory for input ASE055 file
		FIRN_FILE['zs'] = 'FDM_zs_ASE055_1979-2015.nc'
		FIRN_FILE['FirnAir'] = 'FDM_FirnAir_ASE055_1979-2015.nc'
		FIRN_DIRECTORY = ['RACMO','ASE055_1979-2015']
	elif (MODEL == 'XPEN055'):
		#-- filename and directory for input XPEN055 file
		FIRN_FILE['zs'] = 'FDM_zs_XPEN055_1979-2016.nc'
		FIRN_FILE['FirnAir'] = 'FDM_FirnAir_XPEN055_1979-2016.nc'
		FIRN_DIRECTORY = ['RACMO','XPEN055_1979-2016']

	#-- Open the RACMO NetCDF file for reading
	ddir = os.path.join(base_dir,*FIRN_DIRECTORY)
	fileID = netCDF4.Dataset(os.path.join(ddir,FIRN_FILE[VARIABLE]), 'r')
	#-- Get data from each netCDF variable and remove singleton dimensions
	fd = {}
	fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][:].copy())
	fd['lon'] = fileID.variables['lon'][:,:].copy()
	fd['lat'] = fileID.variables['lat'][:,:].copy()
	fd['time'] = fileID.variables['time'][:].copy()
	#-- invalid data value
	fv = np.float(fileID.variables[VARIABLE]._FillValue)
	#-- input shape of RACMO firn data
	nt,ny,nx = np.shape(fd[VARIABLE])
	#-- close the NetCDF files
	fileID.close()

	#-- indices of specified ice mask
	i,j = np.nonzero(fd[VARIABLE][0,:,:] != fv)
	#-- create mask object for interpolating data
	fd['mask'] = np.zeros((ny,nx))
	fd['mask'][i,j] = 1.0

	#-- use a gaussian filter to smooth mask
	gs = {}
	gs['mask'] = scipy.ndimage.gaussian_filter(fd['mask'], SIGMA,
		mode='constant', cval=0)
	#-- indices of smoothed ice mask
	ii,jj = np.nonzero(np.ceil(gs['mask']) == 1.0)
	#-- use a gaussian filter to smooth each firn field
	gs[VARIABLE] = np.ma.zeros((nt,ny,nx), fill_value=fv)
	gs[VARIABLE].mask = (gs['mask'] == 0.0)
	for t in range(nt):
		#-- replace fill values before smoothing data
		temp1 = np.zeros((ny,nx))
		#-- reference to first firn field
		temp1[i,j] = fd[VARIABLE][t,i,j] - fd[VARIABLE][0,i,j]
		#-- smooth firn field
		temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
			mode='constant', cval=0)
		#-- scale output smoothed firn field
		gs[VARIABLE][t,ii,jj] = temp2[ii,jj]/gs['mask'][ii,jj]
		#-- replace valid firn values with original
		gs[VARIABLE][t,i,j] = temp1[i,j]

	#-- convert RACMO latitude and longitude to input coordinates (EPSG)
	proj1 = pyproj.Proj("+init={0}".format(EPSG))
	proj2 = pyproj.Proj("+init=EPSG:{0:d}".format(4326))
	xg,yg = pyproj.transform(proj2, proj1, fd['lon'], fd['lat'])

	#-- construct search tree from original points
	#-- can use either BallTree or KDTree algorithms
	xy1 = np.concatenate((xg[ii,jj,None],yg[ii,jj,None]),axis=1)
	tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

	#-- output interpolated arrays of firn variable (height or firn air content)
	extrap_firn = np.full_like(tdec,fv,dtype=np.float)
	#-- type designating algorithm used (1: interpolate, 2: backward, 3:forward)
	extrap_type = np.zeros_like(tdec,dtype=np.uint8)

	#-- find days that can be interpolated
	if np.any((tdec >= fd['time'].min()) & (tdec < fd['time'].max())):
		#-- indices of dates for interpolated days
		ind,=np.nonzero((tdec >= fd['time'].min()) & (tdec < fd['time'].max()))
		f = scipy.interpolate.interp1d(fd['time'], np.arange(nt), kind='linear')
		date_indice = f(tdec[ind]).astype(np.int)
		#-- set interpolation type (1: interpolated in time)
		extrap_type[ind] = 1
		#-- for each unique firn date
		#-- linearly interpolate in time between two firn maps
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
			#-- firn height or air content for times before and after tdec
			firn1 = gs[VARIABLE][k,ii,jj]
			firn2 = gs[VARIABLE][k+1,ii,jj]
			#-- linearly interpolate to date
			dt = (tdec[kk] - fd['time'][k])/(fd['time'][k+1] - fd['time'][k])
			#-- spatially extrapolate using inverse distance weighting
			extrap_firn[kk] = (1.0-dt)*np.sum(w*firn1[indices],axis=1) + \
				dt*np.sum(w*firn2[indices], axis=1)

	#-- check if needing to extrapolate backwards in time
	count = np.count_nonzero(tdec < fd['time'].min())
	if (count > 0):
		#-- indices of dates before firn model
		ind, = np.nonzero(tdec < fd['time'].min())
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
		N = 365
		#-- spatially interpolate firn elevation or air content to coordinates
		FIRN = np.zeros((count,N))
		T = np.zeros((N))
		#-- create interpolated time series for calculating regression model
		for k in range(N):
			#-- time at k
			T[k] = fd['time'][k]
			#-- spatially extrapolate firn elevation or air content
			firn1 = gs[VARIABLE][k,ii,jj]
			FIRN[:,k] = np.sum(w*firn1[indices],axis=1)

		#-- calculate regression model
		for n,v in enumerate(ind):
			extrap_firn[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
				CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])

	#-- check if needing to extrapolate forward in time
	count = np.count_nonzero(tdec >= fd['time'].max())
	if (count > 0):
		#-- indices of dates after firn model
		ind, = np.nonzero(tdec >= fd['time'].max())
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
		N = 365
		#-- spatially interpolate firn elevation or air content to coordinates
		FIRN = np.zeros((count,N))
		T = np.zeros((N))
		#-- create interpolated time series for calculating regression model
		for k in range(N):
			kk = nt - N + k
			#-- time at k
			T[k] = fd['time'][kk]
			#-- spatially extrapolate firn elevation or air content
			firn1 = gs[VARIABLE][kk,ii,jj]
			FIRN[:,k] = np.sum(w*firn1[indices],axis=1)

		#-- calculate regression model
		for n,v in enumerate(ind):
			extrap_firn[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
				CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])

	#-- replace fill value if specified
	if FILL_VALUE:
		ind, = np.nonzero(extrap_firn == fv)
		extrap_firn[ind] = FILL_VALUE
		fv = FILL_VALUE

	#-- return the interpolated values
	return (extrap_firn,extrap_type,fv)

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


#-- PURPOSE: interpolate RACMO firn height to a set of coordinates and times
#-- wrapper function to extract EPSG and print to terminal
def racmo_extrap_firn_height(base_dir, MODEL, COORDINATES=None,
	DATES=None, CSV=None, SIGMA=None, FILL_VALUE=None):

	#-- get projection information from model shortname
	#-- this is the projection of the coordinates being interpolated into
	EPSG = set_projection(MODEL)

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

	#-- read and interpolate/extrapolate RACMO2.3 firn corrections
	zs,itype,fv = extrapolate_racmo_firn(base_dir, EPSG, MODEL, tdec, X, Y,
		VARIABLE='zs', SIGMA=SIGMA, FILL_VALUE=FILL_VALUE)
	air,itype,fv = extrapolate_racmo_firn(base_dir, EPSG, MODEL, tdec, X, Y,
		VARIABLE='FirnAir', SIGMA=SIGMA, FILL_VALUE=FILL_VALUE)
	interpolate_types = ['invalid','interpolated','backward','forward']
	for z,a,t in zip(zs,air,itype):
		print(z,a,interpolate_types[t])

#-- PURPOSE: help module to describe the optional input parameters
def usage():
	print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
	print(' -D X, --directory=X\tWorking data directory')
	print(' -S X, --smb=X\t\tSMB model to interpolate')
	print('\tFGRN055: 1km interpolated Greenland RACMO2.3p2')
	print('\tFGRN11: 11km Greenland RACMO2.3p2')
	print('\tXANT27: 27km Antarctic RACMO2.3p2')
	print('\tASE055: 5.5km Amundsen Sea Embayment RACMO2.3p2')
	print('\tXPEN055: 5.5km Antarctic Peninsula RACMO2.3p2')
	print(' --coordinate=X\tPolar Stereographic X and Y of point')
	print(' --date=X\t\tDates to interpolate in year-decimal format')
	print(' --csv=X\t\tRead dates and coordinates from a csv file')
	print(' --sigma=X\t\tStandard deviation for Gaussian kernel')
	print(' --fill-value\tReplace invalid values with fill value\n')

#-- Main program that calls racmo_extrap_firn_height()
def main():
	#-- Read the system arguments listed after the program
	long_options = ['help','directory=','smb=','coordinate=','date=','csv=',
		'sigma=','fill-value=']
	optlist,arglist = getopt.getopt(sys.argv[1:], 'hD:S:', long_options)

	#-- data directory
	base_dir = os.getcwd()
	#-- SMB model to interpolate (RACMO or MAR)
	MODEL = 'FGRN055'
	#-- coordinates and times to run
	COORDINATES = None
	DATES = None
	#-- read coordinates and dates from csv file
	CSV = None
	#-- standard deviation for gaussian kernel
	SIGMA = 1.5
	#-- invalid value (default is from RACMO file)
	FILL_VALUE = None
	#-- extract parameters
	for opt, arg in optlist:
		if opt in ('-h','--help'):
			usage()
			sys.exit()
		elif opt in ("-D","--directory"):
			base_dir = os.path.expanduser(arg)
		elif opt in ("-S","--smb"):
			MODEL = arg
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
	racmo_extrap_firn_height(base_dir, MODEL, COORDINATES=COORDINATES,
		DATES=DATES, CSV=CSV, SIGMA=SIGMA, FILL_VALUE=FILL_VALUE)

#-- run main program
if __name__ == '__main__':
	main()
