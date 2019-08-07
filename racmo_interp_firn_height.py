#!/usr/bin/env python
u"""
racmo_interp_firn_height.py
Written by Tyler Sutterley (08/2019)
Interpolates and extrapolates firn heights to times and coordinates

CALLING SEQUENCE:
	python racmo_interp_firn_height.py --directory=<path> --smb=FGRN055 \
		--coordinate=[-39e4,-133e4],[-39e4,-133e4] --date=2016.1,2018.1

COMMAND LINE OPTIONS:
	-Y X, --year=X: end years to use in reduction separated by commas
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
	Updated 08/2019: convert to model coordinates (rotated pole lat/lon)
		and interpolate using N-dimensional functions
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
import scipy.spatial
import scipy.interpolate

#-- PURPOSE: set the projection parameters based on the firn model shortname
def set_projection(FIRN_MODEL):
	if FIRN_MODEL in ('XANT27','ASE055','XPEN055'):
		projection_flag = 'EPSG:3031'
	elif FIRN_MODEL in ('FGRN11','FGRN055'):
		projection_flag = 'EPSG:3413'
	return projection_flag

#-- PURPOSE: read and interpolate RACMO2.3 firn corrections
def interpolate_racmo_firn(base_dir, EPSG, m, tdec, X, Y):
	if (m == 'FGRN11'):
		#-- filename and directory for input FGRN11 file
		height_file = 'FDM_zs_FGRN11_1960-2016.nc'
		firnair_file = 'FDM_FirnAir_FGRN11_1960-2016.nc'
		FIRN_DIRECTORY = ['RACMO','FGRN11_1960-2016']
		#-- time is year decimal from 1960-01-01 at time_step 10 days
		time_step = 10.0/365.25
		#-- rotation parameters
		rot_lat = -18.0
		rot_lon = -37.5
	elif (m == 'FGRN055'):
		#-- filename and directory for input FGRN055 file
		height_file = 'FDM_zs_FGRN055_1960-2017_interpol.nc'
		firnair_file = 'FDM_FirnAir_FGRN055_1960-2017_interpol.nc'
		FIRN_DIRECTORY = ['RACMO','FGRN055_1960-2017']
		#-- time is year decimal from 1960-01-01 at time_step 10 days
		time_step = 10.0/365.25
		#-- rotation parameters
		rot_lat = -18.0
		rot_lon = -37.5
	elif (m == 'XANT27'):
		#-- filename and directory for input XANT27 file
		height_file = 'FDM_zs_ANT27_1979-2016.nc'
		firnair_file = 'FDM_FirnAir_ANT27_1979-2016.nc'
		FIRN_DIRECTORY = ['RACMO','ANT27_1979-2016']
		#-- time is year decimal from 1979-01-01 at time_step 10 days
		time_step = 10.0/365.25
	elif (m == 'ASE055'):
		#-- filename and directory for input ASE055 file
		height_file = 'FDM_zs_ASE055_1979-2015.nc'
		firnair_file = 'FDM_FirnAir_ASE055_1979-2015.nc'
		FIRN_DIRECTORY = ['RACMO','ASE055_1979-2015']
		#-- time is year decimal from 1979-01-01 at time_step 10 days
		time_step = 10.0/365.25
	elif (m == 'XPEN055'):
		#-- filename and directory for input XPEN055 file
		height_file = 'FDM_zs_XPEN055_1979-2016.nc'
		firnair_file = 'FDM_FirnAir_XPEN055_1979-2016.nc'
		FIRN_DIRECTORY = ['RACMO','XPEN055_1979-2016']
		#-- time is year decimal from 1979-01-01 at time_step 10 days
		time_step = 10.0/365.25

	#-- Open the RACMO NetCDF file for reading
	ddir = os.path.join(base_dir,*FIRN_DIRECTORY)
	fid1 = netCDF4.Dataset(os.path.join(ddir,height_file), 'r')
	# fid2 = netCDF4.Dataset(os.path.join(ddir,firnair_file), 'r')
	#-- Get data from each netCDF variable and remove singleton dimensions
	fd = {}
	fd['zs'] = np.squeeze(fid1.variables['zs'][:].copy())
	# fd['FirnAir'] = np.squeeze(fid2.variables['FirnAir'][:].copy())
	fd['lon'] = fid1.variables['lon'][:,:].copy()
	fd['lat'] = fid1.variables['lat'][:,:].copy()
	fd['time'] = fid1.variables['time'][:].copy()
	#-- invalid data
	fv = np.float(fid1.variables['zs']._FillValue)
	#-- input shape of RACMO firn data
	nt,ny,nx = np.shape(fd['zs'])
	#-- close the NetCDF files
	fid1.close()
	# fid2.close()

	#-- indices of specified ice mask
	i,j = np.nonzero(fd['zs'][0,:,:] != fv)
	#-- create mask object for interpolating data
	fd['mask'] = np.ones((ny,nx),dtype=np.bool)
	fd['mask'][i,j] = False

	#-- rotated pole longitude and latitude of input model (model coordinates)
	xg,yg = rotate_coordinates(fd['lon'], fd['lat'], rot_lon, rot_lat)
	#-- recreate arrays to fix small floating point errors
	fd['x'] = np.linspace(np.mean(xg[:,0]),np.mean(xg[:,-1]),nx)
	fd['y'] = np.linspace(np.mean(yg[0,:]),np.mean(yg[-1,:]),ny)

	#-- convert projection from input coordinates (EPSG) to model coordinates
	#-- RACMO models are rotated pole latitude and longitude
	proj1 = pyproj.Proj("+init={0}".format(EPSG))
	proj2 = pyproj.Proj("+init=EPSG:{0:d}".format(4326))
	ilon,ilat = pyproj.transform(proj1, proj2, X, Y)
	#-- calculate rotated pole coordinates of input coordinates
	ix,iy = rotate_coordinates(ilon, ilat, rot_lon, rot_lat)

	#-- check that input points are within convex hull of valid model points
	points = np.concatenate((xg[i,j,None],yg[i,j,None]),axis=1)
	triangle = scipy.spatial.Delaunay(points, qhull_options='Qt Qbb Qc Qz')
	interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
	valid = (triangle.find_simplex(interp_points) >= 0)

	#-- output interpolated arrays of firn height and air content
	interp_zs = np.full_like(tdec,fv,dtype=np.float)
	interp_air = np.full_like(tdec,fv,dtype=np.float)
	#-- type designating algorithm used (1: interpolate, 2: backward, 3:forward)
	interp_type = np.zeros_like(tdec,dtype=np.uint8)

	#-- find days that can be interpolated
	date_indices = np.array((tdec - fd['time'].min())/time_step, dtype='i')
	if np.any((date_indices >= 0) & (date_indices < nt) & valid):
		#-- indices of dates for interpolated days
		ind, = np.nonzero((date_indices >= 0) & (date_indices < nt) & valid)
		#-- create an interpolator for firn elevation
		ZI = scipy.interpolate.RegularGridInterpolator(
			(fd['time'],fd['y'],fd['x']), fd['zs'])
		#-- create an interpolator for firn air content
		# AI = scipy.interpolate.RegularGridInterpolator(
		# 	(fd['time'],fd['y'],fd['x']), fd['FirnAir'])

		#-- interpolate to points
		interp_zs[ind] = ZI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
		# interp_air[ind] = AI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
		#-- set interpolation type (1: interpolated)
		interp_type[ind] = 1

	#-- check if needing to extrapolate backwards in time
	count = np.count_nonzero((date_indices < 0) & valid)
	if (count > 0):
		#-- indices of dates before firn model
		ind, = np.nonzero((date_indices < 0) & valid)
		#-- set interpolation type (2: extrapolated backwards)
		interp_type[ind] = 2
		#-- calculate a regression model for calculating values
		#-- read first 10 years of data to create regression model
		N = 365
		#-- spatially interpolate firn elevation to coordinates
		ZS = np.zeros((count,N))
		# AIR = np.zeros((count,N))
		T = np.zeros((N))
		#-- spatially interpolate mask to coordinates
		mspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
			fd['mask'].T, kx=1, ky=1)
		mask = mspl.ev(ix[ind],iy[ind])
		#-- create interpolated time series for calculating regression model
		for k in range(N):
			#-- time at k
			T[k] = fd['time'][k]
			#-- spatially interpolate firn elevation to coordinates
			fspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
				fd['zs'][k,:,:].T, kx=1, ky=1)
			# #-- spatially interpolate firn air content to coordinates
			# aspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
			# 	fd['FirnAir'][k,:,:].T, kx=1, ky=1)
			#-- create numpy masked array of interpolated values
			ZS[:,k] = fspl.ev(ix[ind],iy[ind])
			# AIR[:,k] = aspl.ev(ix[ind],iy[ind])

		#-- calculate regression model
		for n,v in enumerate(ind):
			interp_zs[v] = regress_model(T, ZS[n,:], tdec[v], ORDER=2,
				CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])
			# interp_air[v] = regress_model(T, AIR[n,:], tdec[v], ORDER=2,
			# 	CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])

	#-- check if needing to extrapolate forward in time
	count = np.count_nonzero((date_indices >= nt) & valid)
	if (count > 0):
		#-- indices of dates after firn model
		ind, = np.nonzero((date_indices >= nt) & valid)
		#-- set interpolation type (3: extrapolated forward)
		interp_type[ind] = 3
		#-- calculate a regression model for calculating values
		#-- read last 10 years of data to create regression model
		N = 365
		#-- spatially interpolate firn elevation to coordinates
		ZS = np.zeros((count,N))
		# AIR = np.zeros((count,N))
		T = np.zeros((N))
		#-- spatially interpolate mask to coordinates
		mspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
			fd['mask'].T, kx=1, ky=1)
		mask = mspl.ev(ix[ind],iy[ind])
		#-- create interpolated time series for calculating regression model
		for k in range(N):
			kk = nt - N + k
			#-- time at k
			T[k] = fd['time'][kk]
			#-- spatially interpolate firn elevation to coordinates
			fspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
				fd['zs'][kk,:,:].T, kx=1, ky=1)
			# #-- spatially interpolate firn air content to coordinates
			# aspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
			# 	fd['FirnAir'][kk,:,:].T, kx=1, ky=1)
			#-- create numpy masked array of interpolated values
			ZS[:,k] = fspl.ev(ix[ind],iy[ind])
			# AIR[:,k] = aspl.ev(ix[ind],iy[ind])

		#-- calculate regression model
		for n,v in enumerate(ind):
			interp_zs[v] = regress_model(T, ZS[n,:], tdec[v], ORDER=2,
				CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])
			# interp_air[v] = regress_model(T, AIR[n,:], tdec[v], ORDER=2,
			# 	CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])

	#-- return the interpolated values
	return (interp_zs,interp_air,interp_type,fv)

#-- PURPOSE: calculate rotated pole coordinates
def rotate_coordinates(lon, lat, rot_lon, rot_lat):
	#-- convert from degrees to radians
	phi = np.pi*lon/180.0
	phi_r = np.pi*rot_lon/180.0
	th = np.pi*lat/180.0
	th_r = np.pi*rot_lat/180.0
	#-- calculate rotation parameters
	R1 = np.sin(phi - phi_r)*np.cos(th)
	R2 = np.cos(th_r)*np.sin(th) - np.sin(th_r)*np.cos(th)*np.cos(phi - phi_r)
	R3 = -np.sin(th_r)*np.sin(th) - np.cos(th_r)*np.cos(th)*np.cos(phi - phi_r)
	#-- rotated pole longitude and latitude of input model
	#-- convert back into degrees
	Xr = np.arctan2(R1,R2)*180.0/np.pi
	Yr = np.arcsin(R3)*180.0/np.pi
	#-- return the rotated coordinates
	return (Xr,Yr)

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
def racmo_interp_firn_height(base_dir, MODEL, COORDINATES=None,
	DATES=None, CSV=None):

	#-- get projection information from model shortname
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
	zs,air,itype,fv = interpolate_racmo_firn(base_dir, EPSG, MODEL, tdec, X, Y)
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
	print(' --csv=X\t\tRead dates and coordinates from a csv file\n')

#-- Main program that calls racmo_interp_firn_height()
def main():
	#-- Read the system arguments listed after the program
	long_options = ['help','directory=','smb=','coordinate=','date=','csv=']
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

	#-- run program with parameters
	racmo_interp_firn_height(base_dir, MODEL, COORDINATES=COORDINATES,
		DATES=DATES, CSV=CSV)

#-- run main program
if __name__ == '__main__':
	main()
