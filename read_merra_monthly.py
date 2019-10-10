#!/usr/bin/env python
u"""
read_merra_monthly.py
Written by Tyler Sutterley (01/2017)

Reads MERRA-2 monthly datafiles to calculate ice sheet surface mass balance
	converting from fluxes into monthly mass variables

From tavgM_2d_int collection:
	PRECCU (convective rain)
	PRECLS (large-scale rain)
	PRECSN (snow)
	and EVAP (evaporation)
from tavgM_2d_glc collection:
	RUNOFF (runoff over glaciated land)

INPUTS:
	SMB: Surface Mass Balance
	PRECIP: Total Precipitation
	RUNOFF: Meltwater Runoff

COMMAND LINE OPTIONS:
	--help: list the command line options
	--directory: working data directory
	-Y X, --year=X: years to run separated by commas
	-M X, --mode=X: Local permissions mode of the directories and files
	--clobber: Overwrite existing data in transfer

PYTHON DEPENDENCIES:
	numpy: Scientific Computing Tools For Python (http://www.numpy.org)
	netCDF4: Python interface to the netCDF C library
	 	(https://unidata.github.io/netcdf4-python/netCDF4/index.html)

PROGRAM DEPENDENCIES:
	convert_calendar_decimal.py: Return the decimal year for a calendar date

UPDATE HISTORY:
	Updated 01/2017: can output different data products (SMB, PRECIP, RUNOFF)
	Written 11/2016
"""
from __future__ import print_function

import sys
import os
import re
import time
import getopt
import numpy as np
import netCDF4
from convert_calendar_decimal import convert_calendar_decimal

#-- PURPOSE: read variables from MERRA-2 tavgM_2d_int and tavgM_2d_glc files
def read_merra_variables(merra_flux_file, merra_ice_surface_file):
	dinput = {}
	fill_value = {}
	#-- read each variable of interest in MERRA-2 flux file
	fileID = netCDF4.Dataset(merra_flux_file, 'r')
	dinput['lon'] = fileID.variables['lon'][:].copy()
	dinput['lat'] = fileID.variables['lat'][:].copy()
	for key in ['PRECCU','PRECLS','PRECSN','EVAP']:
		#-- Getting the data from each NetCDF variable of interest
		dinput[key] = np.asarray(fileID.variables[key][0,:,:].copy())
		fill_value[key] = fileID.variables[key]._FillValue
	fileID.close()
	#-- read each variable of interest in MERRA-2 ice surface file
	fileID = netCDF4.Dataset(merra_ice_surface_file, 'r')
	for key in ['RUNOFF']:
		#-- Getting the data from each NetCDF variable of interest
		dinput[key] = np.asarray(fileID.variables[key][0,:,:].copy())
		fill_value[key] = fileID.variables[key]._FillValue
	fileID.close()
	return dinput, fill_value

#-- PURPOSE: calculate number of seconds in month
def calc_total_seconds(YEAR,MONTH):
	#-- Rules in the Gregorian calendar for a year to be a leap year:
	#-- divisible by 4, but not by 100 unless divisible by 400
	#-- True length of the year is about 365.2422 days
	#-- Adding a leap day every four years ==> average 365.25
	#-- Subtracting a leap year every 100 years ==> average 365.24
	#-- Adding a leap year back every 400 years ==> average 365.2425
	#-- Subtracting a leap year every 4000 years ==> average 365.24225
	m4 = (YEAR % 4)
	m100 = (YEAR % 100)
	m400 = (YEAR % 400)
	m4000 = (YEAR % 4000)
	#-- days per month in a leap and a standard year
	#-- only difference is February (29 vs. 28)
	dpm_leap = np.array([31,29,31,30,31,30,31,31,30,31,30,31], dtype=np.float)
	dpm_stnd = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.float)
	#-- return number of seconds for month in standard years and leap years
	if ((m4 == 0) & (m100 != 0) | (m400 == 0) & (m4000 != 0)):
		return dpm_leap[MONTH-1]*24.0*60.0*60.0
	elif ((m4 != 0) | (m100 == 0) & (m400 != 0) | (m4000 == 0)):
		return dpm_stnd[MONTH-1]*24.0*60.0*60.0

#-- PURPOSE: read monthly MERRA-2 datasets to calculate monthly data
def read_merra_monthly(base_dir,PRODUCT,YY,MODE=0,CLOBBER=False,VERBOSE=False):
	#-- directory setup
	DIRECTORY = os.path.join(base_dir,'merra.dir')
	#-- setup output subdirectories
	SUBDIRECTORY = '{0}.5.12.4'.format(PRODUCT)
	if not os.access(os.path.join(DIRECTORY,SUBDIRECTORY,str(YY)), os.F_OK):
		os.makedirs(os.path.join(DIRECTORY,SUBDIRECTORY,str(YY)), MODE)

	#-- global grid parameters
	nlon = 576
	nlat = 361
	#-- output bad value
	fill_value = -9999.0
	#-- regular expression operator to find datafiles (and not the xml files)
	regex_pattern = 'MERRA2_(\d+).{0}.{1:4d}(\d+).nc4(?!.xml)'
	#-- sign for each product to calculate total SMB
	smb_sign = {'PRECCU':1.0,'PRECLS':1.0,'PRECSN':1.0,'EVAP':1.0,'RUNOFF':-1.0}
	#-- netcdf titles for each output data product
	merra_products = {}
	merra_products['SMB'] = 'MERRA-2 Surface Mass Balance'
	merra_products['PRECIP'] = 'MERRA-2 Precipitation'
	merra_products['RUNOFF'] = 'MERRA-2 Meltwater Runoff'

	#-- compile regular expression operator for PRODUCT and YEAR
	P1 = 'M2TMNXINT.5.12.4'
	P2 = 'M2TMNXGLC.5.12.4'
	rx = re.compile(regex_pattern.format('tavgM_2d_int_Nx',YY), re.VERBOSE)
	#-- find input files for PRODUCT
	f=[f for f in os.listdir(os.path.join(DIRECTORY,P1,str(YY))) if rx.match(f)]
	#-- for each input file
	for f1 in sorted(f):
		#-- extract month from input flux file
		N,MM = np.array(rx.findall(f1).pop(),dtype=np.int)
		#-- corresponding ice surface product file
		args = (N,'tavgM_2d_glc_Nx',YY,MM)
		f2 = 'MERRA2_{0:d}.{1}.{2:4d}{3:02d}.nc4'.format(*args)
		#-- full path for flux and ice surface files
		merra_flux_file = os.path.join(DIRECTORY,P1,str(YY),f1)
		merra_ice_surface_file = os.path.join(DIRECTORY,P2,str(YY),f2)
		if not os.access(merra_ice_surface_file,os.F_OK):
			raise Exception('File {0} not in file system'.format(f2))
		#-- output MERRA-2 data file for product
		args = (N,PRODUCT,YY,MM)
		FILE = 'MERRA2_{0:d}.tavgM_2d_{1}_Nx.{2:4d}{3:02d}.nc'.format(*args)
		merra_data_file = os.path.join(DIRECTORY,SUBDIRECTORY,str(YY),FILE)
		#-- if data file exists in file system: check if any Merra file is newer
		TEST = False
		OVERWRITE = ' (clobber)'
		MT12 = np.zeros((2))
		MT12[0] = os.stat(merra_flux_file).st_mtime
		MT12[1] = os.stat(merra_ice_surface_file).st_mtime
		#-- check if local version of file exists
		if os.access(merra_data_file,os.F_OK):
			#-- check last modification time of output file
			MT3 = os.stat(merra_data_file).st_mtime
			#-- if any input merra file is newer: overwrite the data file
			if (MT12 > MT3).any():
				TEST = True
				OVERWRITE = ' (overwrite)'
		else:
			TEST = True
			OVERWRITE = ' (new)'

		#-- if file does not exist locally, is to be overwritten or clobbered
		if TEST or CLOBBER:
			#-- if VERBOSE
			if VERBOSE:
				print('{0}{1}'.format(merra_data_file,OVERWRITE))
			#-- read netCDF4 files for variables of interest
			VV,FF = read_merra_variables(merra_flux_file,merra_ice_surface_file)
			#-- calculate mid-month time in year decimal using year and month
			tdec = convert_calendar_decimal(YY,MM)
			#-- calculate total seconds in month
			seconds = calc_total_seconds(YY,MM)
			#-- valid indice mask
			mask = np.ones((nlat,nlon),dtype=np.bool)
			for key in ['PRECCU','PRECLS','PRECSN','EVAP','RUNOFF']:
				ii,jj = np.nonzero(VV[key] == FF[key])
				mask[ii,jj] = False
			#-- valid indices for all variables
			indx,indy = np.nonzero(mask)
			#-- output data for product
			DATA = np.zeros((nlat,nlon))
			if (PRODUCT == 'SMB'):
				#-- calculate SMB and convert from flux to monthly
				for key in ['PRECCU','PRECLS','PRECSN','EVAP','RUNOFF']:
					val = VV[key]
					DATA[indx,indy] += val[indx,indy]*seconds*smb_sign[key]
			elif (PRODUCT == 'PRECIP'):
				#-- calculate precipitation and convert from flux to monthly
				for key in ['PRECCU','PRECLS','PRECSN']:
					val = VV[key]
					DATA[indx,indy] += val[indx,indy]*seconds
			elif (PRODUCT == 'RUNOFF'):
				#-- convert runoff from flux to monthly
				for key in ['RUNOFF']:
					val = VV[key]
					DATA[indx,indy] += val[indx,indy]*seconds

			#-- set invalid to fill_value
			ii,jj = np.nonzero(~mask)
			DATA[ii,jj] = fill_value
			#-- Writing output data to netcdf file
			ncdf_write(DATA, VV['lon'], VV['lat'], tdec, FILL_VALUE=fill_value,
				FILENAME=merra_data_file, LONGNAME='Equivalent Water Thickness',
				TITLE=merra_products[PRODUCT], UNITS='mm w.e.', VARNAME=PRODUCT)
			#-- change permissions of output file to specified mode
			os.chmod(merra_data_file, MODE)

#-- PURPOSE: writes COARDS-compliant NetCDF4 files
def ncdf_write(data, lon, lat, tim, FILENAME='sigma.H5',
	UNITS='cmH2O', LONGNAME='Equivalent_Water_Thickness',
	LONNAME='lon', LATNAME='lat', VARNAME='z', TIMENAME='time',
	TIME_UNITS='years', TIME_LONGNAME='Date_in_Decimal_Years',
	FILL_VALUE=None, TITLE = 'Spatial_Data', CLOBBER='Y', VERBOSE='N'):

	#-- setting NetCDF clobber attribute
	if CLOBBER in ('Y','y'):
		clobber = 'w'
	else:
		clobber = 'a'

	#-- opening NetCDF file for writing
	#-- Create the NetCDF file
	fileID = netCDF4.Dataset(FILENAME, clobber, format="NETCDF4")

	#-- Dimensions of parameters
	n_time = 1 if (np.ndim(tim) == 0) else len(tim)
	#-- Defining the NetCDF dimensions
	fileID.createDimension(LONNAME, len(lon))
	fileID.createDimension(LATNAME, len(lat))
	fileID.createDimension(TIMENAME, n_time)

	#-- defining the NetCDF variables
	nc = {}
	#-- lat and lon
	nc[LONNAME] = fileID.createVariable(LONNAME, lon.dtype, (LONNAME,))
	nc[LATNAME] = fileID.createVariable(LATNAME, lat.dtype, (LATNAME,))
	#-- spatial data
	if (n_time > 1):
		nc[VARNAME] = fileID.createVariable(VARNAME, data.dtype,
			(LATNAME,LONNAME,TIMENAME,), fill_value=FILL_VALUE, zlib=True)
	else:
		nc[VARNAME] = fileID.createVariable(VARNAME, data.dtype,
			(LATNAME,LONNAME,), fill_value=FILL_VALUE, zlib=True)
	#-- time (in decimal form)
	nc[TIMENAME] = fileID.createVariable(TIMENAME, 'f8', (TIMENAME,))

	#-- filling NetCDF variables
	nc[LONNAME][:] = lon
	nc[LATNAME][:] = lat
	if (n_time > 1):
		nc[VARNAME][:,:,:] = data
	else:
		nc[VARNAME][:,:] = data
	nc[TIMENAME][:] = tim

	#-- Defining attributes for longitude and latitude
	nc[LONNAME].long_name = 'longitude'
	nc[LONNAME].units = 'degrees_east'
	nc[LATNAME].long_name = 'latitude'
	nc[LATNAME].units = 'degrees_north'
	#-- Defining attributes for dataset
	nc[VARNAME].long_name = LONGNAME
	nc[VARNAME].units = UNITS
	#-- Defining attributes for date
	nc[TIMENAME].long_name = TIME_LONGNAME
	nc[TIMENAME].units = TIME_UNITS
	#-- global variable of NetCDF file
	fileID.TITLE = TITLE
	fileID.date_created = time.strftime('%Y-%m-%d',time.localtime())

	#-- Output NetCDF structure information
	if VERBOSE in ('Y','y'):
		print(FILENAME)
		print(list(fileID.variables.keys()))

	#-- Closing the NetCDF file
	fileID.close()

#-- PURPOSE: help module to describe the optional input parameters
def usage():
	print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
	print(' --directory=X\t\tWorking data directory')
	print(' -Y X, --year=X\t\tYears to run separated by commas')
	print(' -M X, --mode=X\t\tPermission mode of directories and files')
	print(' -C, --clobber\t\tOverwrite existing data')
	print(' -V, --verbose\t\tVerbose output of datafiles\n')

#-- Main program that calls read_merra_monthly()
def main():
	#-- Read the system arguments listed after the program
	long_options = ['help','year=','directory=','mode=','clobber','verbose']
	optlist,arglist = getopt.getopt(sys.argv[1:],'hY:M:CV',long_options)

	#-- command line parameters
	years = np.arange(1980,2017)
	base_dir = os.getcwd()
	#-- permissions mode of the local directories and files (number in octal)
	MODE = 0o775
	CLOBBER = False
	VERBOSE = False
	for opt, arg in optlist:
		if opt in ('-h','--help'):
			usage()
			sys.exit()
		elif opt in ("-Y","--year"):
			years = np.array(arg.split(','),dtype=np.int)
		elif opt in ("--directory"):
			base_dir = os.path.expanduser(arg)
		elif opt in ("-M","--mode"):
			MODE = int(arg, 8)
		elif opt in ("-C","--clobber"):
			CLOBBER = True
		elif opt in ("-V","--verbose"):
			VERBOSE = True

	#-- enter MERRA-2 Product as system argument
	if not arglist:
		raise Exception('No System Arguments Listed')

	#-- run program with parameters
	for PRODUCT in arglist:
		for YEAR in years:
			read_merra_monthly(base_dir, PRODUCT, YEAR, MODE=MODE,
				CLOBBER=CLOBBER, VERBOSE=VERBOSE)

#-- run main program
if __name__ == '__main__':
	main()
