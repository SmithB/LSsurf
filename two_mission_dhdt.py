#! /usr/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:12:48 2019

@author: ben
"""
from PointDatabase import geo_index, point_data, matlabToYear
from PointDatabase.mapData import mapData
from PointDatabase.ATL06_filters import segDifferenceFilter
from PointDatabase.check_ATL06_blacklist import check_ATL06_blacklist, check_rgt_cycle_blacklist
from LSsurf.RDE import RDE
from LSsurf.unique_by_rows import unique_by_rows
import argparse
import numpy as np
import matplotlib.pyplot as plt
from LSsurf.smooth_xyt_fit import smooth_xyt_fit
from LSsurf.twoSlope_dhdt import twoSlope_fit
from matplotlib.colors import Normalize
import scipy.interpolate as si
import os
import sys
import h5py
from glob import glob

def data_key(key):
    return {'ICESat1':1, 'ICESat2':2}[key]

def GI_files(hemisphere):
    if hemisphere==-1:
        GI_files={
        'ICESat1':'/Volumes/insar7/gmap/oib_database/glas/AA/rel_634_filtered_satCorr/geo_index.h5',
        'ICESat2':'/Volumes/ice2/ben/scf/AA_06/205/index/GeoIndex.h5'}
        GI_files['ICESat2']='/Volumes/ice2/ben/scf/AA_06/tiles/205_v1/GeoIndex.h5'
        return GI_files

def read_ICESat(xy0, W, gI, sensor=1):
    fields=[ 'IceSVar', 'deltaEllip', 'numPk', 'ocElv', 'reflctUC', 'satElevCorr',  'time',  'x', 'y', 'z','sensor']
    D0=gI.query_xy_box(xy0[0]+np.array([-W['x']/2, W['x']/2]), xy0[1]+np.array([-W['y']/2, W['y']/2]), fields=fields)
    # Per Adrian: Subtract 1.7 cm from laser 2, add 1.7 cm to laser 3 
    # these are the laser numbers in order:
    laser=np.array([1, 2,  2,  2, 3, 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2, 2]);
    # these are the biases.  Index biases by laser number, and there's your correction!
    bias_correction_vals=np.array([np.NaN, -0.017, +0.011])
    for ind, D in enumerate(D0):
        good=(D.IceSVar < 0.035) & (D.reflctUC >0.05) & (D.satElevCorr < 1) & (D.numPk==1)
        good=good.ravel()
        D.subset(good, datasets=['x','y','z','time'])
        D.assign({'sigma':np.zeros_like(D.x)+0.02, 'sigma_corr':np.zeros_like(D.x)+0.05})
        # 'sensor' in the matlab-h5 world is equivalent to campaign/100.  Let's use these as cycles
        D.assign({'cycle': -D.sensor*100})        
        # apply the Uyuni correcion
        D.z += bias_correction_vals[laser[-D.cycle.astype(int)]]
        
        # assign a fake RGT, BP, LR number
        D.assign({'rgt':-100+np.zeros_like(D.x)})
        D.assign({'BP':np.zeros_like(D.x)})
        D.assign({'LR':np.zeros_like(D.x)})
        D.assign({'sensor':np.zeros_like(D.x)+sensor})
        D.assign({'year':(D.time-737061)/365.25+2018})
        D0[ind]=D
    return D0

def reconstruct_tracks(D):
    if ('cycle' in D.list_of_fields):
        _, bin_dict=unique_by_rows(np.c_[D.cycle, D.rgt, D.BP, D.LR], return_dict=True)
    else:
        _, bin_dict=unique_by_rows(np.c_[D.cycle_number, D.rgt, D.BP, D.LR], return_dict=True)
    D0=[]
    for key in bin_dict:
        if "delta_time" in D.list_of_fields:
            ind=np.argsort(D.delta_time[bin_dict[key]])
        else:
            ind=np.argsort(D.time[bin_dict[key]])
        this_D=D.subset(bin_dict[key][ind])
        D0.append(this_D)   
    return D0


def read_ICESat2(xy0, W, gI_file, sensor=2, SRS_proj4=None, tiled=True, seg_diff_tol=2):
    field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude','atl06_quality_summary','segment_id','sigma_geo_h'], 
                'fit_statistics':['dh_fit_dx'],
                'ground_track':['x_atc'],
                'geophysical' : ['dac'],
                'orbit_info':['rgt','cycle_number'],
                'derived':['valid','matlab_time', 'BP','LR']}
    if tiled:
        fields=[]
        for key in field_dict:
            fields += field_dict[key]
    else:
        fields=field_dict
    px, py=np.meshgrid(np.arange(xy0[0]-W['x']/2, xy0[0]+W['x']/2+1.e4, 1.e4),np.arange(xy0[1]-W['y']/2, xy0[1]+W['y']/2+1.e4, 1.e4) )  
    D0=geo_index().from_file(gI_file).query_xy((px.ravel(), py.ravel()), fields=fields)
    # check the D6 filenames against the blacklist of bad files
    if tiled:
        D0=reconstruct_tracks(point_data(list_of_fields=fields).from_list(D0))
    blacklist=None
    D1=list()
    for ind, D in enumerate(D0):
        if D.size<2:
            continue
        delete_file, blacklist=check_rgt_cycle_blacklist(rgt_cycle=[D.rgt[0], D.cycle_number[0]],  blacklist=blacklist)
        if delete_file==0:
            D1.append(D)
        else:
            print("deleting %s" % D.filename)
            
    for ind, D in enumerate(D1):
        try:
            segDifferenceFilter(D, setValid=False, toNaN=True, tol=seg_diff_tol)
        except TypeError as e:
            print("HERE")          
            print(e)
        D.h_li[D.atl06_quality_summary==1]=np.NaN
        # rename the h_li field to 'z', and rename the 'matlab_time' to 'day'
        D.assign({'z': D.h_li+D.dac,'time':D.matlab_time,'sigma':D.h_li_sigma,'sigma_corr':D.sigma_geo_h,'cycle':D.cycle_number})
        D.assign({'year':D.delta_time/24./3600./365.25+2018})
        # thin to 40 m
        D.index(np.mod(D.segment_id, 2)==0)       
        D.get_xy(SRS_proj4)
        #D.index(np.isfinite(D.h_li), list_of_fields=['x','y','z','time','year','sigma','sigma_corr','rgt','cycle'])
                
        D.assign({'sensor':np.zeros_like(D.x)+sensor})
        D1[ind]=D.subset(np.isfinite(D.h_li), datasets=['x','y','z','time','year','sigma','sigma_corr','rgt','cycle','sensor', 'BP','LR'])
    return D1

def reread_data_from_files(xy0, W, thedir, template='E%d_N%d.h5'):
    d_i, d_j = np.meshgrid([-1., 0, 1.], [-1., 0., 1.])
    data_list=list()
    #plt.figure()
    index_list=[]
    for ii, jj in zip(d_i.ravel(), d_j.ravel()):
        this_xy=[xy0[0]+W/2*ii, xy0[1]+W/2*jj]
        this_file=thedir+'/'+template % (this_xy[0]/1000, this_xy[1]/1000)
        if os.path.isfile(this_file):
            with h5py.File(this_file,'r') as h5f:
                this_data=dict()
                for key in h5f['data'].keys():
                    this_data[key]=np.array(h5f['data'][key])
            this_data=point_data(list_of_fields=this_data.keys()).from_dict(this_data)       
            these=(np.abs(this_data.x-xy0[0])<W/2) & \
                (np.abs(this_data.y-xy0[1])<W/2)
            this_data.index(these) # & (this_data.three_sigma_edit))
            data_list.append(this_data)
            this_index={}
            Lb=1.e4
            this_index['xyb']=np.round((this_data.x+1j*this_data.y-(1+1j)*Lb/2)/Lb)*Lb
            this_index['xyb0']=np.unique(this_index['xyb'])
            this_index['dist0']=np.abs(this_index['xyb0']-(this_xy[0]+1j*this_xy[1]))
            this_index['N']=len(data_list)-1+np.zeros_like(this_index['xyb0'], dtype=int)
            index_list.append(this_index) 
    index={key:np.concatenate([item[key] for item in index_list]) for key in ['xyb0', 'dist0','N']}
    bins=np.unique(index['xyb0'])
    data_sub_list=[]
    for this_bin in bins:
        these=np.where(index['xyb0']==this_bin)[0]
        this=index['N'][these[np.argmin(index['dist0'][these])]]
        data_sub_list.append(data_list[this].subset(index_list[this]['xyb']==this_bin))
    fields=data_list[0].list_of_fields
    return point_data(list_of_fields=fields).from_list(data_sub_list)
    #return reconstruct_tracks(point_data(list_of_fields=fields).from_list(data_sub_list))

def read_OIB_data(xy0, W, reread_file=None, hemisphere=-1, blockmedian_scale=100, SRS_proj4=None):
    """
    Read indexed files for standard datasets
    """
    sensor_dict={1:'ICESat1', 2:'ICESat2'}
    if reread_file is None:
        GI=geo_index().from_file(GI_files(hemisphere)['ICESat2'], read_file=False)
        D = read_ICESat2(xy0, W, GI,  SRS_proj4=SRS_proj4)
        GI=None    
        GI=geo_index().from_file(GI_files(hemisphere)['ICESat1'], read_file=False)
        D += read_ICESat(xy0, W, GI)    
        GI=None
        data=point_data(list_of_fields=['x','y','z','time','sigma','sigma_corr','sensor','cycle','rgt', 'BP','LR']).from_list(D)
        data.assign({'day':np.floor(data.time)})
        data.time=matlabToYear(data.time)
        for field in data.list_of_fields:
            setattr(data, field, getattr(data, field).astype(np.float64))
        data.index(np.isfinite(data.z) & np.isfinite(data.sigma_corr) & np.isfinite(data.sigma))
    else:
        data=dict()
        with h5py.File(reread_file,'r') as h5f:
            for key in h5f['data'].keys():
                data[key]=np.array(h5f['data'][key])
        data=point_data(list_of_fields=data.keys()).from_dict(data)
    return data, sensor_dict

def save_fit_to_file(S,  filename, sensor_dict=None):
    if os.path.isfile(filename):
        os.remove(filename)
    with h5py.File(filename,'w') as h5f:
        h5f.create_group('/data')
        for key in S['data'].list_of_fields:
            h5f.create_dataset('/data/'+key, data=getattr(S['data'], key))
        h5f.create_group('/meta')
        h5f.create_group('/meta/timing')
        for key in S['timing']:
            h5f['/meta/timing/'].attrs[key]=S['timing'][key]
        if sensor_dict is not None:
            h5f.create_group('meta/sensors')
            for key in sensor_dict:
                h5f['/meta/sensors'].attrs['sensor_%d' % key]=sensor_dict[key]
        for this_key in ['z0','dz','dzdt1','dzdt2']:
            if this_key not in S['m']:
                continue
            h5f.create_group('/'+this_key)
            for ii, name in enumerate(['y','x','t']):
                try:
                    h5f.create_dataset('/'+this_key+'/'+name, data=S['grids'][this_key].ctrs[ii])
                except: 
                    pass
            h5f.create_dataset('/'+this_key+'/z', data=S['m'][this_key])
        if 'RMS' in S:
            h5f.create_group('/RMS')
            for key in S['RMS']:
                h5f.create_dataset('/RMS/'+key, data=S['RMS'][key])
        if 'E_RMS' in S:
            h5f.create_group('E_RMS')
            for key in S['E_RMS']:
                if S['E_RMS'][key] is not None:
                    h5f.create_dataset('/E_RMS/'+key, data=S['E_RMS'][key])
        for key in S['m']['bias']:
            h5f.create_dataset('/bias/'+key, data=S['m']['bias'][key])
    return

def read_fit(filename, fields=['data','z0','dz']):
    S={}
    if 'data' in fields:
        S['data']=point_data().from_file(filename, group='/data')
    if 'z0' in fields:
        S['z0']=mapData().from_h5(filename, group='/z0/', field_mapping={'z':'z0'})
    if 'z0' in fields:
        S['dz']=mapData().from_h5(filename, group='/dz/', field_mapping={'z':'dz'})
        S['dz'].z=S['dz'].z[:,:,0]
    if 'bias' in fields:
        S['bias']={}
        with h5py.File(filename,'r') as h5f:
            for key in h5f['bias'].keys():
                S['bias'][key]=np.array(h5f['bias'][key])
    return S

def DEM_edit(data, xy0, W, sigma_min=1., hemisphere=-1):
    DEM=mapData().from_geotif('/Volumes/insar9/ben/REMA/REMA_200m_dem_filled.tif', bounds=[xy0[0]+np.array([-W/2, W/2]), xy0[1]+np.array([-W/2, W/2])] )
    mI=si.RectBivariateSpline(DEM.y[::-1], DEM.x, DEM.z[::-1, :])
    dz=data.z-mI.ev(data.y, data.x)
    #dz=data.z-np.array([mI(data.x[ii], data.y[ii]) for ii in range(len(data.x))]).ravel()
    valid=np.ones_like(data.x, dtype=bool)
    for sensor in np.unique(data.sensor):
        these=(data.sensor==sensor) & np.isfinite(data.z)
        if 'valid' in data.list_of_fields:
            these=these & (data.valid==1)
        this_sigma=np.maximum(RDE(dz[these]), sigma_min)
        this_med=np.nanmedian(dz[these])
        toMask=(data.sensor==sensor ) & np.isfinite(dz)
        valid[toMask] = np.abs(dz[toMask]-this_med) < 3*this_sigma
    return valid


def make_queue( args, file_template='/E%d_N%d.h5', replace=False, reread=False):
    if args.hemisphere==-1:
        #mask_G=mapData().from_geotif('/Volumes/insar5/gmap/OIB_data/AA/masks/MOA_2009_grounded.tif')
        mask_G=mapData().from_geotif('/Volumes/ice1/ben/MOA/moa_2009_5km.tif')
        mask_G.z=(mask_G.z>100)
        #xg, yg=np.meshgrid(np.arange(-2500, 0, args.W/2000.)*1000, np.arange(-600, 600, args.W/2000.)*1000)
        xg, yg=np.meshgrid(np.arange(-2500, 2800, args.W/2000.)*1000, np.arange(-2500, 2500, args.W/2000.)*1000)
        mI=si.interp2d(mask_G.x, mask_G.y, mask_G.z.astype(np.float64))
        maski=mI(xg[1,:], yg[:,1]).ravel()>0.5
        # keep everything north of -86 degrees
        maski = maski & ((xg.ravel()**2+yg.ravel()**2) > 4.3478e+05**2)
        # keep everything south of ~73 degrees
        #maski = maski & ((xg.ravel()**2+yg.ravel()**2) < 2.e+06**2)
        out_template= 'two_mission_dhdt %d %d %d'# -D -r %s -o %s\n       
        if args.DEM_edit is not None:
            out_template += ' -D'
        if args.reread_dir is not None:
            out_template += ' -r '+args.reread_dir
        if args.scale_resolution > 1:
            out_template += ' -s %1.1f' % args.scale_resolution
        if args.scale.dz_scale != 1:
            out_template += ' -d %2.2f' % args.dz_scale
        out_template += (' -i %d' % (args.iteration_max))
        out_template += ' -o '+args.out_dir
        out_template +='\n'
        xg=xg.ravel()[maski]
        yg=yg.ravel()[maski]
        
        with open(args.queue_file,'w') as qF:
            for xy in zip(xg, yg):
                out_file=args.out_dir+file_template % (xy[0]/1000, xy[1]/1000)
                # queue_this will be true if the file does not exist
                queue_this = os.path.isfile(out_file) == False
                if not replace:
                    if reread and not queue_this:
                        try:
                            with h5py.File(out_file,'r') as h5f:
                                assert('dz/dz' in h5f)
                        except (OSError, AssertionError):
                            print('requeue: %s' % out_file)
                            queue_this=True
                            
                if queue_this:
                    # -800000 200000 100000 -D -o /Volumes/ice2/ben/scf/intermission_south/V4 -r /Volumes/ice2/ben/scf/intermission_south/                
                    qF.write(out_template % (xy[0], xy[1], args.W))
    return                

def collect_biases(thedir):
    files=glob(thedir+'/*.h5')
    bias_dict={}
    for file in files:
        B=read_fit(file,['bias'])['bias']
        worst=np.argmax(np.abs(B['val']))
        vWorst=B['val'][worst]
        if np.abs(vWorst) > 10:
            TC=(B['rgt'][worst], B['cycle'][worst])        
            if TC not in bias_dict:
                bias_dict[TC]={'bias':[vWorst], 'file':[file]}
            else:
                bias_dict[TC]['bias'].append(vWorst)
                bias_dict[TC]['file'].append(file)
    return bias_dict
     
#----------------Command-line interface: 
def main(argv):
    if isinstance(argv,dict):
        args=argv
    else:
        # account for a bug in argparse that misinterprets negative agruents
        for i, arg in enumerate(argv):
            if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg
        
        parser=argparse.ArgumentParser(argv)
        parser.add_argument('x0', type=float, default=-1570000)
        parser.add_argument('y0', type=float, default=-220000)
        parser.add_argument('W', type=float, default=4e4)
        parser.add_argument('-n', '--no_twoslope', action="store_true")
        parser.add_argument('-d', '--dz_scale', type=float, default=1)
        parser.add_argument('-z', '--z0_scale', type=float, default=1)
        parser.add_argument('-o', '--out_dir', type=str, default='/Volumes/ice2/ben/scf/intermission_south/')
        parser.add_argument('--hemisphere', type=int, default=-1)
        parser.add_argument('-q', '--queue_file', type=str, default=None)
        parser.add_argument('-r', '--reread_dir', type=str, default=None)
        parser.add_argument('-k', '--keep', action='store_true')
        parser.add_argument('-D', '--DEM_edit',  action='store_true')
        parser.add_argument('-i', '--iteration_max', type=int, default=6)
        parser.add_argument('-t', '--time_span', type=float, nargs=3, default=[2003, 2020])
        parser.add_argument('-g', '--grid_resolution', type=float, nargs=3, default=[250, 4000, 17])
        args=parser.parse_args()
    
    xy0=(args.x0, args.y0)
    
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    
    if args.queue_file is not None:
        make_queue(args, replace=False, reread=False)
        return
        
    out_template=args.out_dir+'/E%d_N%d.h5'
    spacing={'z0':args.grid_resolution[0], 'dz':args.grid_resolution[1], 'time':args.grid_resolution[2]}
    sensor_dict={1:'ICESat1', 2:'ICESat2'}
    print("working on %d_%d" % xy0)
    if args.hemisphere==1:
        SRS_proj4='+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    else:
        SRS_proj4='+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    
    E_RMS0={'d2z0_dx2':200000./3000/3000*args.z0_scale, 'd3z_dx2dt':100./3000/3000*args.dz_scale, 'd2z_dxdt':100/3000*args.dz_scale, 'd2z_dt2':None}
    out_name=out_template %(xy0[0]/1000, xy0[1]/1000)
    print("output file is %s" % out_name)
    if args.keep is True and os.path.isfile(out_name):
            return None,None
    W={'x':args.W, 'y':args.W,'t':np.diff(args.time_span)}
    ctr={'x':xy0[0], 'y':xy0[1], 't':np.mean(args.time_span)}
    if args.reread_dir is None:
        data, sensor_dict=read_OIB_data(xy0, W, blockmedian_scale=100., SRS_proj4=SRS_proj4, hemisphere=args.hemisphere) 
    else:
        data=reread_data_from_files(xy0, args.W, args.reread_dir, template='E%d_N%d.h5')
    print("done reading data")
    if args.DEM_edit:
        valid=DEM_edit(data, xy0, args.W, hemisphere=args.hemisphere)
        data.index(valid==1)
    # run the fit -- try repeat_res=None (changed from 250 m)
    if args.no_twoslope:
        S=smooth_xyt_fit(data=data, ctr=ctr, W=args.W, spacing=spacing, E_RMS=E_RMS0, reference_epoch=1,\
            N_subset=None, compute_E=False, dzdt_lags=[1], bias_params=['rgt','cycle'], \
            repeat_res=None, max_iterations=args.max_iterations, srs_WKT=SRS_proj4,  VERBOSE=True)
    else:
        S=dict()
        S['grids'], S['m'], S['data'], S['timing'] = twoSlope_fit([ctr['x'], ctr['y']], args.W, spacing, [2003., 2010., 2020.], sensor_dict={'ICESat1':1, 'ICESat2':2}, D=data, E_RMS=E_RMS0)
        
    if out_name is not None:
        save_fit_to_file(S, out_name, sensor_dict=sensor_dict)
    return S, sensor_dict

if __name__=="__main__":
    main(sys.argv)
