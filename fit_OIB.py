# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:12:48 2019

@author: ben
"""
import numpy as np
import matplotlib.pyplot as plt
from LSsurf.smooth_xyt_fit import smooth_xyt_fit
from LSsurf.read_ICESat2 import read_ICESat2
from PointDatabase import geo_index, point_data, matlabToYear, mapData
from LSsurf.assign_firn_correction import assign_firn_correction
from LSsurf.reread_data_from_fits import reread_data_from_fits
from LSsurf.subset_DEM_stack import subset_DEM_stack
from matplotlib.colors import Normalize

import os
import h5py
import sys
import glob

def get_SRS_proj4(hemisphere):
    if hemisphere==1:
        return '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    else:
        return '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

def laser_key():
    return {'ICESat1':1, 'ICESat2':2, 'ATM':3, 'LVIS':4}

def GI_files(hemisphere):
    if hemisphere==1:
        GI_files={'ATM':'/Volumes/insar10/ben/OIB_database/ATM_Qfit/GL/geo_index.h5',
                  'LVIS':'/Volumes/insar10/ben/OIB_database/LVIS/GL/geo_index.h5',
                  'ICESat1':'/Volumes/insar10/ben/OIB_database/glas/GL/rel_634/GeoIndex.h5',
                  'ICESat2':'/Volumes/ice2/ben/scf/GL_06/tiles/001/GeoIndex.h5',
                  'DEM':'/Volumes/insar7/ben/ArcticDEM/geocell_v3/GeoIndex.h5' };
        return GI_files

def read_ICESat(xy0, W, gI_file, sensor=1):
    fields=[ 'IceSVar', 'deltaEllip', 'numPk', 'ocElv', 'reflctUC', 'satElevCorr',  'time',  'x', 'y', 'z']
    D0=geo_index().from_file(gI_file).query_xy_box(xy0[0]+np.array([-W['x']/2, W['x']/2]), xy0[1]+np.array([-W['y']/2, W['y']/2]), fields=fields)
    for ind, D in enumerate(D0):
        good=(D.IceSVar < 0.035) & (D.reflctUC >0.05) & (D.satElevCorr < 1) & (D.numPk==1)
        good=good.ravel()
        D.subset(good, datasets=['x','y','z','time'])
        D.assign({'sigma':np.zeros_like(D.x)+0.02, 'sigma_corr':np.zeros_like(D.x)+0.25})
        D.assign({'sensor':np.zeros_like(D.x)+sensor})
        D0[ind]=D
    return D0

def read_ATM(xy0, W, gI_file, sensor=3, blockmedian_scale=100.):
    fields=['x','y','z', 'time','bias_50m', 'noise_50m', 'N_50m','slope_x','slope_y']
    if W is not None:
        xg, yg=np.meshgrid(xy0[0]+np.arange(-W['x']/2, W['x']/2+1.e4, 1.e4), xy0[1]+np.arange(-W['y']/2, W['y']/2+1.e4, 1.e4))
    else:
        xg=np.array([xy0[0]])
        yg=np.array([xy0[1]])

    D0=geo_index().from_file(gI_file).query_xy([xg.ravel(), yg.ravel()], fields=fields)
    if D0 is None:
        return D0
    for ind, D in enumerate(D0):
        for field in D.list_of_fields:
            setattr(D, field, getattr(D, field).ravel())
        good=np.isfinite(D.bias_50m) & (D.N_50m > 20) & (np.abs(D.bias_50m) < 20)
        slope_mag=np.sqrt(D.slope_x**2 + D.slope_y**2)
        good=good & (D.bias_50m < 0.5) & (slope_mag < 6*np.pi/180) & (np.abs(D.bias_50m) < 10)
        D.assign({'sigma': np.sqrt((4*slope_mag)**2+D.noise_50m**2+0.05**2),
                     'sigma_corr':0.025+np.zeros_like(D.time)})
        good = good & np.isfinite(D.sigma+D.sigma_corr)
        good=good.ravel()
        D.assign({'sensor':np.zeros_like(D.time)+sensor})
        D0[ind]=D.subset(good, datasets=['x','y','z','time','sigma','sigma_corr','sensor'])
        if blockmedian_scale is not None:
            D0[ind].blockmedian(blockmedian_scale)
    return D0

def read_LVIS(xy0, W, gI_file, sensor=4, blockmedian_scale=100):
    fields={'x','y','z', 'zb', 'time','bias_50m', 'noise_50m','slope_x','slope_y'}
    D0=geo_index().from_file(gI_file).query_xy_box(xy0[0]+np.array([-W['x']/2, W['x']/2]), xy0[1]+np.array([-W['y']/2, W['y']/2]), fields=fields)
    if D0 is None:
        return [None]
    for ind, D in enumerate(D0):
        # LVIS data have the wrong sign on their 'bias' field
        D.bias_50m *=-1
        slope_mag=np.sqrt(D.slope_x**2 + D.slope_y**2)
        good = (D.bias_50m < 0.5) & (slope_mag < 6*np.pi/180) & (np.abs(D.bias_50m) < 10)
        good=good.ravel()
        if not np.any(good):
            return None
        D.assign({  'sigma': np.sqrt((4*slope_mag)**2+D.noise_50m**2+0.05**2),
                    'sigma_corr':0.025+np.zeros_like(D.time)})
        if D.size==D.zb.size:
            # some LVIS data have a zb field, which is a better estimator of surface elevation than the 'z' field
            D.z=D.zb
        D.assign({'sensor':np.zeros_like(D.time)+sensor})
        D0[ind]=D.subset(good, datasets=['x','y','z','time','sigma','sigma_corr','sensor'])
        if D0[ind].size > 5:
            D0[ind].blockmedian(blockmedian_scale)
    return D0

def read_data(xy0, W, hemisphere=1, blockmedian_scale=100, laser_only=False):
    laser_dict=laser_key()
    D = read_ICESat2(xy0, W, GI_files(hemisphere)['ICESat2'],  SRS_proj4=get_SRS_proj4(hemisphere), blockmedian_scale=blockmedian_scale,
                     sensor=laser_dict['ICESat2'])
    try:
        D_IS = read_ICESat(xy0, W, GI_files(hemisphere)['ICESat1'], sensor=laser_key()['ICESat1'])
        if D_IS is not None:
            D += D_IS
    except Exception as e:
        print("problem with ICESat-1")
        print(e)
    #try:
    if True:
        D_LVIS=read_LVIS(xy0, W, GI_files(hemisphere)['LVIS'], blockmedian_scale=100., sensor=laser_dict['LVIS'])
        if D_LVIS is not None:
            D += D_LVIS
    #except Exception as e:
    #    print("Problem with LIVS:" )
    #    print(e)
    try:
        D_ATM = read_ATM(xy0, W, GI_files(hemisphere)['ATM'], blockmedian_scale=100., sensor=laser_dict['ATM'])
        if D_ATM is not None:
            D += D_ATM
    except Exception as e:
        print("Problem with ATM")
        print(e)
        
    sensor_dict={laser_dict[key]:key for key in ['ICESat1', 'ICESat2', 'ATM','LVIS']}
    if not laser_only:
        D_DEM, sensor_dict = read_DEM_data(xy0, W, sensor_dict, \
                                                      GI_file=GI_files(hemisphere)['DEM'], \
                                                      hemisphere=hemisphere, \
                                                      blockmedian_scale=blockmedian_scale)
        D += D_DEM
    return D, sensor_dict

def read_DEM_data(xy0, W, sensor_dict, GI_file=None, hemisphere=1, sigma_corr=20., blockmedian_scale=100.):

    D = geo_index().from_file(GI_file, read_file=False).query_xy_box(xy0[0]+np.array([-W['x']/2, W['x']/2]), xy0[1]+np.array([-W['y']/2, W['y']/2]), fields=['x','y','z','sigma','time','sensor'])
    if len(sensor_dict) > 0:
        first_key_num=np.max([key for key in sensor_dict.keys()])+1
    else:
        first_key_num=0
    key_num=0
    temp_sensor_dict=dict()
    for Di in D:
        temp_sensor_dict[key_num]=os.path.basename(Di.filename)
        if blockmedian_scale is not None:
            Di.blockmedian(blockmedian_scale)
        Di.assign({'sensor':np.zeros_like(Di.x)+key_num})
        Di.assign({'sigma_corr':np.zeros_like(Di.x)+sigma_corr})
        key_num += 1
    
    # subset the DEMs so that there is about one per year
    DEM_number_list=subset_DEM_stack(D, xy0, W['x'], bin_width=400)
    new_D=[]
    for count, num in enumerate(DEM_number_list):
        new_D += [D[num]]
        new_D[-1].sensor[:]=count+first_key_num
        sensor_dict[count+first_key_num] = temp_sensor_dict[num]
   
    return new_D, sensor_dict

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
        # this is how far we'll get if we're just in 'edit only' mode
        if 'dz' not in S['m']:
            return
        h5f.create_group('/dz')
        for ii, name in enumerate(['y','x','t']):
            h5f.create_dataset('/dz/'+name, data=S['grids']['dz'].ctrs[ii])
        h5f.create_dataset('/dz/dz', data=S['m']['dz'])
        h5f.create_group('/z0')
        for ii, name in enumerate(['y','x']):
            h5f.create_dataset('/z0/'+name, data=S['grids']['z0'].ctrs[ii])
        h5f.create_dataset('/z0/z0', data=S['m']['z0'])
        h5f.create_group('/RMS')
        for key in S['RMS']:
            h5f.create_dataset('/RMS/'+key, data=S['RMS'][key])
        h5f.create_group('E_RMS')
        for key in S['E_RMS']:
            h5f.create_dataset('E_RMS/'+key, data=S['E_RMS'][key])
        for key in S['m']['bias']:
            h5f.create_dataset('/bias/'+key, data=S['m']['bias'][key])
        if 'slope_bias' in S['m']:
            sensors=np.array(list(S['m']['slope_bias'].keys()))
            h5f.create_dataset('/slope_bias/sensors', data=sensors)
            x_slope=[S['m']['slope_bias'][key]['slope_x'] for key in sensors]
            y_slope=[S['m']['slope_bias'][key]['slope_y'] for key in sensors]
            h5f.create_dataset('/slope_bias/x_slope', data=np.array(x_slope))
            h5f.create_dataset('/slope_bias/y_slope', data=np.array(y_slope))
    return

def make_map(file=None, glob_str=None, t_slice=[0, -1], caxis=[-1, 1], spacing=np.Inf ):
    files=[]
    if file is not None:
        files += [file]
    if glob_str is not None:
        files += glob.glob(glob_str)
    h=[]
    zero_files=[]
    for file in files:
        try:
            D=mapData().from_h5(file, group='/dz/', field_mapping={'z':'dz'})
            if np.isfinite(spacing):
                w01=np.array([-0.5, 0.5])*spacing
                D.subset(np.mean(D.x)+w01 , np.mean(D.y)+w01)
                h.append(plt.imshow((D.z[:,:,t_slice[1]]-D.z[:,:,t_slice[0]])/(D.t[t_slice[1]]-D.t[t_slice[0]]), \
                            extent=D.extent, vmin=caxis[0], vmax=caxis[1], label=file, origin='lower',
                            cmap="Spectral"))
                if np.all(D.z.ravel()==0):
                    #print("All-zero dz for file = %s" % file)
                    plt.plot(0.5*(D.extent[0]+D.extent[1]), 0.5*(D.extent[2]+D.extent[3]),'k*')
                    zero_files.append(file)
        except OSError as e:
            print(e)
            pass
        except KeyError as e:
            print("Error for file "+file)
            print(e)

    plt.axis('tight');
    plt.axis('equal');
    plt.show()
    return h, zero_files


def fit_OIB(xy0, Wxy=4e4, E_RMS={}, t_span=[2003, 2020], spacing={'z0':2.5e2, 'dz':5.e2, 'dt':0.5},  hemisphere=1, reference_epoch=None, reread_dirs=None, N_subset=8, Edit_only=False, sensor_dict={}, out_name=None, replace=False, DOPLOT=False, spring_only=False, laser_only=False, firn_correction=False):
    """
        Wrapper for smooth_xyt_fit that can find data and set the appropriate parameters
    """
    print("fit_OIB: working on %s" % out_name)

    SRS_proj4=get_SRS_proj4(hemisphere)

    E_RMS0={'d2z0_dx2':200000./3000/3000, 'd3z_dx2dt':3000./3000/3000, 'd2z_dxdt':3000/3000, 'd2z_dt2':5000}
    E_RMS0.update(E_RMS)

    W={'x':Wxy, 'y':Wxy,'t':np.diff(t_span)}
    ctr={'x':xy0[0], 'y':xy0[1], 't':np.mean(t_span)}
    if out_name is not None:
        try:
            out_name=out_name %(xy0[0]/1000, xy0[1]/1000)
        except:
            pass
        if replace is False and os.path.isfile(out_name):
            return None,None

    if reread_dirs is None:       
        D, sensor_dict=read_data(xy0, W, blockmedian_scale=100., laser_only=False)

        for ind, Di in enumerate(D):
            if Di is None:
                continue
            if 'rgt' not in Di.list_of_fields:
                Di.assign({'rgt':np.zeros_like(Di.x)+np.NaN})
            if 'cycle' not in Di.list_of_fields:
                Di.assign({'cycle':np.zeros_like(Di.x)+np.NaN})
        data=point_data(list_of_fields=['x','y','z','time','sigma','sigma_corr','sensor','rgt','cycle']).from_list(D)
        data.assign({'day':np.floor(data.time)})
        # smooth_xyt_fit needs time in years, so reassign time from matlab days to years.
        data.time=matlabToYear(data.time)
    else:
        data, sensor_dict = reread_data_from_fits(xy0, Wxy, reread_dirs, template='E%d_N%d.h5')
        N_subset=None
    laser_sensors=[item for key, item in laser_key().items()]
    DEM_sensors=np.array([key for key in sensor_dict.keys() if key not in laser_sensors ])
    if reference_epoch is None:
        reference_epoch=len(np.arange(t_span[0], t_span[1], spacing['dt']))

    if spring_only:
        # edit out summer data
        data_ID_dict={item:key for key, item in sensor_dict.items()}
        good=np.ones_like(data.x, dtype=bool)
        # edit out summer data from ATM and LVIS
        good[(np.mod(data.time, 1)>0.6) & np.in1d(data.sensor, [data_ID_dict['LVIS'], data_ID_dict['ATM']])]=False
        data.index(good)
    for field in data.list_of_fields:
        setattr(data, field, getattr(data, field).astype(np.float64))
 
    if reread_dirs is None and firn_correction is not None:
        assign_firn_correction(data, firn_correction, hemisphere)
    #report data counts
    vals=[]
    for val, sensor in sensor_dict.items():
        if val < 5:
            print("for %s found %d data" %(sensor, np.sum(data.sensor==val)))
            vals += [val]
    print("for DEMs, found %d data" % np.sum(np.in1d(data.sensor, np.array(vals))==0))

    # run the fit
    S=smooth_xyt_fit(data=data, ctr=ctr, W=W, spacing=spacing, E_RMS=E_RMS0,
                     reference_epoch=reference_epoch, N_subset=N_subset, compute_E=False,
                     bias_params=['day','sensor'], repeat_res=250, max_iterations=4, srs_WKT=SRS_proj4,
                     VERBOSE=True, Edit_only=Edit_only, data_slope_sensors=DEM_sensors)
    if Edit_only:
        print('N_subsets=%d, t=%f' % ( N_subset, S['timing']['edit_by_subset']))

    if out_name is not None:
        save_fit_to_file(S, out_name, sensor_dict)
    if DOPLOT:
        plt.figure(); plt.clf()
        plt.imshow(np.gradient(S['m']['z0'], S['grids']['z0'].delta[0])[1], extent=S['m']['extent'], cmap=plt.cm.gray, origin='lower');
        cmap=plt.cm.RdYlBu
        alpha=0.5
        dzImage=(S['m']['dz'][:,:,-1]-S['m']['dz'][:,:,0])/(S['grids']['dz'].ctrs[2][-1]-S['grids']['dz'].ctrs[2][0])
        dzIm4=cmap(Normalize(-15, 15, clip=True)(dzImage))
        dzIm4[...,-1]=alpha
        plt.imshow( dzIm4, extent=S['m']['extent'],  origin='lower');
        plt.title(out_name)
    return S, data, sensor_dict

def main(argv):
    # account for a bug in argparse that misinterprets negative agruents
    for i, arg in enumerate(argv):
        if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg

    import argparse
    parser=argparse.ArgumentParser(description="function to fit icebridge data with a smooth elevation-change model", \
                                   fromfile_prefix_chars="@")
    parser.add_argument('xy0', type=float, nargs=2, help="fit center location")
    parser.add_argument('--Width','-W',  type=float, help="fit width")
    parser.add_argument('--time_span','-t', type=str, help="time span, first year,last year AD (comma separated, no spaces)")
    parser.add_argument('--grid_spacing','-g', type=str, help='grid spacing:DEM (meters),dh maps xy (meters),dh_maps time (years): comma-separated, no spaces', default='250.,4000.,1.')
    parser.add_argument('--Hemisphere','-H', type=int, default=1, help='hemisphere: -1=Antarctica, 1=Greenland')
    parser.add_argument('--base_directory','-b', type=str, help='base directory')
    parser.add_argument('--out_name', '-o', type=str, help="output file name")
    parser.add_argument('--Replace', '-R', type=str, help="replace (1=yes, 0=no)")
    parser.add_argument('--centers', action="store_true")
    parser.add_argument('--edges', action="store_true")
    parser.add_argument('--corners', action="store_true")
    parser.add_argument('--E_d2zdt2', type=float, default=5000)
    parser.add_argument('--E_d2z0dx2', type=float, default=0.02)
    parser.add_argument('--E_d3zdx2dt', type=float, default=0.0003)
    parser.add_argument('--data_gap_scale', type=float,  default=2500)
    parser.add_argument('--spring_only', '-s', action="store_true")
    parser.add_argument('--laser_only','-l', action="store_true")
    parser.add_argument('--map_dir','-m', type=str)
    parser.add_argument('--firn_correction','-f', type=str, default=None)
    args=parser.parse_args()


    if args.map_dir is not None:
        h=make_map(glob_str=args.map_dir+'/E*.h5', spacing=args.Width)
        plt.show()
        return h

    args.grid_spacing = [np.float(temp) for temp in args.grid_spacing.split(',')]
    args.time_span = [np.float(temp) for temp in args.time_span.split(',')]

    spacing={'z0':args.grid_spacing[0], 'dz':args.grid_spacing[1], 'dt':args.grid_spacing[2]}
    E_RMS={'d2z0_dx2':args.E_d2z0dx2, 'd3z_dx2dt':args.E_d3zdx2dt, 'd2z_dxdt':args.E_d3zdx2dt*args.data_gap_scale,  'd2z_dt2':args.E_d2zdt2}

    if args.centers:
        reread_dirs=None
        dest_dir = args.base_directory+'/centers'
    if args.edges or args.corners:
        reread_dirs=[args.base_directory+'/centers']
        dest_dir = args.base_directory+'/edges'
    if args.corners:
        reread_dirs += [args.base_directory+'/edges']
        dest_dir = args.base_directory+'/corners'

    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    if args.out_name is None:
        args.out_name=dest_dir + '/E%d_N%d.h5' % (args.xy0[0]/1e3, args.xy0[1]/1e3)

    fit_OIB(args.xy0, Wxy=4e4, E_RMS=E_RMS, t_span=args.time_span, spacing=spacing,  hemisphere=args.Hemisphere, reread_dirs=reread_dirs, \
            out_name=args.out_name, replace=args.Replace, DOPLOT=False, laser_only=args.laser_only, spring_only=args.spring_only, firn_correction=args.firn_correction)

if __name__=='__main__':
    main(sys.argv)

# 0 0 -m /Volumes/ice2/ben/ATL14_test/Jako_d2zdt2=5000_d3z=0.00001_d2zdt2=1500_RACMO -W 4e4
#-80000 -2320000 -W 40000 -t 2002.5 2019.5 -g 200 2000 1  -o /Volumes/ice2/ben/ATL14_test/Jako_d2zdt2=5000_d3z=0.00001_d2zdt2=1500_RACMO/E-80_N-2320.h5 --E_d3zdx2dt 0.00001 --E_d2zdt2 1500 -f RACMO