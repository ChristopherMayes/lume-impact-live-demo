#!/usr/bin/env python
# coding: utf-8


from lume_live_demo.vcc_image import vcc_device_pvlist, disgten_image_settings_from_pvdata

from impact import evaluate_impact_with_distgen, run_impact_with_distgen
from impact.tools import isotime
from impact.evaluate import  default_impact_merit
from impact import Impact

from make_dashboard import make_dashboard


from lcls_live.tools import NpEncoder

import matplotlib as mpl

from pmd_beamphysics.units import e_charge




import pandas as pd
import numpy as np

import h5py
import json
import epics

import time
import sys
import os
import toml
import time


import matplotlib.pyplot as plt


import logging
logger = logging.getLogger(__name__)




# Saving and loading
def save_pvdata(filename, pvdata, isotime):
    with h5py.File(filename, 'w') as h5:
        h5.attrs['isotime'] = np.string_(isotime)
        for k, v in pvdata.items():
            if isinstance(v, str):
                v =  np.string_(v)
            h5[k] = v 
def load_pvdata(filename):
    
    if not os.path.exists(filename):
        raise ValueError(f'H5 file does not exist: {filename} ')
    pvdata = {}
    with h5py.File(filename, 'r') as h5:
        isotime = h5.attrs['isotime']#.decode('utf-8')  
        for k in h5:
            v = np.array(h5[k])        
            if v.dtype.char == 'S':
                v = str(v.astype(str))
            pvdata[k] = v
            
    return pvdata, isotime



def get_path(key):
    val = config.get(key)
    if not val:
        raise ValueError(f"{key} not defined in toml.")
    val=os.path.expandvars(val)
    if not os.path.exists(val):
        raise ValueError(f"{val} does not exist")
    return os.path.abspath(val)




def get_saved_snapshot(snapshot_file):
        pvdata, itime = load_pvdata(snapshot_file)
        itime = itime.decode('utf-8')    


def get_live_snapshot(monitor_dict, good_image_std=3):
    """
    Gets a live snapshot from a dict of PV objects (monitors).
    """

    # Filter out image PV names
    image_pvnames = []
    scalar_pvnames = []
    for key in monitor_dict:
        if ':IMAGE:ARRAYDATA' in key.upper():
            image_pvnames.append(key)
        else:
            scalar_pvnames.append(key)

    pvdata = {}        
    for key in image_pvnames:
        found = False
        ii = 0
        while not found:
            ii += 1
            v = monitor_dict[key].get()
            if v is None:
                continue
            if v.std() > good_image_std:
                found = True
            
            if ii % 100 == 0:
                logger.info(f"Waited {ii} times for good {key}")                 
        # Found       
        if v.ptp() < 128:
            v = v.astype(np.int8) # Downcast preemptively                 
        pvdata[key] = v
        
        
    for key in scalar_pvnames:
        v = monitor_dict[key].get()
        if v is None:
            raise ValueError(f'EPICS get for {key} returned None')
        pvdata[key] = v
        
    itime = isotime()
    pvdata =  {k:monitor_dict[k].get() for k in monitor_dict}

    logger.info(f'Acquired settings from EPICS at: {itime}')           
            
    return pvdata, itime



def table_settings_from_pvdata(pvdata, mapping_df):
    
    df = mapping_df[mapping_df['device_pv_name'].notna()]
    
    pv_names = list(df['device_pv_name'])
    
    # Assign values
    df['pv_value'] = [pvdata[k] for k in pv_names]
    
    # Assign impact
    df['impact_value'] = df['impact_factor']*df['pv_value'] 
    if 'impact_offset' in df:
        df['impact_value'] = df['impact_value']  + df['impact_offset']    
    
    # Create settings
    settings = dict(zip(df['impact_name'], df['impact_value']))
    
    return settings


def settings_from_pvdata(pvdata, mapping_df, vcc_device=None, distgen_laser_file=None, fclip=None):
    
    settings = table_settings_from_pvdata(pvdata, mapping_df)
    
    if vcc_device:
        distgen_settings = disgten_image_settings_from_pvdata(pvdata,
                                         vcc_device=vcc_device,
                                        distgen_laser_file=distgen_laser_file,
                                         fclip=fclip
                                        )
        settings.update(distgen_settings)
        
    return settings





# gfile = CONFIG0['distgen_input_file']
# from distgen import Generator
# #fout = res[0]
# G = Generator(gfile)
# #G['xy_dist:file'] =  DISTGEN_LASER_FILE #'distgen_laser.txt'
# if USE_VCC:
#     G['xy_dist:file'] = res[0]['distgen:xy_dist:file'] 
# G['n_particle'] = 100000
# G.run()
# G.particles.plot('x', 'y', figsize=(5,5))




# Patch this into the function below for the dashboard creation
def my_merit(impact_object, itime, dashboard_kwargs={}):
    # Collect standard output statistics
    merit0 = default_impact_merit(impact_object)
    # Make the dashboard from the evaluated object
    plot_file = make_dashboard(impact_object, itime=itime, **dashboard_kwargs)

    logger.info(f'Dashboard written: {plot_file}')
    
    # Make all readable
    os.chmod(plot_file, 0o644)
    
    # Assign extra info
    merit0['plot_file'] = plot_file    
    merit0['isotime'] = itime
    
    # Clear any buffers
    plt.close('all')

    return merit0



def run1(config=None, monitor_dict=None):
    dat = {}
    
    # Summary output 
    summary_out = config['output']['summary_output_dir']
    assert os.path.exists(summary_out)    
    
    # Acquire PV data 
    pvdata, itime = get_live_snapshot(monitor_dict,
                                       good_image_std=config['vcc']['good_image_std']) 
    
    # Save snapshot
    snapshot_dir = config['output']['snapshot_dir']
    modelname = config['model']['name']
    if snapshot_dir:
        assert os.path.exists(snapshot_dir)
        snapshot_filename = os.path.abspath(os.path.join(snapshot_dir, f'{modelname}-snapshot-{itime}.h5'))        
        save_pvdata(snapshot_filename, pvdata, itime)
        logger.info(f'EPICS shapshot written: {snapshot_filename}')
    

    # Map to settings 
    
    mapping_df = pd.read_csv(config['pv_mapping']['CSV'])
    
    settings = config['settings']
    settings.update(config['run_settings'])
    live_settings = settings_from_pvdata(pvdata, mapping_df,
                    vcc_device=config['vcc']['vcc_device'],
                    distgen_laser_file = config['other']['distgen_laser_file'],
                    fclip = config['vcc']['fclip']
                    )      
    settings.update(live_settings)             

    # Record inputs
    dat['isotime'] = itime
    dat['inputs'] = settings
    dat['config'] = config
    dat['pv_mapping_dataframe'] = mapping_df.to_dict()
    
    logger.info(f'Running evaluate_impact_with_distgen...')

    t0 = time.time()
    
    # Validate
    total_charge_pC = settings['distgen:total_charge:value']
    if total_charge_pC < config['other']['min_charge_pC']:
        logger.info(f'total charge is too low: {total_charge_pC:.2f} pC, skipping')
        return dat
    
    
    dashboard_kwargs = config['dashboard_kwargs']
    dashboard_kwargs['outpath'] = config['output']['plot_output_dir']
    
    outputs = evaluate_impact_with_distgen(settings,
                                       merit_f=lambda x: my_merit(x, itime, dashboard_kwargs),
                                       archive_path=config['output']['archive_dir'],
                                       **config['input'], # input files and workdir
                                     verbose=True )
    
    dat['outputs'] =  outputs   
    logger.info(f'...finished in {(time.time()-t0)/60:.1f} min')

    return outputs
    
    fname = fname=f"{summary_out}/{PREFIX}-{itime}.json"

    json.dump(dat, open(fname, 'w'), cls=NpEncoder)
    logger.info(f'Summary output written: {fname}')
    return dat

