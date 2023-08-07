#!/usr/bin/env python
# coding: utf-8

# # Live cu-inj-live-impact 

# In[46]:


# Setup directories, and convert dashboard notebook to a script for importing
#!./setup.bash
print("Running LUME IMPACT SERVICE.....")


# In[47]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[48]:


from impact import evaluate_impact_with_distgen, run_impact_with_distgen
from impact.tools import isotime
from impact.evaluate import  default_impact_merit
from impact import Impact

from make_dashboard import make_dashboard
from get_vcc_image import get_live_distgen_xy_dist, VCC_DEVICE_PV

from lcls_live.tools import NpEncoder

import matplotlib as mpl

from pmd_beamphysics.units import e_charge


# In[49]:


import pandas as pd
import numpy as np

import h5py
import json
import epics

import sys
import os
import toml
from time import sleep, time
import datetime


import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('Agg')

# Nicer plotting
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# # Top level config

# In[ ]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", help = "Debug Mode", default = False)
parser.add_argument("-v", "--use_vcc", help = "Use VCC - True When VCC is Active", default = True)
parser.add_argument("-l", "--live", help = "Live Mode -  True When BEAM is Active", default = True)
parser.add_argument("-m", "--model", help = "Mention the Injector Model", default = "sc_inj")
parser.add_argument("-t", "--host", help = "Mention the host", default = "singularity")
parser.add_argument("-p", "--num_procs", help = "Mention the Num Procs", default = 64)


# In[ ]:


def convertStringToBoolean(argument):
    if argument == 'True' or argument == 'true' or argument == True:
        return True
    else:
        return False


# In[52]:


args = vars(parser.parse_args())

DEBUG = convertStringToBoolean(args['debug'])
USE_VCC = convertStringToBoolean(args['use_vcc'])
LIVE = convertStringToBoolean(args['live'])
MODEL = args['model']
HOST = args['host']
NUM_PROCS_ARGS = int(args['num_procs'])

SNAPSHOT = 'examples/sc_inj-snapshot-2022-11-12T12:38:08-08:00.h5'
MIN_CHARGE_pC = 10
config = toml.load(f"configs/{HOST}_{MODEL}.toml")
PREFIX = f'lume-impact-live-demo-{HOST}-{MODEL}'


# In[ ]:


def convertToDatedFormat(destionation_folder):
    curr_date = datetime.date.today()
    year,month,day = curr_date.strftime('%Y'),curr_date.strftime('%m'),curr_date.strftime('%d')
    destionation_folder_dated = destionation_folder + "/" + year + "/" + month + "/" + day

    if not os.path.exists(destionation_folder_dated):
        os.makedirs(destionation_folder_dated)
    
    return destionation_folder_dated


# ## Logging

# In[54]:


import logging
from logging.handlers import RotatingFileHandler

# Gets or creates a logger
logger = logging.getLogger(PREFIX)  

# set log level
logger.setLevel(logging.INFO)

LOG_OUTPUT_DIR = config.get("log_output_dir")
# define file handler and set formatter
file_handler = RotatingFileHandler(f'{LOG_OUTPUT_DIR}/{PREFIX}.log', mode='a', encoding=None, maxBytes=50*1024*1024, 
                                 backupCount=2, delay=0)
formatter    = logging.Formatter(fmt="%(asctime)s :  %(name)s : %(message)s ", datefmt="%Y-%m-%dT%H:%M:%S%z")

# Add print to stdout
logger.addHandler(logging.StreamHandler(sys.stdout))

file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)


# In[ ]:


#Arguments -

logger.info('Start of Script Marker - Script Running with Arguments - ')
logger.info(f'Debug - {DEBUG}')
logger.info(f'USE_VCC - {USE_VCC}')
logger.info(f'LIVE - {LIVE}')
logger.info(f'MODEL - {MODEL}')
logger.info(f'HOST - {HOST}')
logger.info(f'NUM_PROCS_ARGS - {NUM_PROCS_ARGS}')
logger.info(f'Config TOML Loaded - {config}')


# ## Utils

# In[55]:


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
        isotime = h5.attrs['isotime']
        for k in h5:
            v = np.array(h5[k])        
            if v.dtype.char == 'S':
                v = str(v.astype(str))
            pvdata[k] = v
            
    return pvdata, isotime


# # Configuration
# 
# Set up basic input sources and output path, loaded from toml environment file.
# 
# See README for required toml definition.

# In[56]:


HOST = config.get('host') # mcc-simul or 'sdf'
if not HOST:
    raise ValueError("host not defined in toml.")
    
def get_path(key):
    val = config.get(key)
    if not val:
        raise ValueError(f"{key} not defined in toml.")
    val=os.path.expandvars(val)
    if not os.path.exists(val):
        raise ValueError(f"{val} does not exist")
    return os.path.abspath(val)


# Output dirs

SUMMARY_OUTPUT_DIR = get_path('summary_output_dir')
ARCHIVE_DIR = get_path('archive_dir')
SNAPSHOT_DIR = get_path('snapshot_dir')

# Dummy file for distgen
DISTGEN_LASER_FILE = config.get('distgen_laser_file')
if not DISTGEN_LASER_FILE:
    raise ValueError("distgen_laser_file not defined in toml.")

# Number of processors
NUM_PROCS = config.get('num_procs')
if not NUM_PROCS:
    raise ValueError("num_procs not defined in toml.")
else:
    NUM_PROCS = int(NUM_PROCS)

if NUM_PROCS_ARGS != NUM_PROCS:
    NUM_PROCS = NUM_PROCS_ARGS

# if using sdf:
if HOST == 'sdf':    
    #check that environment variables are configured for execution
    IMPACT_COMMAND = config.get("impact_command")
    if not IMPACT_COMMAND:
       raise ValueError("impact_command not defined in toml.")


    IMPACT_COMMAND_MPI = config.get("impact_command_mpi")
    if not IMPACT_COMMAND_MPI:
       raise ValueError("impact_command_mpi not defined in toml.")



# In[57]:


CONFIG0 = {}

# Base settings
SETTINGS0 = {
 'distgen:n_particle': 10_000,   
 'timeout': 10000,
 'header:Nx': 32,
 'header:Ny': 32,
 'header:Nz': 32,
 'numprocs': NUM_PROCS,
   }

SETTINGS0['numprocs'] = NUM_PROCS
CONFIG0["workdir"] = get_path('workdir')

if DEBUG:
    logger.info('DEBUG MODE: Running without space charge for speed. ')
    SETTINGS0['distgen:n_particle'] = 1000
    SETTINGS0['total_charge'] = 0
    
# Host config    
if HOST in ('sdf'):
    
    #SDF setup 
    SETTINGS0['command'] =  IMPACT_COMMAND
    SETTINGS0['command_mpi'] =  IMPACT_COMMAND_MPI
    SETTINGS0['mpi_run'] = config.get("mpi_run_cmd")
    
elif HOST == 'local':
    logger.info('Running locally')
    
else:
    raise ValueError(f'Unknown host: {HOST}')
    


# # Select: LCLS or FACET

# In[59]:


# PV -> Sim conversion table
CSV =  f'pv_mapping/{MODEL}_impact.csv'  

CONFIG0['impact_config']      =  get_path('config_file')
CONFIG0['distgen_input_file'] =  get_path('distgen_input_file')

PLOT_OUTPUT_DIR = get_path('plot_output_dir')

if MODEL == 'cu_inj':
    VCC_DEVICE = 'CAMR:IN20:186' # LCLS   
    
    DASHBOARD_KWARGS = {'outpath':PLOT_OUTPUT_DIR,
                    'screen1': 'YAG02',
                    'screen2': 'YAG03',
                    'screen3': 'OTR2',
                    'ylim' : (0, None), # Emittance scale   
                    'ylim2': (0, None), # sigma_x scale
                    'name' : PREFIX
                   }    
    
    SETTINGS0['stop'] = 16.5
    SETTINGS0['distgen:t_dist:length:value'] =  4 * 1.65   #  Inferred pulse stacker FWHM: 4 ps, converted to tukey length
    
if MODEL == 'sc_inj':
    VCC_DEVICE = 'CAMR:LGUN:950' # LCLS-II 
    
    DASHBOARD_KWARGS = {'outpath':PLOT_OUTPUT_DIR,
                    'screen1': 'YAG01B',
                  #  'screen2': 'BEAM0',
                  #  'screen3': 'OTR0H04',
                    'screen2': 'CM01BEG',
                    'screen3': 'BEAM0',
                    'ylim' : (0, 3e-6), # Emittance scale   
                    'ylim2': (0, None), # sigma_x scale                    
                    'name' : PREFIX
                   }    
    
    SETTINGS0['stop'] = 14 # 28
    SETTINGS0['distgen:t_dist:sigma_t:value'] =  16 / 2.355   # ps, equivalent to 16ps FWHM from Feng
    
elif MODEL == 'f2e_inj':
    VCC_DEVICE = 'CAMR:LT10:900' # FACET-II
    
    DASHBOARD_KWARGS = {'outpath':PLOT_OUTPUT_DIR,
                    'screen1': 'PR10241',
                    'screen2': 'PR10465',
                    'screen3': 'PR10571',
                    'ylim' : (0, 20e-6), # Emittance scale
                    'name' : PREFIX
                   }        
    
    SETTINGS0['distgen:t_dist:length:value'] =  3.65 * 1.65   #  Measured FWHM: 3.65 ps, converted to tukey length
     
else:
    raise


# In[60]:


CONFIG0, SETTINGS0
logger.info(f'FINAL SETTINGS - {SETTINGS0}')


# # Set up monitors

# In[61]:


# Gun: 700 kV
# Buncher: 200 keV energy gain
# Buncher: +60 deg relative to on-crest


# In[62]:


DF = pd.read_csv(CSV)#.dropna()

PVLIST = list(DF['device_pv_name'].dropna()) 

if USE_VCC:
    PVLIST = PVLIST + list(VCC_DEVICE_PV[VCC_DEVICE].values())
else:
    logger.info('USE VCC set to False. VCC is not working right now.')
#DF.set_index('device_pv_name', inplace=True)
DF


# In[63]:


if LIVE:
    MONITOR = {pvname:epics.PV(pvname) for pvname in PVLIST}
    SNAPSHOT = None
    sleep(5)


# In[64]:


def get_snapshot(snapshot_file=None):
        
    if LIVE:
        itime = isotime()
        pvdata =  {k:MONITOR[k].get() for k in MONITOR}
    else:
        pvdata, itime = load_pvdata(snapshot_file)
        itime = itime.decode('utf-8')
    
    logger.info(f'Acquired settings from EPICS at: {itime}')
    
    epics_working_check = [val for val in pvdata.values() if val is None]
    
    if len(epics_working_check) == len(list(pvdata.keys())):
        raise Exception(f'EPICS returned None for all keys. Please check if you are able to connect to Accelerator')

    VCC_Key = None
    
    for k, v in pvdata.items():
        
        if v is None:
            raise ValueError(f'EPICS get for {k} returned None')
        
        if ':IMAGE:ARRAYDATA' in k.upper():
            VCC_Key = k
            found = False
            logger.info(f'Waiting for good {k}')
            counter = 0
            USE_VCC_LOCAL = True
            while not found and counter < 5:
                counter += 1
                if v is None:
                    continue
                if v.std() > 10:
                    found = True
                else:
                    v = MONITOR[k].get()
            if counter == 5:
                logger.info(f'VCC is not working. Defaulting to None.')
                USE_VCC_LOCAL = False
            elif v.ptp() < 128:
                v = v.astype(np.int8) # Downcast preemptively 
            pvdata[k] = v
        else:
            USE_VCC_LOCAL = False

    if not USE_VCC_LOCAL and VCC_Key in pvdata:
        del pvdata[VCC_Key]

    return pvdata, itime, USE_VCC_LOCAL


# # EPICS -> Simulation settings

# In[66]:


def get_settings(csv, base_settings={}, snapshot_dir=None, snapshot_file=None):
    """
    Fetches live settings for all devices in the CSV table, and translates them to simulation inputs
     
    """
    df = DF[DF['device_pv_name'].notna()]
    assert len(df) > 0, 'Empty dataframe!'
    
    pv_names = list(df['device_pv_name'])

    pvdata, itime, USE_VCC_LOCAL = get_snapshot(snapshot_file)
    
    df['pv_value'] = [pvdata[k] for k in pv_names]
    
    # Assign impact
    df['impact_value'] = df['impact_factor']*df['pv_value'] 
    if 'impact_offset' in df:
        df['impact_value'] = df['impact_value']  + df['impact_offset']

    # Collect settings
    settings = base_settings.copy()
    settings.update(dict(zip(df['impact_name'], df['impact_value'])))
    
    if DEBUG:
        settings['total_charge'] = 0
    else:
        settings['total_charge'] = 1 # Will be updated with particles

    # VCC image
    if USE_VCC_LOCAL:
        logger.info('Getting VCC Live Distgen')
        dfile, img, cutimg = get_live_distgen_xy_dist(filename=DISTGEN_LASER_FILE, vcc_device=VCC_DEVICE, pvdata=pvdata)  
        settings['distgen:xy_dist:file'] = dfile
    else:
        img, cutimg = None, None
        #settings['distgen:r_dist:max_r:value'] = 0.35 # TEMP     
        
    if snapshot_dir and not snapshot_file:
        filename = os.path.abspath(os.path.join(snapshot_dir, f'{MODEL}-snapshot-{itime}.h5'))
        total_charge_pC = settings['distgen:total_charge:value']
        if total_charge_pC < MIN_CHARGE_pC:
            logger.info(f'total charge is too low: {total_charge_pC:.2f} pC, not saving snapshot')         
        else:
            save_pvdata(filename, pvdata, itime)
            logger.info(f'EPICS shapshot written: {filename}')
        
        
    return settings, df, img, cutimg, itime


# In[69]:


DO_TIMING = False

if DO_TIMING:
    import numpy as np
    import time
    results = []
    tlist = []
    nlist = 2**np.arange(1,8, 1)[::-1]
    for n in nlist:
        t1 = time.time()
        LIVE_SETTINGS['numprocs'] = n
        print(f'running wit {n}')
        result = run_impact_with_distgen(LIVE_SETTINGS, **CONFIG0, verbose=False )
        results.append(result)
        dt = time.time() - t1
        tlist.append(dt)
        print(n, dt)     
        
    tlist, nlist        


# # Get live values, run Impact-T, make dashboard

# In[70]:


# Patch this into the function below for the dashboard creation
def my_merit(impact_object, itime):
    # Collect standard output statistics
    merit0 = default_impact_merit(impact_object)
    
    PLOT_OUTPUT_DIR_DATED = convertToDatedFormat(PLOT_OUTPUT_DIR)
    #Overriding at runtime to save in dated folders
    DASHBOARD_KWARGS["outpath"] = PLOT_OUTPUT_DIR_DATED
    
    # Make the dashboard from the evaluated object
    plot_file = make_dashboard(impact_object, itime=itime, **DASHBOARD_KWARGS)
    #print('Dashboard written:', plot_file)
    logger.info(f'Dashboard written: {plot_file}')
    
    # Make all readable
    os.chmod(plot_file, 0o644)
    
    # Assign extra info
    merit0['plot_file'] = plot_file    
    merit0['isotime'] = itime
    
    # Clear any buffers
    plt.close('all')

    return merit0


# In[71]:


def run1():
    dat = {}

    SNAPSHOT_DIR_DATED = convertToDatedFormat(SNAPSHOT_DIR)
    ARCHIVE_DIR_DATED = convertToDatedFormat(ARCHIVE_DIR)
    SUMMARY_OUTPUT_DIR_DATED = convertToDatedFormat(SUMMARY_OUTPUT_DIR)
        
    # Acquire settings
    mysettings, df, img, cutimg, itime = get_settings(CSV,
                                                           SETTINGS0,
                                                           snapshot_dir=SNAPSHOT_DIR_DATED,
                                                          snapshot_file=SNAPSHOT)        
    dat['isotime'] = itime
    
    # Record inputs
    dat['inputs'] = mysettings
    dat['config'] = CONFIG0
    dat['pv_mapping_dataframe'] = df.to_dict()
    
    logger.info(f'Running evaluate_impact_with_distgen...')

    t0 = time()
    
    total_charge_pC = mysettings['distgen:total_charge:value']
    if total_charge_pC < MIN_CHARGE_pC:
        logger.info(f'total charge is too low: {total_charge_pC:.2f} pC, skipping')
        return dat
    
    outputs = evaluate_impact_with_distgen(mysettings,
                                       merit_f=lambda x: my_merit(x, itime),
                                       archive_path=ARCHIVE_DIR_DATED,
                                       **CONFIG0, verbose=True )
    
    dat['outputs'] =  outputs   
    logger.info(f'...finished in {(time()-t0)/60:.1f} min')
    fname = fname=f'{SUMMARY_OUTPUT_DIR_DATED}/{PREFIX}-{itime}.json'

    json.dump(dat, open(fname, 'w'), cls=NpEncoder)
    logger.info(f'Summary output written: {fname}')
    return dat
    


# # loop it
# 

# In[78]:


if __name__ == '__main__':
    while True:
        try:
            result = run1()
        except Exception as e:
            logger.info(e)
            if (e.__class__.__name__ == 'Exception'):
                logger.info('Stopping the Program')
                break
            else:
                logger.info('Something BAD happened. Sleeping for 10 s ...')      
                sleep(10)
            


# In[ ]:




